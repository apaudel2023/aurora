# fine_tuning2/pipeline.py

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml
import numpy as np

import torch
torch.backends.cuda.enable_flash_sdp(False)

from fine_tuning2.data_parser.static_reader import StaticReader
from fine_tuning2.data_parser.wrf2d_reader import WRF2DReader
from fine_tuning2.data_parser.wrf3d_reader import WRF3DReader
from fine_tuning2.data_parser.file_pairs import get_file_pairs
from fine_tuning2.data_prep.lat_lon_normalize import LatLonNormalizer
from fine_tuning2.data_prep.crop_grid import GridCropper

from fine_tuning2.batch.get_batch import BuildBatch
from fine_tuning2.model.get_model import AuroraModel

import pdb

class Pipeline:
    """
    Aurora data pipeline orchestrator.
    Encapsulates configuration, logging, and sequential steps:
      1) Load static constants
      2) Pair & load each 2D/3D file
      3) Build a Batch of history frames
      4) Run the model rollout (prediction_steps)
    """

    def __init__(self, config_path: str):
        # Load config
        self.config_path = Path(config_path)
        self.cfg = self._load_config(self.config_path)

        # Setup logging
        self._setup_logging(self.cfg.get("logging", {}))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized pipeline with config %s", self.config_path)

        # --- Static reader ---
        static_cfg = self.cfg["static"]
        self.static_reader = StaticReader(
            path=static_cfg["path"],
            engine=static_cfg.get("engine", "netcdf4"),
            decode_times=static_cfg.get("decode_times", False),
            var_map=static_cfg.get("var_map")
        )

        # --- WRF readers config ---
        wrf2d_cfg = self.cfg["wrf2d"]
        self.wrf2d_kwargs = {
            "var_map":      wrf2d_cfg["var_map"],
            "engine":       wrf2d_cfg.get("engine", "netcdf4"),
            "decode_times": wrf2d_cfg.get("decode_times", False),
        }

        wrf3d_cfg = self.cfg["wrf3d"]
        self.wrf3d_kwargs = {
            "var_map":      wrf3d_cfg["var_map"],
            "engine":       wrf3d_cfg.get("engine", "netcdf4"),
            "decode_times": wrf3d_cfg.get("decode_times", False),
        }

        # Target pressure levels
        self.target_levels = self.cfg["target_levels"]

        # --- File pairing ---
        dd = self.cfg["data_dirs"]
        fp = self.cfg["filename_prefixes"]
        suffix = self.cfg.get("filename_suffix", ".nc")
        self.file_pairs = get_file_pairs(
            dir2d    = dd["wrf2d_dir"],
            dir3d    = dd["wrf3d_dir"],
            prefix2d = fp["wrf2d"],
            prefix3d = fp["wrf3d"],
            suffix   = suffix
        )
        if not self.file_pairs:
            raise RuntimeError("No matching 2D/3D file pairs found")
        self.logger.info("Discovered %d file pairs", len(self.file_pairs))


        # ------------- Preprocessing -------------
        prep_cfg = self.cfg.get("preprocessing", {})
        patch_size = prep_cfg.get("patch_size", 4)
        self.normer  = LatLonNormalizer()
        self.cropper = GridCropper(patch_size=patch_size)
        self.logger.info("Preprocessor: patch_size=%d", patch_size)

        # --- Batch & Model configs ---
        batch_cfg      = self.cfg["batch"]
        self.history   = batch_cfg["history"]
        self.start_idx = batch_cfg["start_index"]

        model_cfg = self.cfg["model"]
        self.prediction_steps = model_cfg.get("prediction_steps", 1)

        # Instantiate BuildBatch
        self.batcher = BuildBatch(
            history     = self.history,
            start_index = self.start_idx
        )

        # Instantiate AuroraModel
        self.model = AuroraModel(
            model_type=model_cfg["type"],
            use_lora=model_cfg.get("use_lora", False),
            device=model_cfg.get("device", "cpu")
        )
        self.logger.info("Using model on device %s", self.model.device)

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _setup_logging(log_cfg: Dict[str, Any]):
        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            handlers=[
                logging.FileHandler(log_cfg.get("log_file", "pipeline.log"), mode="w"),
                logging.StreamHandler()
            ]
        )

    def _load_static(self) -> Dict[str, np.ndarray]:
        self.logger.info("Loading static constants")
        static_vars = self.static_reader.load()
        self.logger.debug("Static shapes: %s", {k: v.shape for k, v in static_vars.items()})
        return static_vars

    def _load_2d(self, path2d: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Returns:
          metadata: {'lat':(H,), 'lon':(W,), 'time':(1,)}
          surf:     {'t2','u10','v10','psfc'} each (1,H,W)
        """
        self.logger.info("Loading 2D from %s", path2d)
        reader = WRF2DReader(path=path2d, **self.wrf2d_kwargs)
        meta = reader.load_metadata()
        surf = reader.load_surface()
        self.logger.debug("2D meta: time=%s, surf_shapes=%s",
                          meta["time"].shape,
                          {k: v.shape for k, v in surf.items()})
        return meta, surf

    def _load_3d(self, path3d: str) -> Dict[str, np.ndarray]:
        self.logger.info("Loading 3D data from %s", path3d)
        reader3d = WRF3DReader(path=path3d, **self.wrf3d_kwargs)
        atmos = reader3d.load_and_adjust(self.target_levels)
        self.logger.debug("3D atmos shapes %s", {k: v.shape for k, v in atmos.items()})
        return atmos

    def run(self):
        # 1) Load static constants once
        static_vars = self._load_static()

        # 2) Collect per‑pair outputs
        surf_list, atmos_list, time_list = [], [], []
        for idx, (p2d, p3d) in enumerate([(p["2d"], p["3d"]) for p in self.file_pairs]):
            self.logger.info("Pair %d: %s / %s", idx, p2d, p3d)

            meta2d, surf = self._load_2d(p2d)
            surf_list.append(surf)
            time_list.append(meta2d["time"][0])

            atmos = self._load_3d(p3d)
            atmos_list.append(atmos)

            self.logger.info("Completed pair %d", idx)

        N = len(time_list)
        self.logger.info("Collected %d time points", N)

        # 3) Concatenate surf & atmos along time (axis=0)
        surf_full = {
            key: np.concatenate([d[key] for d in surf_list], axis=0)
            for key in surf_list[0]
        }
        atmos_full = {
            key: np.concatenate([d[key] for d in atmos_list], axis=0)
            for key in atmos_list[0]
        }

        # 4) Build the metadata dict
        lat = meta2d["lat"]
        lon = meta2d["lon"]
        metadata = {
            "lat":          lat,
            "lon":          lon,
            "time":         tuple(time_list),
            "atmos_levels": tuple(self.target_levels),
        }

        self.logger.info("After concatenation (%d time steps):", len(metadata["time"]))
        self.logger.info("  Surface variables:")
        for name, arr in surf_full.items():
            self.logger.info("    %-6s --> %s", name, arr.shape)
        self.logger.info("  Atmospheric variables:")
        for name, arr in atmos_full.items():
            self.logger.info("    %-6s --> %s", name, arr.shape)


        # 5) PREPROCESSING
        # 5a) normalize lat/lon
        lat360, lon360 = self.normer.normalize(metadata["lat"], metadata["lon"])
        metadata["lat"], metadata["lon"] = lat360, lon360
        self.logger.info("Normalized lat --> %s, lon --> %s",
                         metadata["lat"].shape, metadata["lon"].shape)

        # 5b) Crop spatial dims 
        self.logger.info("Cropping all spatial variables to patch_size=%d", self.cropper.patch)

        # Surface
        self.logger.info("  Surface variables:")
        for name, arr in surf_full.items():
            cropped, rh, rw = self.cropper.crop_grid(arr)
            surf_full[name] = cropped
            self.logger.info("    %-6s: %16s --> %-16s   (-%2d rows, -%2d cols)",
                             name, str(arr.shape), str(cropped.shape), rh, rw)

        # Atmospheric
        self.logger.info("  Atmospheric variables:")
        for name, arr in atmos_full.items():
            cropped, rh, rw = self.cropper.crop_grid(arr)
            atmos_full[name] = cropped
            self.logger.info("    %-6s: %16s --> %-16s   (-%2d rows, -%2d cols)",
                             name, str(arr.shape), str(cropped.shape), rh, rw)

        # Static
        self.logger.info("  Static variables:")
        for name, arr in static_vars.items():
            cropped, rh, rw = self.cropper.crop_grid(arr)
            static_vars[name] = cropped
            self.logger.info("    %-6s: %16s --> %-16s   (-%2d rows, -%2d cols)",
                             name, str(arr.shape), str(cropped.shape), rh, rw)

        # 5c) Crop lat/lon now that crop_grid stored last removal internally
        lat_c, lon_c = self.cropper.crop_latlon(metadata["lat"], metadata["lon"])
        metadata["lat"], metadata["lon"] = lat_c, lon_c
        self.logger.info("Cropped lat --> %s, lon --> %s", lat_c.shape, lon_c.shape)


        # 6) Build batch
        batch = self.batcher.make(
            metadata   = metadata,
            surf_full  = surf_full,
            static     = static_vars,
            atmos_full = atmos_full
        )

        self.logger.info("Built Batch:")
        # surf_vars: each is (B, history, H, W)
        for name, tensor in batch.surf_vars.items():
            self.logger.info("  Surf '%s': %s", name, tuple(tensor.shape))
        # atmos_vars: each is (B, history, levels, H, W)
        for name, tensor in batch.atmos_vars.items():
            self.logger.info("  Atmos '%s': %s", name, tuple(tensor.shape))
        # static_vars: each is (B, H, W)
        for name, tensor in batch.static_vars.items():
            self.logger.info("  Static '%s': %s", name, tuple(tensor.shape))
        # metadata:
        self.logger.info("  Metadata.time length: %d", len(batch.metadata.time))

        pdb.set_trace()



        # 7a) Free up and reset
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.model.device)

        # 7b) Log pre‐rollout memory stats
        allocated = torch.cuda.memory_allocated(self.model.device) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.model.device) / (1024**3)
        self.logger.info(
            "Pre-rollout GPU Mem (GB) — allocated: %.2f, reserved: %.2f",
            allocated, reserved
        )

        # 8) Begin rollout
        self.logger.info("Beginning rollout: %d steps on %s",
                        self.prediction_steps, self.model.device)
        preds = self.model.predict(batch, steps=self.prediction_steps)

        # 9) Log peak after rollout
        peak_alloc = torch.cuda.max_memory_allocated(self.model.device) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(self.model.device) / (1024**3)
        self.logger.info(
            "Post-rollout Peak GPU Mem (GB) — alloc: %.2f, reserved: %.2f",
            peak_alloc, peak_reserved
        )

        return preds




if __name__ == '__main__':
    config_path = '/home/user/Documents/aurora/aurora_foked/aurora/fine_tuning2/cfg/configs.yml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    pipeline = Pipeline(config_path)
    pipeline.run()
