# pipeline.py

import logging
import os
from pathlib import Path
from typing import List, Dict, Tuple
import yaml
import numpy as np

from fine_tuning2.data_parser.static_reader import StaticReader
from fine_tuning2.data_parser.wrf2d_reader import WRF2DReader
from fine_tuning2.data_parser.wrf3d_reader import WRF3DReader
from fine_tuning2.data_parser.file_pairs import get_file_pairs


class Pipeline:
    """
    Aurora data pipeline orchestrator.
    Encapsulates configuration, logging, and helper methods for:
     - static loading
     - 2D time & surface reading
     - 3D atmospheric processing
     - automatic discovery and pairing of 2D/3D files
    """

    def __init__(self, config_path: str):
        # Load config
        self.config_path = Path(config_path)
        self.cfg = self._load_config(self.config_path)

        # Setup logging
        self._setup_logging(self.cfg.get('logging', {}))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initialized pipeline with config: %s', self.config_path)

        # Static reader config
        static_cfg = self.cfg.get('static', {})
        self.static_path    = static_cfg['path']
        self.static_engine  = static_cfg.get('engine', 'netcdf4')
        self.static_decode  = static_cfg.get('decode_times', False)
        self.static_var_map = static_cfg.get('var_map', None)

        # 2D reader config
        wrf2d_cfg = self.cfg.get('wrf2d', {})
        self.wrf2d_kwargs = {
            'var_map':      wrf2d_cfg.get('var_map'),
            'engine':       wrf2d_cfg.get('engine', 'netcdf4'),
            'decode_times': wrf2d_cfg.get('decode_times', False),
        }

        # 3D reader config
        wrf3d_cfg = self.cfg.get('wrf3d', {})
        self.wrf3d_kwargs = {
            'var_map':      wrf3d_cfg.get('var_map'),
            'engine':       wrf3d_cfg.get('engine', 'netcdf4'),
            'decode_times': wrf3d_cfg.get('decode_times', False),
        }

        # Target pressure levels
        self.target_levels = self.cfg.get('target_levels', [])

        # Discover and pair files automatically
        data_dirs = self.cfg.get('data_dirs', {})
        fprefs   = self.cfg.get('filename_prefixes', {})
        suffix   = self.cfg.get('filename_suffix', '.nc')
        self.file_pairs = get_file_pairs(
            dir2d    = data_dirs['wrf2d_dir'],
            dir3d    = data_dirs['wrf3d_dir'],
            prefix2d = fprefs['wrf2d'],
            prefix3d = fprefs['wrf3d'],
            suffix   = suffix
        )
        self.logger.info('Discovered %d matching file pairs', len(self.file_pairs))

        # Load lat/lon once from the first 2D file
        if not self.file_pairs:
            raise RuntimeError("No file pairs found; cannot load lat/lon metadata")
        first_2d = self.file_pairs[0]['2d']
        reader2d = WRF2DReader(path=first_2d, **self.wrf2d_kwargs)
        base_meta = reader2d.load_metadata()
        self.lat = base_meta['lat']   # shape (south_north,)
        self.lon = base_meta['lon']   # shape (west_east,)
        self.logger.info("Loaded lat/lon shapes %s, %s", self.lat.shape, self.lon.shape)

    @staticmethod
    def _setup_logging(log_cfg: Dict):
        """Configure root logger to file and console."""
        log_file   = log_cfg.get('log_file', 'aurora_pipeline.log')
        level_name = log_cfg.get('level', 'INFO').upper()
        level      = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def _load_config(path: Path) -> Dict:
        """Read YAML config file."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_static(self) -> Dict[str, np.ndarray]:
        """Helper: load static constants."""
        self.logger.info('Loading static constants')
        reader = StaticReader(
            path=self.static_path,
            engine=self.static_engine,
            decode_times=self.static_decode,
            var_map=self.static_var_map,
        )
        static_vars = reader.load()
        self.logger.debug('Static shapes: %s', {k: v.shape for k, v in static_vars.items()})
        return static_vars

    def _load_2d(self, path2d: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Helper: load time and surface from a single 2D file.
        Returns:
          time_arr: 1D array of time stamps (often length 1)
          surf:     dict of surface vars shaped (ntime, H, W)
        """
        self.logger.info('Loading 2D data from %s', path2d)
        reader2d = WRF2DReader(path=path2d, **self.wrf2d_kwargs)
        md       = reader2d.load_metadata()
        surf     = reader2d.load_surface()
        time_arr = md['time']
        self.logger.debug('2D time shape: %s, surf shapes: %s',
                          time_arr.shape,
                          {k: v.shape for k, v in surf.items()})
        return time_arr, surf

    def _load_3d(self, path3d: str) -> Dict[str, np.ndarray]:
        """Helper: load and adjust atmospheric from a 3D file."""
        self.logger.info('Loading 3D data from %s', path3d)
        reader3d = WRF3DReader(path=path3d, **self.wrf3d_kwargs)
        atmos    = reader3d.load_and_adjust(self.target_levels)
        self.logger.debug('3D atmos shapes: %s', {k: v.shape for k, v in atmos.items()})
        return atmos

    def run(self) -> List[Tuple[
        Dict[str, np.ndarray],  # metadata
        Dict[str, np.ndarray],  # surf_vars
        Dict[str, np.ndarray],  # static_vars
        Dict[str, np.ndarray],  # atmos_vars
    ]]:
        """Run pipeline over all file pairs."""
        static_vars = self._load_static()
        results     = []

        for idx, pair in enumerate(self.file_pairs):
            path2d, path3d = pair['2d'], pair['3d']
            self.logger.info('Processing pair %d: 2D=%s, 3D=%s',
                             idx, path2d, path3d)

            # Load 2D: get time and surface
            time_arr, surf = self._load_2d(path2d)
            time_pt = time_arr[0]   # first (only) timestamp

            # Build metadata dict
            metadata = {
                'lat':         self.lat,
                'lon':         self.lon,
                'time':        (time_pt,),                  # length-1 tuple
                'atmos_levels': tuple(self.target_levels),  # from config
            }

            # Load 3D: atmospheric variables
            atmos = self._load_3d(path3d)

            results.append((metadata, surf, static_vars, atmos))
            self.logger.info('Completed pair %d', idx)

        self.logger.info('Pipeline finished processing %d pairs', len(self.file_pairs))
        return results


if __name__ == '__main__':
    config_path = '/home/user/Documents/aurora/aurora_foked/aurora/fine_tuning2/cfg/configs.yml'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    pipeline = Pipeline(config_path)
    pipeline.run()
