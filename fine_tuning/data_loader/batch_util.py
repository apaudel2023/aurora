# fine_tuning/data_loader/batch_util.py

import numpy as np
import torch
from datetime import datetime
from aurora import Batch, Metadata

def make_aurora_batch(meta, surf, static, atom):
    """
    From fully preprocessed dicts, build an aurora.Batch exactly like the example,
    converting the raw two‐step time array into a one‐element datetime tuple.
    Assumes:
      meta['lat'] (H,), meta['lon'] (W,), meta['time'] (array-like len=2),
      meta['pressure_levels'] (tuple len=C),
      surf['t2','u10','v10','psfc'] each (2,H,W),
      static['z'] (H,W),
      atom['z','t','u','v','q'] each (2,C,H,W).
    """

    # Convert raw meta["time"] (2-length array of byte/str) to a single datetime tuple
    raw_times = meta["time"]
    # pick the current (second) time step
    t_raw = raw_times[-1]
    if isinstance(t_raw, (bytes, bytearray)):
        t_str = t_raw.decode("utf-8")
    else:
        t_str = str(t_raw)
    t_str = t_str.replace("_", " ")
    # to numpy datetime64[s], then to Python datetime
    t_np = np.datetime64(t_str).astype("datetime64[s]")
    t_py = t_np.tolist()
    time_tuple = (t_py,)

    return Batch(
        surf_vars={
            "2t":  torch.from_numpy(surf["t2"][None]),
            "10u": torch.from_numpy(surf["u10"][None]),
            "10v": torch.from_numpy(surf["v10"][None]),
            "msl": torch.from_numpy(surf["psfc"][None]),
        },
        static_vars={
            "z": torch.from_numpy(static["z"]),
        },
        atmos_vars={
            "t": torch.from_numpy(atom["t"][None]),
            "u": torch.from_numpy(atom["u"][None]),
            "v": torch.from_numpy(atom["v"][None]),
            "q": torch.from_numpy(atom["q"][None]),
            "z": torch.from_numpy(atom["z"][None]),
        },
        metadata=Metadata(
            lat          = torch.from_numpy(np.ascontiguousarray(meta["lat"], dtype=np.float32)),
            lon          = torch.from_numpy(np.ascontiguousarray(meta["lon"], dtype=np.float32)),
            time         = time_tuple,
            atmos_levels = meta["pressure_levels"],
        ),
    )
