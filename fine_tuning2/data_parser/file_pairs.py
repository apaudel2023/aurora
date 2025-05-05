# data_parsing/file_pairs.py

import logging
from pathlib import Path
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

def get_file_pairs(
    dir2d: str,
    dir3d: str,
    prefix2d: str,
    prefix3d: str,
    suffix: str = ".nc",
) -> List[Dict[str,str]]:
    """
    Scan two directories of 2D/3D WRF files, match on the timestamp
    between prefix and suffix, and return a list of dicts:
      [{'2d': '/path/to/wrf2d_…ts.nc', '3d': '/path/to/wrf3d_…ts.nc'}, ...]
    Any files without a matching partner are skipped (with a warning).
    """
    dir2d = Path(dir2d)
    dir3d = Path(dir3d)
    pat2d = re.compile(re.escape(prefix2d) + r"(.+)" + re.escape(suffix))
    pat3d = re.compile(re.escape(prefix3d) + r"(.+)" + re.escape(suffix))

    ts2d, ts3d = {}, {}
    for p in dir2d.glob(f"{prefix2d}*{suffix}"):
        m = pat2d.match(p.name)
        if m:
            ts2d[m.group(1)] = str(p)
    for p in dir3d.glob(f"{prefix3d}*{suffix}"):
        m = pat3d.match(p.name)
        if m:
            ts3d[m.group(1)] = str(p)

    common = sorted(set(ts2d) & set(ts3d))
    only2d = sorted(set(ts2d) - set(ts3d))
    only3d = sorted(set(ts3d) - set(ts2d))

    if only2d:
        logger.warning("Skipping 2D-only timestamps: %s", only2d)
    if only3d:
        logger.warning("Skipping 3D-only timestamps: %s", only3d)

    return [{"2d": ts2d[t], "3d": ts3d[t]} for t in common]
