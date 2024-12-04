"""File: xdas_loaders.py | Author: agarcon | date: Fri Oct 04 2024

Description: Two xdas-comptatible loaders.

Usages:
    # Single file xdas comptability
    single_file = read_surfo(".data/surfo_1.hdf5")
    # Multiple files xdas comptability (virtual source)
    stacked_files = xdas_dataarray(file_wildcard=".data/surfo_*.hdf5")
"""

import logging

import h5py
import numpy as np
import xdas
from xdas import DataArray, DataMapping, DataSequence
from xdas.virtual import VirtualSource

logger = logging.getLogger("Loaders")

T_ = DataArray | DataSequence | DataMapping


def read_surfo(fname: str) -> DataArray:
    """Read a single hdf surfo file and returns a DataArray."""
    with h5py.File(fname, "r") as file:
        t0 = np.datetime64(file.attrs["t_zero"])
        dt = np.timedelta64(int(file.attrs["dt"]), "ns")
        dx = file.attrs["dx"]
        data = VirtualSource(file["data"])
    nt, nd = data.shape

    t = {"tie_indices": [0, nt - 1], "tie_values": [t0, t0 + (nt - 1) * dt]}
    d = {"tie_indices": [0, nd - 1], "tie_values": [0.0, (nd - 1) * dx]}

    metadata = {"t_zero": t0, "dt": dt, "dx": dx}
    return DataArray(data, {"time": t, "distance": d}, attrs=metadata)


def xdas_dataarray(file_wildcard: str) -> T_:
    """Create an xdas data array over multiple files.

    Args:
        file_wildcard (str): a wildcard of your hdf5 files.
         e.g. "data/surfo_data_*.hdf5".

    Returns:
        DataArray: an xdas iterable object over multiple files
    """
    try:
        data_array = xdas.open_mfdataarray(
            paths=file_wildcard,  # path with wildcard
            engine=read_surfo,
            tolerance=None,  # by default, no tolerance is used to merge the files
        )
    except FileNotFoundError as e:
        msg = f"{file_wildcard} was not found: {e}"
        logger.exception(msg)
        raise
    except Exception as e:
        msg = f"Unknown exception while loading hdf5 to xdas array: {e}"
        logger.exception(msg)
        raise
    return data_array
