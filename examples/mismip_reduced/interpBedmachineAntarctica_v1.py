import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import netCDF4

# Try importing ISSM's InterpFromGrid
try:
    from InterpFromGridToMesh import InterpFromGridToMesh 
except ImportError:
    InterpFromGridToMesh = None


def interpBedmachineAntarctica(
    X: np.ndarray,
    Y: np.ndarray,
    string: str = "bed",
    method: Optional[str] = None,
    ncdate: Union[str, Path, None] = None
) -> np.ndarray:
    """
    Python port of Mathieu Morlighem's interpBedmachineAntarctica.m
    Uses ISSM's InterpFromGrid if available.

    Parameters
    ----------
    X, Y : ndarray
        Coordinates in polar stereographic meters.
    string : {"bed", "surface", "thickness", "mask", "source"}, default "bed"
        Variable to interpolate.
    method : {"linear", "cubic", "nearest"}, optional
        Interpolation method; defaults as in MATLAB version.
    ncdate : str or Path, optional
        Either dataset date tag (e.g., "2020-07-15", "v3.5") or full path
        to a NetCDF file.

    Returns
    -------
    ndarray
        Interpolated field matching the shape of X/Y.
    """
    # Defaults
    if method is None:
        if string.lower() in ("mask", "source"):
            method = "nearest"
        else:
            method = "cubic"

    # Default dataset version if not provided
    if ncdate is None:
        ncdate = "v3"  # Official v3 release

    basename = "BedMachineAntarctica"

    ncfile = None

    # Locate the NetCDF file
    if Path(str(ncdate)).is_file():  # explicit file path
        ncfile = str(ncdate)
    else:
        search_paths = [
            f"/u/astrid-r1b/ModelData/BedMachine/{basename}-{ncdate}.nc",
            f"/home/ModelData/Antarctica/BedMachine/{basename}-{ncdate}.nc",
            f"/totten_1/ModelData/Antarctica/BedMachine/{basename}-{ncdate}.nc",
            f"/local/ModelData/BedMachineAntarctica/{basename}-{ncdate}.nc",
            f"/Users/larour/ModelData/BedMachine/{basename}-{ncdate}.nc",
            f"/Users/u7322062/Downloads/{basename}-{ncdate}.nc",
        ]
        for p in search_paths:
            if Path(p).is_file():
                ncfile = p
                break
        else:
            raise FileNotFoundError(
                f"Could not find {basename}-{ncdate}.nc – "
                f"add the path to the list or provide it as ncdate argument."
            )

    print(f"   -- BedMachine Antarctica version: {ncdate}")

    with netCDF4.Dataset(ncfile, "r") as ds:
        xfull = ds.variables["x"][:].astype(np.float64)
        yfull = ds.variables["y"][:].astype(np.float64)

        # Normalize to ascending for searchsorted (we'll map back)
        x_asc = True
        y_asc = True
        if xfull[0] > xfull[-1]:
            xfull = xfull[::-1]; x_asc = False
        if yfull[0] > yfull[-1]:
            yfull = yfull[::-1]; y_asc = False

        offset = 2
        xmin, xmax = float(np.min(X)), float(np.max(X))
        ymin, ymax = float(np.min(Y)), float(np.max(Y))

        # If totally outside, fail early with a clear message
        if xmax < xfull[0] or xmin > xfull[-1] or ymax < yfull[0] or ymin > yfull[-1]:
            raise ValueError(
                "Query domain lies outside BedMachine extent:\n"
                f"  x in [{xmin:.1f}, {xmax:.1f}] vs grid [{xfull[0]:.1f}, {xfull[-1]:.1f}]\n"
                f"  y in [{ymin:.1f}, {ymax:.1f}] vs grid [{yfull[0]:.1f}, {yfull[-1]:.1f}]"
            )

        # Compute inclusive index window on ASCENDING arrays
        id1x = max(0, np.searchsorted(xfull, xmin, side="left") - offset)
        id2x = min(len(xfull) - 1, np.searchsorted(xfull, xmax, side="right") + offset - 1)
        id1y = max(0, np.searchsorted(yfull, ymin, side="left") - offset)
        id2y = min(len(yfull) - 1, np.searchsorted(yfull, ymax, side="right") + offset - 1)

        # Ensure at least 2 samples per axis
        if id2x - id1x + 1 < 2:
            if id2x < len(xfull) - 1: id2x += 1
            elif id1x > 0:            id1x -= 1
        if id2y - id1y + 1 < 2:
            if id2y < len(yfull) - 1: id2y += 1
            elif id1y > 0:            id1y -= 1

        # Slices in the (now-ascending) logical space
        x_slice = slice(id1x, id2x + 1)
        y_slice = slice(id1y, id2y + 1)

        # Read from file (BedMachine stores as [x, y]); transpose to (y, x)
        print(f"   -- BedMachine Antarctica: loading {string}")
        var = ds.variables[string]
        dims = tuple(var.dimensions)
        if dims == ("x", "y"):
            # file order is (x,y) → transpose to (y,x)
            data = var[id1x:id2x+1, id1y:id2y+1].astype(np.float64).T
        elif dims == ("y", "x"):
            # file order already (y,x)
            data = var[id1y:id2y+1, id1x:id2x+1].astype(np.float64)
        else:
            raise RuntimeError(f"Unexpected dims for '{string}': {dims}")
        xdata = xfull[x_slice].copy()
        ydata = yfull[y_slice].copy()

    # If original file had descending axes, flip DATA back to ASCENDING x/y
    # so that (xdata ascending, ydata ascending, data.shape == (len(ydata), len(xdata)))
    # This ensures InterpFromGridToMesh gets strictly increasing vectors.
    # (We already used ascending xfull/yfull to compute indices, so just keep ascending.)
    # If the read came from descending originals, the transpose above made (y,x) in "current" order;
    # we need to flip data rows/cols to align with ascending xdata/ydata:
    if not x_asc:
        data = data[:, ::-1]
    if not y_asc:
        data = data[::-1, :]

    # Special mask handling
    if string.lower() == "mask":
        data[data == 3] = 0

    # Sanity checks (leave in while debugging)
    assert xdata.ndim == 1 and ydata.ndim == 1
    assert data.shape == (len(ydata), len(xdata)), (data.shape, len(ydata), len(xdata))
    if InterpFromGridToMesh is not None:
        xv = _as_c_double(xdata)           # 1D
        yv = _as_c_double(ydata)           # 1D
        Z  = _as_c_double(data)            # 2D (rows=y, cols=x)

        xq = _as_c_double(X).ravel()       # 1D
        yq = _as_c_double(Y).ravel()       # 1D

        # Old 6-arg ISSM API: (x, y, Z, xq, yq, method)
        try:
            res = InterpFromGridToMesh(xv, yv, Z, xq, yq, method)
        except TypeError:
            # Some builds only accept integer method codes
            method_map = {"nearest": 0, "linear": 1, "cubic": 2}
            res = InterpFromGridToMesh(xv, yv, Z, xq, yq, method_map[method.lower()])

        Zi = res[0] if isinstance(res, tuple) else res
        out = np.asarray(Zi, dtype=np.float64).reshape(X.shape)
    else:
        out = FastInterp(xdata, ydata, data, X, Y, method)

    return out




def FastInterp(x, y, data, xi, yi, method):
    """
    Pure Python equivalent of the MATLAB FastInterp function.
    """
    M, N = data.shape
    ndx = 1.0 / (x[1] - x[0])
    ndy = 1.0 / (y[1] - y[0])

    xi = (xi - x[0]) * ndx
    yi = (yi - y[0]) * ndy

    zi = np.full(xi.shape, np.nan, dtype=float)

    if method.lower() == "nearest":
        rxi = np.rint(xi).astype(int) + 1
        ryi = np.rint(yi).astype(int) + 1
        flag = (
            (rxi > 0) & (rxi <= N) &
            (ryi > 0) & (ryi <= M)
        )
        ind = ryi - 1 + M * (rxi - 1)
        zi[flag] = data.T.flatten()[ind[flag]]

    else:  # bilinear
        fxi = np.floor(xi).astype(int) + 1
        fyi = np.floor(yi).astype(int) + 1
        dfxi = xi - fxi + 1
        dfyi = yi - fyi + 1

        flagIn = (
            (fxi > 0) & (fxi < N) &
            (fyi > 0) & (fyi < M)
        )

        fxi = fxi[flagIn]
        fyi = fyi[flagIn]
        dfxi = dfxi[flagIn]
        dfyi = dfyi[flagIn]

        ind1 = fyi - 1 + M * (fxi - 1)
        ind2 = fyi - 1 + M * fxi
        ind3 = fyi + M * fxi
        ind4 = fyi + M * (fxi - 1)

        flat_data = data.T.flatten()
        zi[flagIn] = (
            flat_data[ind1] * (1 - dfxi) * (1 - dfyi) +
            flat_data[ind2] * dfxi * (1 - dfyi) +
            flat_data[ind4] * (1 - dfxi) * dfyi +
            flat_data[ind3] * dfxi * dfyi
        )

    return zi

def _as_c_double(a):
    a = np.asarray(a, dtype=np.float64, order="C")
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a, dtype=np.float64)
    return a