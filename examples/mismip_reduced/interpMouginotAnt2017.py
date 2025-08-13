import socket
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator


def _interp_from_grid(xg, yg, data, X, Y, **kwargs):
    """
    Vectorised bilinear interpolation on a regular (x,y) grid.

    Parameters
    ----------
    xg, yg : 1-D numpy arrays
        Grid coordinates (must be monotonic).
    data   : 2-D numpy array, shape (len(yg), len(xg))
        Field values on the grid: rows vary in y, columns in x.
    X, Y   : ndarray
        Target coordinates (same shape).
    kwargs : forwarded to RegularGridInterpolator (e.g. fill_value).

    Returns
    -------
    ndarray, same shape as X/Y
    """
    interp = RegularGridInterpolator((yg, xg), data,
                                     bounds_error=False, fill_value=np.nan,
                                     **kwargs)
    pts = np.column_stack((Y.ravel(), X.ravel()))  # (y,x)
    return interp(pts).reshape(X.shape)


def interp_mouginot_ant2017(X, Y, mag_only=False):
    """
    Interpolate Mouginot et al. (2017/19) Antarctic surface velocity
    onto arbitrary (x,y) coordinates.

    Parameters
    ----------
    X, Y     : ndarray
        Coordinates in Mouginot polar-stereographic metres (same shape).
    mag_only : bool, optional
        If True, return speed magnitude instead of (vx, vy).

    Returns
    -------
    vx, vy           if mag_only is False
    speed (|v|)      if mag_only is True
    """
    host = socket.gethostname()

    if host == "ronne":
        nc_path = "/home/ModelData/Antarctica/MouginotVel/vel_nsidc.CF16_2.nc"
    elif host == "totten":
        nc_path = "/totten_1/ModelData/Antarctica/MouginotVel/vel_nsidc.CF16_2.nc"
    elif host == "amundsen.thayer.dartmouth.edu":
        nc_path = "/local/ModelData/AntarcticVelocity/v_mix.v13Mar2019.nc"
    elif host == "nri-085597":
        nc_path = "/Users/u7322062/Downloads/antarctica_ice_velocity_450m_v2.nc"
    else:
        raise RuntimeError(f"hostname '{host}' not supported yet")

    with Dataset(nc_path) as ds:
        xdata = ds.variables["x"][:].astype(float)
        ydata = ds.variables["y"][:].astype(float)

        x_asc = xdata[0] <= xdata[-1]
        y_asc = ydata[0] <= ydata[-1]

        # ------------------------------------------------------------
        # Trim grid to the bounding box of (X,Y) plus a small padding
        # ------------------------------------------------------------
        offset = 2  # cells
        xmin, xmax = np.min(X), np.max(X)
        ymin, ymax = np.min(Y), np.max(Y)

        id1x = max(0, np.searchsorted(xdata, xmin) - offset)
        id2x = min(len(xdata) - 1,
                   np.searchsorted(xdata, xmax, side="right") + offset - 1)

        id1y = max(0, np.searchsorted(ydata, ymin) - offset)
        id2y = min(len(ydata) - 1,
                   np.searchsorted(ydata, ymax, side="right") + offset - 1)
        
        vx = ds.variables["VX"]
        vy = ds.variables["VY"]
        Yq = -np.asarray(Y) 
        # Load the cropped tiles and transpose so (y,x) order
        vx_tile = vx[id1y:id2y+1, id1x:id2x+1].astype(float)
        vy_tile = vy[id1y:id2y+1, id1x:id2x+1].astype(float)

        xsub = xdata[id1x:id2x + 1]
        ysub = ydata[id1y:id2y + 1]

        if not x_asc: vx_tile = vx_tile[:, ::-1]; vy_tile = vy_tile[:, ::-1]
        if not y_asc: vx_tile = vx_tile[::-1, :]; vy_tile = vy_tile[::-1, :]

    # ------------------------------------------------------------
    # Interpolation (bilinear, like InterpFromGrid in MATLAB)
    # ------------------------------------------------------------
    vx_out = _interp_from_grid(xsub, ysub, vx_tile, np.asarray(X), Yq)
    vy_out = _interp_from_grid(xsub, ysub, vy_tile, np.asarray(X), Yq)

    if mag_only:           # mimic MATLAB’s “one output → speed”
        return np.hypot(vx_out, vy_out)
    return vx_out, vy_out
