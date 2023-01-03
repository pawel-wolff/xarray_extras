import numpy as np
import xarray as xr


def is_coord_regularly_gridded(coord, abs_err=None):
    """
    Checks if a coordinate variable is regularly gridded (spaced)
    :param coord: a 1-dim array-like
    :param abs_err: float or timedelta; maximal allowed absolute error when checking for equal spaces
    :return: bool
    """
    coord = np.asanyarray(coord)
    if len(coord.shape) != 1:
        raise ValueError(f'coord must be 1-dimensional')
    n, = coord.shape
    if n < 2:
        return True
    d_coord = np.diff(coord)
    if abs_err is None:
        try:
            eps = np.finfo(coord.dtype).eps
        except ValueError:
            # dtype not inexact
            return np.all(d_coord == d_coord[0])
        # floating-point
        abs_err = 4 * eps * np.max(np.fabs(coord))
    return np.all(np.fabs(d_coord - d_coord[0]) <= 2 * abs_err)


@xr.register_dataset_accessor('regrid')
@xr.register_dataarray_accessor('regrid')
class regrid:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def regrid(self, target_coords, method='mean', tolerance=None, skipna=None, keep_attrs=False,
               keep_ori_dims_order=True, keep_ori_coords=True, **agg_method_kwargs):
        """
        Regrid coordinates of a dataset (or a data array). Coordinates are assumed to be a center of a grid cell.
        Coarser grids are obtained from regular aggregation; to this end, both the initial and target grids must
        each be equally spaced and the target grid must be more coarsed than the initial one. If method is 'linear'
        or 'nearest', a simple re-sampling via interpolation is applied.

        :param self: an xarray Dataset or DataArray
        :param target_coords: dict with keys being dimensions of ds to regrid and values being new coordinates in a form of
        an array-like object
        :param method: 'mean', 'sum', 'max', 'min', etc.
        (see http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
        for an exhaustive list), or 'linear' or 'nearest' for re-sampling
        :param tolerance: float or numpy.timedelta64 or None; default=None; a tolerance for checking coordinates alignment; if None, no check is applied
        :param skipna: bool, optional; default behaviour is to skip NA values if they are of float type; for more see
        http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
        :param keep_attrs: bool, optional; If True, the attributes (attrs) will be copied from the original object
        to the new one. If False (default), the new object will be returned without attributes.
        :param keep_ori_dims_order: if True, keep the dataset/dataarray original order of dimensions;
        if False, impose the dimensions order as the order of targer_coords dict.
        :param keep_ori_coords: if True, keep the dataset/dataarray original coordinate variables (as auxiliary coordinates), otherwise drop them
        :param agg_method_kwargs: keyword arguments passed to an aggregation method; see
        http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
        :return: same type as ds
        """
        ds = self._obj
        # check if target dimensions are contained in dimensions of ds
        ds_dims = ds.dims
        for coord_label in target_coords:
            if coord_label not in ds_dims:
                raise ValueError(f'{coord_label} not found among ds dimensions: {list(ds_dims)}')

        if keep_ori_dims_order:
            target_coords = {coord_label: target_coords[coord_label] for coord_label in ds_dims if coord_label in target_coords}

        ds = ds.xrx.make_coordinates_increasing(target_coords.keys())

        if method in ['linear', 'nearest']:
            interpolator_kwargs = {'fill_value': 'extrapolate'} if method == 'nearest' else None
            regridded_ds = ds.interp(coords=target_coords, method=method, assume_sorted=True, kwargs=interpolator_kwargs)
        else:
            # check if target coordinates are equally spaced
            for coord_label, target_coord in target_coords.items():
                if not is_coord_regularly_gridded(target_coord):
                    raise ValueError(f'{coord_label} is not be regularly gridded: {target_coord}')
            # check if coordinates of ds are equally spaced
            for coord_label in target_coords:
                if not is_coord_regularly_gridded(ds[coord_label]):
                    raise ValueError(f'ds has {coord_label} coordinate not regularly gridded: {ds[coord_label]}')

            # trim the domain of ds to target_coords, if necessary
            trim_by_dim = {}
            for coord_label, target_coord in target_coords.items():
                target_coord = np.asanyarray(target_coord)
                n_target_coord, = target_coord.shape
                target_coord_min, target_coord_max = target_coord.min(), target_coord.max()
                if n_target_coord >= 2:
                    step = (target_coord_max - target_coord_min) / (n_target_coord - 1)
                    if ds[coord_label].min() < target_coord_min - step / 2 or \
                            ds[coord_label].max() > target_coord_max + step / 2:
                        trim_by_dim[coord_label] = slice(target_coord_min - step / 2, target_coord_max + step / 2)
            if trim_by_dim:
                ds = ds.sel(trim_by_dim)

            # coarsen
            window_size = {}
            for coord_label, target_coord in target_coords.items():
                if len(ds[coord_label]) % len(target_coord) != 0:
                    raise ValueError(f'resolution of {coord_label} not compatible: '
                                     f'{len(ds[coord_label])} must be a multiple of {len(target_coord)}\n'
                                     f'ds[{coord_label}] = {ds[coord_label]}\n'
                                     f'target_coord = {target_coord}')
                window_size_for_coord = len(ds[coord_label]) // len(target_coord)
                if window_size_for_coord > 1:
                    window_size[coord_label] = window_size_for_coord
            coarsen_ds = ds.coarsen(dim=window_size, boundary='exact', coord_func='mean')
            coarsen_ds_agg_method = getattr(coarsen_ds, method)
            if skipna is not None:
                agg_method_kwargs['skipna'] = skipna
            if keep_attrs is not None:
                agg_method_kwargs['keep_attrs'] = keep_attrs
            regridded_ds = coarsen_ds_agg_method(**agg_method_kwargs)

            # adjust coordinates of regridded_ds so that they fit to target_coords
            # necessary to map existing coordinates to new coordinates (e.g. x_=x, etc),
            # or to rearrange increasing coords to decreasing (or vice-versa)
            # and if it has no essential effect, it is a cheap operation
            try:
                regridded_ds = regridded_ds.sel(target_coords, method='nearest', tolerance=tolerance)
            except KeyError:
                raise ValueError(f"target grid is not compatible with a source grid; "
                                 f"check grids or adjust 'tolerance' parameter\n"
                                 f"regridded_ds={regridded_ds}\n"
                                 f"target_coords={target_coord}")
            # overwrite coordinates
            regridded_ds = regridded_ds.assign_coords({r_dim: target_coords[r_dim]
                                                       for r_dim in set(regridded_ds.dims).intersection(target_coords)})
        # drop the dataset/dataaray original coordinate variables which became auxiliary coordinates after regridding
        if not keep_ori_coords:
            ori_coords_became_aux_coords = [coord for coord in regridded_ds.coords
                                            if coord in target_coords and coord not in regridded_ds.dims]
            regridded_ds = regridded_ds.reset_coords(ori_coords_became_aux_coords, drop=True)
        return regridded_ds
