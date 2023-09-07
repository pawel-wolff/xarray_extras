import numpy as np
import xarray as xr


@xr.register_dataset_accessor('geo_regrid')
@xr.register_dataarray_accessor('geo_regrid')
class geo_regrid:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def regrid_lon_lat(self, target_resol_ds=None, longitude=None, latitude=None, method='mean', tolerance=None,
                       longitude_circular=None, skipna=None, keep_attrs=False, **agg_method_kwargs):
        """
        Regrid longitude and latitude coordinates of a dataset (or a data array). Coordinates are assumed to be a center
        of a grid cell. Coarser grids are obtained from regular aggregation; to this end, both the initial and target grids
        must each be equally spaced and the target grid must be more coarsed than the initial one. If method is 'linear'
        or 'nearest', a simple re-sampling via interpolation is applied.

        :param self: an xarray Dataset or DataArray
        :param target_resol_ds: an xarray Dataset with new longitude and latitude coordinates;
        alternatively (and exclusively) longitude and latitude parameters can be given
        :param longitude: an array-like; optional
        :param latitude: an array-like; optional
        :param method: 'mean', 'sum', 'max', 'min', etc.
        (see http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
        for an exhaustive list), or 'linear' or 'nearest' for re-sampling
        :param tolerance: float or None, default=None; a tolerance for checking coordinates alignment; if None, no check is applied
        :param longitude_circular: bool, optional; if True then then longitude coordinates are considered as circular; if None, automatic check if applied
        :param skipna: bool, optional; default behaviour is to skip NA values if they are of float type; for more see
        http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
        :param keep_attrs: bool, optional; If True, the attributes (attrs) will be copied from the original object
        to the new one. If False (default), the new object will be returned without attributes.
        :param agg_method_kwargs: keyword arguments passed to an aggregation method; see
        http://xarray.pydata.org/en/stable/generated/xarray.core.rolling.DatasetCoarsen.html#xarray.core.rolling.DatasetCoarsen
        :return: same type as ds
        """
        ds = self._obj
        # prepare target longitude and latitude coordinates
        if target_resol_ds is None and (longitude is None or latitude is None):
            raise ValueError('target_resol_ds or longitude and latitude must be set')
        if target_resol_ds is not None and (longitude is not None or latitude is not None):
            raise ValueError('either target_resol_ds or longitude and latitude must be set, but not both')
        if target_resol_ds is not None:
            lon, lat = target_resol_ds.geo.get_lon_lat_label()
            longitude = target_resol_ds[lon]
            latitude = target_resol_ds[lat]
        # get labels of longitude and latitude dimension for ds
        lon_label, lat_label = ds.geo.get_lon_lat_label()
        ds_lon = ds[lon_label]
        ds_lon_span = abs(ds_lon[-1] - ds_lon[0])

        # check if longitude is circular
        if longitude_circular is None and len(longitude) >= 2:
            eps = np.finfo(ds_lon.dtype).eps
            ds_lon_delta = abs(ds_lon[1] - ds_lon[0])
            longitude_circular = bool(abs(ds_lon_span - 360.) <= 8. * 360. * eps or
                                      abs(ds_lon_span + ds_lon_delta - 360.) <= 8. * 360. * eps)
            # print(f'longitude_circular={longitude_circular} for ds={ds.xrx.short_dataset_repr()}')  # DEBUG INFO

        # handle overlapping target longitude coordinates if necessary
        longitude_ori = None
        if longitude_circular:
            # remove target longitude coordinate which is overlapping mod 360
            eps = np.finfo(longitude.dtype).eps
            if abs(abs(longitude[-1] - longitude[0]) - 360.) <= 8. * 360. * eps:
                longitude_ori = longitude
                longitude = longitude[:-1]
            # remove ds' longitude coordinate which is overlapping mod 360
            eps = np.finfo(ds_lon.dtype).eps
            if abs(ds_lon_span - 360.) <= 8. * 360. * eps:
                ds = ds.isel({lon_label: slice(None, -1)})

        # if necessary, align longitude coordinates of ds to the target longitude by normalizing and rolling or sorting
        smallest_longitude_coord = (np.amin(np.asanyarray(longitude)) + np.amax(np.asanyarray(longitude)) - 360.) / 2
        ds = ds.geo.normalize_longitude(lon_label=lon_label, smallest_lon_coord=smallest_longitude_coord, keep_attrs=True)
        ds_lon = ds[lon_label]

        # duplicate left- and right-most longitude coordinate to facilitate interpolation, if necessary
        if method in ['linear', 'nearest'] and longitude_circular:
            lon_coord = ds_lon.values
            extended_lon_coord = np.concatenate(([lon_coord[-1]], lon_coord, [lon_coord[0]]))
            extended_lon_coord_normalized = np.array(extended_lon_coord)
            extended_lon_coord_normalized[0] = extended_lon_coord_normalized[0] - 360.
            extended_lon_coord_normalized[-1] = extended_lon_coord_normalized[-1] + 360.
            lon_attrs = ds_lon.attrs
            ds = ds\
                .sel({lon_label: extended_lon_coord})\
                .assign_coords({lon_label: extended_lon_coord_normalized})
            ds[lon_label].attrs = lon_attrs

        # do regridding
        ds = ds.regrid.regrid({lon_label: longitude, lat_label: latitude}, method=method, tolerance=tolerance,
                              skipna=skipna, keep_attrs=keep_attrs, keep_ori_coords=False, **agg_method_kwargs)

        # re-establish longitude coordinate which overlaps mod 360
        if longitude_circular and longitude_ori is not None:
            lon_idx = np.arange(len(longitude_ori))
            lon_idx[-1] = 0
            ds = ds.isel({lon_label: lon_idx}).assign_coords({lon_label: longitude_ori})

        return ds
