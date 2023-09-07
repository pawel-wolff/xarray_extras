import xarray as xr


def _normalize_longitude_ufunc(arr, smallest_lon_coord=-180.):
    return (arr - smallest_lon_coord) % 360. + smallest_lon_coord


@xr.register_dataset_accessor('geo')
@xr.register_dataarray_accessor('geo')
class geo:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def get_lon_label(self):
        ds = self._obj
        ds_dims = ds.dims
        if 'longitude' in ds_dims:
            label = 'longitude'
        elif 'lon' in ds_dims:
            label = 'lon'
        else:
            raise ValueError('neither "longitude" nor "lon" dimension found in ds')
        return label

    def get_lat_label(self):
        ds = self._obj
        ds_dims = ds.dims
        if 'latitude' in ds_dims:
            label = 'latitude'
        elif 'lat' in ds_dims:
            label = 'lat'
        else:
            raise ValueError('neither "latitude" nor "lat" dimension found in ds')
        return label

    def get_lon_lat_label(self):
        ds = self._obj
        return ds.geo.get_lon_label(), ds.geo.get_lat_label()

    def normalize_longitude(self, lon_label=None, smallest_lon_coord=-180., keep_attrs=False):
        ds = self._obj
        if lon_label is None:
            lon_label = ds.geo.get_lon_label()
        lon_coords = ds[lon_label]
        aligned_lon_coords = _normalize_longitude_ufunc(lon_coords, smallest_lon_coord=smallest_lon_coord)
        if keep_attrs:
            aligned_lon_coords = aligned_lon_coords.assign_attrs(lon_coords.attrs)

        if not lon_coords.equals(aligned_lon_coords):
            old_lon_coords_monotonic = lon_coords.indexes[lon_label].is_monotonic_increasing
            ds = ds.assign_coords({lon_label: aligned_lon_coords})
            if old_lon_coords_monotonic:
                smallest_lon_idx = aligned_lon_coords.argmin(dim=lon_label).item()
                ds = ds.roll(shifts={lon_label: -smallest_lon_idx}, roll_coords=True)
            else:
                ds = ds.sortby(lon_label)
        return ds
