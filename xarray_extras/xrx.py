import numpy as np
import xarray as xr
import pathlib

from common.log import logger


@xr.register_dataset_accessor('xrx')
@xr.register_dataarray_accessor('xrx')
class xrx:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def short_dataset_repr(self):
        ds = self._obj
        if isinstance(ds, xr.Dataset):
            ds_id = ds.attrs.get('id')
            ds_vars = list(ds)
            ds_repr = f'Dataset(id={ds_id}, vars={ds_vars})'
        elif isinstance(ds, xr.DataArray):
            ds_name = ds.name
            for attr in ['standard_name', 'short_name', 'name', 'long_name']:
                if ds_name is not None:
                    break
                ds_name = ds.attrs.get(attr)
            ds_repr = f'DataArray({ds_name})'
        else:
            ds_repr = str(ds)
        return ds_repr

    def dropna_and_flatten(self):
        da = self._obj
        # alternative method for many variables:
        # ds.where(ds[v] > 0).to_dataframe().dropna(how='all', subset=list(ds)).reset_index()
        if not isinstance(da, xr.DataArray):
            raise TypeError(f'da must be an xarray.DataArray; got {type(da)}')
        # TODO: make it lazy (values -> data);
        # but then a lack of explicit sizes of idxs is a problem when constructing an xarray Dataset
        idxs = np.nonzero(da.notnull().values)
        idxs = (('idx', idx) for idx in idxs)
        idxs = xr.Dataset(dict(zip(da.dims, idxs)))
        return da.isel(idxs)

    def make_coordinates_increasing(self, coord_labels, allow_sorting=True):
        """
        Sorts coordinates
        :param self: an xarray Dataset or DataArray
        :param coord_labels: a string or an interable of strings - labels of dataset's coordinates
        :param allow_sorting: bool; default True; indicate if sortby method is allowed to be used as a last resort
        :return: ds with a chosen coordinate(s) in increasing order
        """
        ds = self._obj
        if isinstance(coord_labels, str):
            coord_labels = (coord_labels, )
        for coord_label in coord_labels:
            if not ds.indexes[coord_label].is_monotonic_increasing:
                if ds.indexes[coord_label].is_monotonic_decreasing:
                    ds = ds.isel({coord_label: slice(None, None, -1)})
                elif allow_sorting:
                    ds = ds.sortby(coord_label)
                else:
                    raise ValueError(f'{ds.xrx.short_dataset_repr()} has coordinate {coord_label} which is neither increasing nor decreasing')
        return ds

    def drop_duplicated_coord(self, dim):
        ds = self._obj
        _, idx = np.unique(ds[dim], return_index=True)
        if len(idx) != len(ds[dim]):
            ds = ds.isel({dim: idx})
        return ds


def open_dataset_from_netcdf_with_disk_chunks(url, chunks='auto', **kwargs):
    """
    Open a dataset from a netCDF file using on disk chunking. The parameter 'chunks' mimics xarray.open_zarr behaviour.
    :param url: str; path to a netCDF file.
    :param chunks: 'auto' or dict or None; default 'auto'. If 'auto', open with chunks on disk; if a dictionary is given,
    it must map dimensions into chunks sizes (size -1 means the whole dimension length); the dictionary updates chunk
    sizes found in the file; if None, open without chunking.
    :param kwargs: extra keyword arguments passed to xarray.open_dataset
    :return: xarray Dataset
    """
    if chunks is not None:
        with xr.open_dataset(url) as ds:
            chunks_by_dim = {}
            dims = ds.dims
            for v in list(ds.data_vars) + list(ds.coords):
                if v not in dims:
                    v_dims = ds[v].sizes
                    v_chunks = ds[v].encoding.get('chunksizes')
                    if v_chunks is not None:
                        if len(v_dims) == len(v_chunks):
                            for dim, chunk in zip(v_dims, v_chunks):
                                chunks_by_dim.setdefault(dim, {})[v] = chunk
                        else:
                            logger().warning(f'variable {v}: sizes={v_dims}, chunks={v_chunks}; '
                                             f'ignoring the chunks specification for this variable')
        chunk_by_dim = {}
        for d, chunks_by_var in chunks_by_dim.items():
            chunk = min(chunks_by_var.values())
            if chunk < dims[d]:
                chunk_by_dim[d] = chunk
        if chunks != 'auto':
            chunk_by_dim.update(chunks)
        if all(chunk_by_dim[d] in (dims[d], -1) for d in chunk_by_dim):
            chunk_by_dim = None
        ds = xr.open_dataset(url, chunks=chunk_by_dim, **kwargs)
    else:
        ds = xr.open_dataset(url, **kwargs)
    return ds


def open_dataset_with_disk_chunks(url, chunks='auto', **kwargs):
    """
    Open a dataset from a netCDF or zarr file using on disk chunking.
    The parameter 'chunks' mimics xarray.open_zarr behaviour.
    :param url: str; path to a netCDF or zarr file.
    :param chunks: 'auto' or dict or None; default 'auto'. If 'auto', open with chunks on disk; if a dictionary is given,
    it must map dimensions into chunks sizes (size -1 means the whole dimension length); the dictionary updates chunk
    sizes found in the file; if None, open without chunking.
    :param kwargs: extra keyword arguments passed to xarray.open_dataset
    :return: xarray Dataset
    """
    fmt = pathlib.PurePath(url).suffix
    if fmt == '.nc':
        return open_dataset_from_netcdf_with_disk_chunks(url, chunks=chunks, **kwargs)
    elif fmt == '.zarr':
        return xr.open_zarr(url, chunks=chunks, **kwargs)
    else:
        raise ValueError(f'unknown format: {fmt}; must be .nc or .zarr')
