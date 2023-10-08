import warnings
import numpy as np
import pandas as pd
import xarray as xr
import pathlib


@xr.register_dataset_accessor('xrx')
@xr.register_dataarray_accessor('xrx')
class xrx:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def iterate(self, dim, drop=False):
        ds = self._obj
        assert dim in ds.dims, f'{dim} is not a dimension'
        for i, coord in enumerate(ds[dim].values):
            yield coord, ds.isel({dim: i}, drop=drop)

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

    def dropna_and_flatten(self, how='all', subset=None):
        """
        Flatten to a 1-dim DataArray or Dataset (indexed by 'idx') and drop NaN's. Dimensional coordinates become
        non-dimensional coordinates indexed by 'idx'.
        :param how: str; if 'all' then drop an index if all variables are NaN;
        if 'any' then drop an index if any of the variables is NaN; has no effect if self is a DataArray
        :param subset: str or list of str; in case self is a Dataset, restrict checking for NaN's to this subset of variables
        :return: same type as self
        """
        ds = self._obj
        if subset is not None:
            if not isinstance(subset, (list, tuple)):
                subset = [subset]

        # alternative method for many variables:
        # ds.where(ds[v] > 0).to_dataframe().dropna(how='all', subset=list(ds)).reset_index()
        # if not isinstance(da, xr.DataArray):
        #     raise TypeError(f'da must be an xarray.DataArray; got {type(da)}')
        # TODO: make it lazy (values -> data);
        #  but then a lack of explicit sizes of idxs is a problem when constructing an xarray Dataset
        ds_notnull = ds.notnull()
        da_notnull = None
        if not isinstance(ds_notnull, xr.Dataset):
            da_notnull = ds_notnull
        else:
            if subset is not None:
                ds_notnull = ds_notnull[subset]
            for da in ds_notnull.values():
                if da_notnull is None:
                    da_notnull = da
                else:
                    if how == 'all':
                        da_notnull = da_notnull & da
                    elif how == 'any':
                        da_notnull = da_notnull | da
                    else:
                        raise ValueError(f'how must be all or any; got how={how}')

        idxs = np.nonzero(da_notnull.values)
        idxs = (('idx', idx) for idx in idxs)
        idxs = xr.Dataset(dict(zip(da_notnull.dims, idxs)))
        return ds.isel(idxs)

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

    def assign_dummy_coords(self, dims=None):
        ds = self._obj
        if dims is None:
            ds_coords = list(ds.coords)
            dims_with_dummy_coords = [dim for dim in ds.dims if dim not in ds_coords]
        else:
            dims_with_dummy_coords = dims
        if dims_with_dummy_coords:
            dummy_coords_by_dim = {dim: np.arange(ds.sizes[dim]) for dim in dims_with_dummy_coords}
            ds = ds.assign_coords(dummy_coords_by_dim)
        return ds

    # TODO: test if it works for DataArray
    def flexible_unstack(self, dim, dict_of_lists_of_coords):
        """
        Example of usage: flexible_unstack(ds, '__composed_idx', {'flight_id_and_airport_code': ['idx', 'airport_code'], 'air_press_AC_binned': None})
        :param ds:
        :param dim:
        :param dict_of_lists_of_coords:
        :return:
        """
        def _multi_idx_to_simple_idx(multi_idx, sample_idx_name):
            multi_idx_unique = multi_idx.unique()
            simple_idx_unique = pd.Series(
                np.arange(len(multi_idx_unique)),
                index=multi_idx_unique,
                name=sample_idx_name
            )
            simple_idx = simple_idx_unique.loc[multi_idx].values
            coords_df = simple_idx_unique.reset_index().set_index(sample_idx_name)
            return simple_idx, coords_df

        ds = self._obj
        assert len(dict_of_lists_of_coords) >= 1
        coords_by_label = {}
        dims_to_convert_to_multi_idx = []
        for new_dim, list_of_coords in dict_of_lists_of_coords.items():
            if not list_of_coords or len(list_of_coords) == 1 and list_of_coords[0] == new_dim:
                continue
            elif len(list_of_coords) == 1 and list_of_coords[0] != new_dim:
                raise NotImplementedError
            elif len(list_of_coords) >= 2:
                multi_idx = pd.MultiIndex.from_arrays([ds[c].values for c in list_of_coords], names=list_of_coords)
                simple_idx, _coord_df = _multi_idx_to_simple_idx(multi_idx, new_dim)
                coords_by_label.update(dict(_coord_df))
                ds = ds.reset_coords(names=list_of_coords, drop=True).assign_coords({new_dim: (dim, simple_idx)})
                dims_to_convert_to_multi_idx.append(new_dim)
            else:
                raise RuntimeError
        new_dims = list(dict_of_lists_of_coords)
        compound_idx = pd.MultiIndex.from_arrays([ds[new_dim].values for new_dim in new_dims], names=new_dims)
        ds = ds. \
            reset_coords(names=new_dims, drop=True). \
            assign_coords({dim: compound_idx}). \
            unstack(dim). \
            assign_coords(coords_by_label)
        for new_dim in dims_to_convert_to_multi_idx:
            list_of_coords = dict_of_lists_of_coords[new_dim]
            multi_idx = pd.MultiIndex.from_arrays([ds[c].values for c in list_of_coords], names=list_of_coords)
            ds = ds.assign_coords({new_dim: multi_idx})
        return ds


def get_dataset_dims_chunks_sizes_itemsize(url, **kwargs):
    with xr.open_dataset(url, **kwargs) as ds:
        chunks_by_v = {}
        sizes_by_v = {}
        itemsize_by_v = {}
        dims = dict(ds.dims)
        vs = list(ds.data_vars) + list(ds.coords)
        for v in vs:
            if v not in dims:
                da = ds[v]
                v_dims = da.sizes
                sizes_by_v[v] = dict(v_dims)
                itemsize_by_v[v] = da.dtype.itemsize
                v_chunks = da.encoding.get('chunksizes')
                if v_chunks is not None:
                    if len(v_dims) == len(v_chunks):
                        chunks_by_v[v] = dict(zip(v_dims, v_chunks))
                    else:
                        warnings.warn(
                            f'variable {v}: sizes={v_dims}, chunks={v_chunks}; '
                            f'ignoring the chunks specification for this variable'
                        )
    return dims, chunks_by_v, sizes_by_v, itemsize_by_v


def open_dataset_from_netcdf_with_disk_chunks(url, chunks='auto', max_chunk_size=None, **kwargs):
    """
    Open a dataset from a netCDF file using on disk chunking. The parameter 'chunks' mimics xarray.open_zarr behaviour.
    :param url: str; path to a netCDF file.
    :param chunks: 'auto' or dict or None; default 'auto'. If 'auto', open with chunks on disk; if a dictionary is given,
    it must map dimensions into chunks sizes (size -1 means the whole dimension length); the dictionary updates chunk
    sizes found in the file; if None, open without chunking.
    :param max_chunk_size: int; when chunks is 'auto', determines chunking coarser than disk chunking
    :param kwargs: extra keyword arguments passed to xarray.open_dataset
    :return: xarray Dataset
    """
    if chunks is not None:
        dims, chunks_by_v, sizes_by_v, itemsize_by_v = get_dataset_dims_chunks_sizes_itemsize(url, **kwargs)
        vs = list(sizes_by_v)

        chunks_by_dim = {d: {} for d in dims}
        for v, chunks_for_v in chunks_by_v.items():
            for d, chunk_for_v in chunks_for_v.items():
                chunks_by_dim[d][v] = chunk_for_v

        if isinstance(chunks, dict):
            for d, chunks_for_dim in chunks_by_dim.items():
                if d in chunks:
                    chunk = chunks[d] if chunks[d] != -1 else dims[d]
                    for v in list(chunks_for_dim):
                        chunks_for_dim[v] = chunk

        if max_chunk_size is not None:
            # possibly coarsen chunks_by_dim so that max_chunk_size is satisfied
            chunks_by_v = {}
            for v in vs:
                if v not in dims:
                    chunks_by_v[v] = dict(sizes_by_v[v])
                    chunks_by_v[v].update(
                        {
                            d: chunks_for_d[v]
                            for d, chunks_for_d in chunks_by_dim.items()
                            if v in chunks_for_d
                        }
                    )

            for v, chunks_by_dim_for_v in chunks_by_v.items():
                v_dims = sizes_by_v[v]
                v_itemsize = itemsize_by_v[v]
                for d, size in reversed(list(v_dims.items())):
                    if not isinstance(chunks, dict) or d not in chunks:
                        chunk_size = np.prod(list(chunks_by_dim_for_v.values())) * v_itemsize
                        mul = max_chunk_size / chunk_size
                        if size / chunks_by_dim_for_v[d] <= mul:
                            chunks_by_dim_for_v[d] = size
                        elif mul >= 2:
                            chunks_by_dim_for_v[d] = min(chunks_by_dim_for_v[d] * int(mul), size)
                        chunks_by_dim[d][v] = chunks_by_dim_for_v[d]

        chunk_by_dim = {}
        for d, chunks_by_var in chunks_by_dim.items():
            chunk = min(chunks_by_var.values())
            if chunk < dims[d]:
                chunk_by_dim[d] = chunk

        if all(chunk_by_dim[d] in (dims[d], -1) for d in chunk_by_dim):
            chunk_by_dim = None
        ds = xr.open_dataset(url, chunks=chunk_by_dim, **kwargs)
    else:
        ds = xr.open_dataset(url, **kwargs)
    return ds


def open_dataset_with_disk_chunks(url, chunks='auto', max_chunk_size=None, **kwargs):
    """
    Open a dataset from a netCDF or zarr file using on disk chunking.
    The parameter 'chunks' mimics xarray.open_zarr behaviour.
    :param url: str; path to a netCDF or zarr file.
    :param chunks: 'auto' or dict or None; default 'auto'. If 'auto', open with chunks on disk, or coarser,
    if max_chunk_size is given (see below); if a dictionary is given, it must map dimensions into chunks sizes
    (size -1 means the whole dimension length); the dictionary updates chunk sizes found in the file;
    if None, open without chunking.
    :param max_chunk_size: int; when chunks is 'auto', determines chunking coarser than disk chunking
    :param kwargs: extra keyword arguments passed to xarray.open_dataset
    :return: xarray Dataset
    """
    fmt = pathlib.PurePath(url).suffix
    if fmt == '.nc':
        return open_dataset_from_netcdf_with_disk_chunks(url, chunks=chunks, max_chunk_size=max_chunk_size, **kwargs)
    elif fmt == '.zarr':
        if max_chunk_size is not None:
            warnings.warn('max_chunk_size is not supported for zarr and thus ignored')
        return xr.open_zarr(url, chunks=chunks, **kwargs)
    else:
        raise ValueError(f'unknown format: {fmt}; must be .nc or .zarr')


def concat_from_nested_dict(dict_of_ds, dims, **concat_kwargs):
    if len(dims) == 0:
        return dict_of_ds
    dim, *other_dims = dims
    dict_of_ds_items = list(dict_of_ds.items())
    if len(dict_of_ds_items) == 0:
        raise ValueError(f'a (nested) dictionary for dim={dim} cannot be empty')
    coords, nested_dicts_of_ds = zip(*dict_of_ds.items())
    dss = [
        concat_from_nested_dict(nested_dict_of_ds, other_dims, **concat_kwargs)
        for nested_dict_of_ds in nested_dicts_of_ds
    ]
    return xr.concat(dss, dim=pd.Index(coords, name=dim), **concat_kwargs)
