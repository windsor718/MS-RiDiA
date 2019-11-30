import numpy as np
import h5py
import camavec

"""
sets of functions frequently used in the pyletkf context.
may be included in pyletkf package in future release.
"""


def load_data3d(dpath, nt, nlat, nlon, map2vec, nvec,
                dtype=np.float32, memmap=True):
    """
    load binary datasets from dpath [nt, nlat, nlon]

    Args:
        dpath (str): path to data
        nt (int): first dimension value
        dtype (np object): dtype, default 4byte real
        memmap (bool): True to use np.memmap backend to load data.
            provides efficient IO with ignorable perfomace loss.
            effective for large datasets.
    Returns:
        numpy array-like object: [nt, nlat*nlon]
                                    1d vectorized map for nt layers.
    Notes:
        this function will be called everytime an assimilation happens
        to load/save data. If this overhead of IO time is unignorable
        consider to use memmap=True.

        when using memmap, make sure that your data shape is correct.
        Namely, shape=nt*nlat*nlon. If shape > actual data length,
        memmap throws error. But if shape < actual data length, then
        memmap reads part of data without any warning,
        which is problematic.
    """
    if memmap:
        # data shape must be correct,
        # otherwise memmap will only load part of the data without warnings
        data = np.memmap(dpath, dtype=dtype, mode="r",
                         shape=(nt, nlat, nlon), order="C")
    else:
        # not using out-of-core memory mapping, load whole data onto memory
        # this will benefit slight performance gain for array manipulation
        # but takes time if your data is large to load on.
        data = np.fromfile(dpath, dtype=dtype).reshape(nt, nlat, nlon)
    vecdata = vectorize_map(data, map2vec, nvec)
    return vecdata


def define_state_vector(keys, datapaths):
    """
    Define what will be included as a state vector

    Args:
        keys (list): state vector variable names.
        datapaths (str): path to that variable file.

    Returns:
        dict: state vector description dictionary
    """
    infodict = dict()
    for key, dpath in zip(keys, datapaths):
        infodict["key"] = dpath
    return infodict


def get_logn_perturbation(mean, std, eTot):
    """
    returns sampled values fron lognormal dist.

    Args:
        mean (float): mean of underlying normal distribution
        std (float): std of underlying normal distribution
        eTot (float): number of ensemble members

    Returns:
        ndarray: [eTot,] sampled values
    """
    gains = np.random.lognormal(mean, std, size=eTot)
    return gains


def get_unifm_perturbation(self, low, high, eTot):
    """
    returns sampled values fron lognormal dist.

    Args:
        low (float): lowest value
        high (float): highest value
        eTot (float): number of ensemble members

    Returns:
        ndarray: [eTot,] sampled values
    """
    gains = np.random.uniform(low, high, size=eTot)
    return gains


def get_normal_perturbation(mean, std, eTot):
    """
    returns sampled values fron lognormal dist.

    Args:
        mean (float): mean of underlying normal distribution
        std (float): std of underlying normal distribution
        eTot (float): number of ensemble members

    Returns:
        ndarray: [eTot,] sampled values
    """
    gains = np.random.normal(mean, std, size=eTot)
    return gains


def load_cached_patches(cachepath):
    """
    load pre-cached local patches in hdf5 format.

    Args:
        cachepath (str): path to the cached hdf5 file

    Returns:
        list: local patches vectorized

    Notes:
        this pre-cache process is implemented in pyletkf/exTools.py.
        exTools.constLocalPatches_nextxy() is tailored for CaMa.
    """
    with h5py.File(cachepath, "r") as f:
        key = list(f.keys())[0]
        patches = f[key][:].tolist()
    return patches


def vectorize_2dIndex(domainarr):
    """
    Vectorize 2d map index (lon/lat) into 1d vector array

    Args:
        domainarr (np.ndarray): domain 2d map [nlat, nlon]
                               positive for calculation domain,
                               negative for non-calculation domain.
                               negative area will not be included in vectors.

    Returns:
        ndarray: 2d-1d mapper to vectorize
        ndarray: 1d-2d mapper for latitude
        ndarray: 1d-2d mapper for longitude

    Notes:
        as long as your basinarr is positive at your calculation domain
        and negative at your non-calculation domain (e.g., ocean in river
        routing model), the function returns vectorization information
        only for your calculation domain by removing non-calc. ones.
    """
    return camavec.make_vectorizedIndex(domainarr)


def getvecid(ilat, ilon, map2vec):
    """
    Returns vector index based on C style from 2d map coords.

    Args:
        ilon (int): longitudinal number
        ilat (int): latitudinal number
        nlon (int): number of longirtudinal grid cells

    Returns:
        int: index in 1d vectorized map at (ilat, ilon) in 2d map.
    """
    return map2vec[ilat, ilon]


def vectorize_map(map3d, map2vec, nvec):
    """
    Returns vectorized 2dmap

    Args:
        map3d (np.ndarray): input 2d map with layers [nlayers, nlat, nlon]
        map2vec (np.ndarray): 2d-1d mapper
        nvec (int): length of whole vector (len(vec2lat))
    """
    DTYPE = map3d.dtype
    if DTYPE == np.float32:
        outvector = camavec.vectorize_map3d_float32(map3d, map2vec, nvec)
    elif DTYPE == np.int32:
        outvector = camavec.vectorize_map3d_int32(map3d, map2vec, nvec)
    else:
        TypeError("type {0} is not supported".format(DTYPE))
    return outvector


def revert_map(vec2d, vec2lat, vec2lon, nlat, nlon):
    """
    get 2d map with layers (3d) from 1d vector with layers (2d).

    Args:
        vec2d (np.ndarray): vectorized 1d array with layers
                            [nlayers, nvec]
        vec2lat (np.ndarray): 1d-2d mapper for latitude
        vec2lon (np.ndarray): 1d-2d mapper for longitude
        nlon (int): number of longitudinal grid cells
        nlat (int): number of latitudinal grid cells

    Returns:
        np.ndarray: []
    """
    DTYPE = vec2d.dtype
    if DTYPE == np.float32:
        outmap = camavec.revert_layers_grid_float32(vec2d, vec2lat, vec2lon,
                                                    nlat, nlon)
    elif DTYPE == np.int32:
        outmap = camavec.revert_layers_grid_int32(vec2d, vec2lat, vec2lon,
                                                  nlat, nlon)
    else:
        TypeError("type {0} is not supported".format(DTYPE))
    return outmap
