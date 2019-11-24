import numpy as np
import h5py


def load_data3d(dpath, nt, nlat, nlon,
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
    shape2d = nlat * nlon
    if memmap:
        # data shape must be correct,
        # otherwise memmap will only load part of the data without warnings
        data = np.memmap(dpath, dtype=dtype, mode="r",
                         shape=(nt, shape2d), order="C")
    else:
        # not using out-of-core memory mapping, load whole data onto memory
        # this will benefit slight performance gain for array manipulation
        # but takes time if your data is large to load on.
        data = np.fromfile(dpath, dtype=dtype).reshape(nt, shape2d)
    return data


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


def vectorize_2dIndex(nlon, nlat):
    """
    Vectorize 2d map index (lon/lat) into 1d vector array
    Args:
        nlon (int): number of longitudinal grid cells
        nlat (int): number of latitudinal grid cells
    Returns:
        ndarray: 1d-2d mapper for longitude
        ndarray: 1d-2d mapper for latitude
    """
    veclon = np.tile(np.arange(0, nlon), nlat)
    veclat = np.repeat(np.arange(0, nlat), nlon)
    return veclon, veclat


def getvecid(ilon, ilat, nlon):
    """
    Returns vector index based on C style from 2d map coords.
    Args:
        ilon (int): longitudinal number
        ilat (int): latitudinal number
        nlon (int): number of longirtudinal grid cells
    Returns:
        int: index in 1d vectorized map at (ilat, ilon) in 2d map.
    """
    return ilat*nlon + ilon


def vec2map(array1d, nlon, nlat):
    """
    simply re-shape 1d vectorized array into 2d map.
    Args:
        array1d (np.ndarray): vectorized 1d array whose size is nlon*nlat
        nlon (int): number of longitudinal grid cells
        nlat (int): number of latitudinal grid cells
    Returns:
        np.ndarray: 2d array
    """
    return array1d.reshape(nlat, nlon)
