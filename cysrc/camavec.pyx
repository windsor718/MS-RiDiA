import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np

ctypedef fused my_type:
    np.int32_t
    float

cdef np.int32_t undef_int = -9999
cdef np.float32_t undef_float = 1e+20

@cython.boundscheck(False)
@cython.wraparound(False)
def make_vectorizedIndex(const np.int32_t[:,:] domain):
    """
    Based on array-like object, domain, make vector indices
    and convertion matrix. domain should contain any positive values
    in the cell where you want to include in vectors, and negatives for
    those you do not want to.

    Args:
        domain (np.ndarray): domain info

    Returns:

    Notes:
        For CaMa-Flood Users:
          This function does not use nextxy.bin to make vector,
          which is the case in MAP2VEC subroutine in CaMa-Flood.
          DO NOT use this resultant to convert vector data from
          CaMa-Flood to a 2dmap. This is only for an closed usage
          in pyletkf context only.
    """

    cdef int nlat = domain.shape[0]
    cdef int nlon = domain.shape[1]
    cdef int bid
    cdef int vecid = 0
    map2vec = np.zeros([nlat, nlon], dtype=np.int32)
    cdef np.int32_t [:, :] map2vec_view = map2vec
    for ilat in range(nlat):
        for ilon in range(nlon):
            bid = domain[ilat, ilon]
            if bid < 0:
                map2vec_view[ilat, ilon] = undef_int
            else:
                map2vec_view[ilat, ilon] = vecid
                vecid += 1

    vec2lat = np.zeros([vecid], dtype=np.int32)
    cdef np.int32_t [:] vec2lat_view = vec2lat
    vec2lon = np.zeros([vecid], dtype=np.int32)
    cdef np.int32_t [:] vec2lon_view = vec2lon

    vecid = 0
    for ilat in range(nlat):
        for ilon in range(nlon):
            bid = domain[ilat, ilon]
            if bid < 0:
                # ocean
                continue
            vec2lat_view[vecid] = ilat
            vec2lon_view[vecid] = ilon
            vecid += 1
    return map2vec, vec2lat, vec2lon


@cython.boundscheck(False)
@cython.wraparound(False)
def vectorize_map2d_int32(const np.int32_t[:, :] inputmap,
                          const np.int32_t[:, :] map2vec, int nvec):
    """
    vectorize 2d map based on pre-calculated
    map2vec.

    Args:
        inputmap (np.ndarray): 2dim, np.int32 and np.float32
                               is accepted.
        map2vec (np.ndarray): output of make_vectorizeIndex()
        nvec (int): number of total vector ids
    Returns:
        vec (np.ndarray)
    """
    cdef int nlat = map2vec.shape[0]
    cdef int nlon = map2vec.shape[1]
    cdef int ilat
    cdef int ilon
    cdef int ivec
    vec = np.zeros([nvec], dtype=np.int32)
    cdef np.int32_t [:] vec_view = vec

    for ilat in prange(nlat, nogil=True):
        for ilon in prange(nlon):
            ivec = map2vec[ilat, ilon]
            if ivec < 0:
                continue
            else:
                vec_view[ivec] = inputmap[ilat, ilon]
    return vec


@cython.boundscheck(False)
@cython.wraparound(False)
def vectorize_map2d_float32(const np.float32_t[:, :] inputmap,
                            const np.int32_t[:, :] map2vec, int nvec):
    """
    vectorize 2d map based on pre-calculated
    map2vec.

    Args:
        inputmap (np.ndarray): 2dim, np.int32 and np.float32
                               is accepted.
        map2vec (np.ndarray): output of make_vectorizeIndex()
        nvec (int): number of total vector ids
    Returns:
        vec (np.ndarray)
    """
    cdef int nlat = map2vec.shape[0]
    cdef int nlon = map2vec.shape[1]
    cdef int ilat
    cdef int ilon
    cdef int ivec
    vec = np.zeros([nvec], dtype=np.float32)
    cdef np.float32_t [:] vec_view = vec
    for ilat in prange(nlat, nogil=True):
        for ilon in prange(nlon):
            ivec = map2vec[ilat, ilon]
            if ivec < 0:
                continue
            else:
                vec_view[ivec] = inputmap[ilat, ilon]
    return vec


@cython.boundscheck(False)
@cython.wraparound(False)
def vectorize_map3d_int32(const np.int32_t [:, :, :] inputmap,
                          np.int32_t[:, :] map2vec, int nvec):
    """
    wrapper for 3d data (2d map with multiple layers)
    """
    cdef int i
    cdef int nvar = inputmap.shape[0]
    veclayer = np.zeros([nvec], dtype=np.int32)
    cdef np.int32_t [:] veclayer_view = veclayer
    vec = np.zeros([nvar, nvec], dtype=np.int32)
    cdef np.int32_t [:, :] vec_view = vec

    for i in range(nvar):
        veclayer_view = vectorize_map2d_int32(inputmap[i], map2vec, nvec)
        vec_view[i, :] = veclayer_view

    return vec


@cython.boundscheck(False)
@cython.wraparound(False)
def vectorize_map3d_float32(const np.float32_t [:, :, :] inputmap,
                            np.int32_t[:, :] map2vec, int nvec):
    """
    wrapper for 3d data (2d map with multiple layers)
    """
    cdef int i
    cdef int nvar = inputmap.shape[0]
    veclayer = np.zeros([nvec], dtype=np.float32)
    cdef np.float32_t [:] veclayer_view = veclayer
    vec = np.zeros([nvar, nvec], dtype=np.float32)
    cdef np.float32_t [:, :] vec_view = vec

    for i in range(nvar):
        veclayer_view = vectorize_map2d_float32(inputmap[i], map2vec, nvec)
        vec_view[i, :] = veclayer_view

    return vec


@cython.boundscheck(False)
@cython.wraparound(False)
def revert_grid_int32(const np.int32_t[:] inputvector,
                      np.int32_t[:] vec2lat,
                      np.int32_t[:] vec2lon,
                      int nlat, int nlon):
    """
    revert vector to grid based on pre-calculated
    vec2lat and vec2lon array.
    """
    cdef int iv
    cdef int nvec = vec2lat.shape[0]
    cdef np.int32_t ilat
    cdef np.int32_t ilon

    mapgrid = np.ones([nlat, nlon], dtype=np.int32)*undef_int
    cdef np.int32_t [:, :] mapgrid_view = mapgrid

    for iv in prange(nvec, nogil=True):
        ilat = vec2lat[iv]
        ilon = vec2lon[iv]
        mapgrid_view[ilat, ilon] = inputvector[iv]
    return mapgrid


@cython.boundscheck(False)
@cython.wraparound(False)
def revert_layers_grid_int32(const np.int32_t[:, :] inputvector,
                             np.int32_t[:] vec2lat,
                             np.int32_t[:] vec2lon,
                             int nlat, int nlon):
    """
    wrapper to handle 1d vector with multiple layers
    """
    cdef int il
    cdef int nvec = len(vec2lat)
    cdef int nlayer = inputvector.shape[0]
    layer = np.ones([nlat, nlon], dtype=np.int32)*undef_int
    cdef np.int32_t [:, :] layer_view = layer
    mapgrid = np.ones([nlayer, nlat, nlon], dtype=np.int32)*undef_int
    cdef np.int32_t [:, :, :] mapgrid_view = mapgrid

    for il in range(nlayer):
        layer_view = revert_grid_int32(inputvector[il], vec2lat, vec2lon,
                                       nlat, nlon)
        mapgrid_view[il] = layer_view
    return mapgrid


@cython.boundscheck(False)
@cython.wraparound(False)
def revert_grid_float32(const np.float32_t[:] inputvector,
                        np.int32_t[:] vec2lat,
                        np.int32_t[:] vec2lon,
                        int nlat, int nlon):
    """
    revert vector to grid based on pre-calculated
    vec2lat and vec2lon array.
    """
    cdef int iv
    cdef int nvec = len(vec2lat)
    cdef np.int32_t ilat
    cdef np.int32_t ilon

    mapgrid = np.ones([nlat, nlon], dtype=np.float32)*undef_float
    cdef np.float32_t [:, :] mapgrid_view = mapgrid

    for iv in prange(nvec, nogil=True):
        ilat = vec2lat[iv]
        ilon = vec2lon[iv]
        mapgrid_view[ilat, ilon] = inputvector[iv]
    return mapgrid


@cython.boundscheck(False)
@cython.wraparound(False)
def revert_layers_grid_float32(const np.float32_t[:, :] inputvector,
                               np.int32_t[:] vec2lat,
                               np.int32_t[:] vec2lon,
                               int nlat, int nlon):
    """
    wrapper to handle 1d vector with multiple layers
    """
    cdef int il
    cdef int nvec = vec2lat.shape[0]
    cdef int nlayer = inputvector.shape[0]

    layer = np.ones([nlat, nlon], dtype=np.float32)*undef_float
    cdef np.float32_t [:, :] layer_view = layer
    mapgrid = np.ones([nlayer, nlat, nlon], dtype=np.float32)*undef_float
    cdef np.float32_t [:, :, :] mapgrid_view = mapgrid

    for il in range(nlayer):
        layer_view = revert_grid_float32(inputvector[il], vec2lat, vec2lon,
                                         nlat, nlon)
        mapgrid_view[il] = layer_view
    return mapgrid
