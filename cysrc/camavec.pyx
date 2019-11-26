import numpy as np
from cython.parallel import prange
cimport cython
cimport numpy as np

ctypedef fused my_type:
    np.int32_t
    float


#@cython.boundscheck(False)
#@cython.wraparound(False)
def make_vectorizeIndex(np.int32_t[:,:] basin):
    """
    Based on array-like object from basin.bin in CaMa-Flood,
    make make vector indices and conversion matrix.
    
    Args:
        basin (np.ndarray): basin ids

    Returns:

    Notes:
        This function does not use nextxy.bin to make vector,
        which is the case in MAP2VEC subroutine in CaMa-Flood.
        DO NOT use this resultant to convert vector data from
        CaMa-Flood to a 2dmap. This is only for an closed usage
        in pyletkf context only.
    """

    cdef int nlat = basin.shape[0]
    cdef int nlon = basin.shape[1]
    cdef int bid
    cdef int vecid = 0
    map2vec = np.zeros([nlat, nlon], dtype=np.int32)
    cdef np.int32_t [:, :] map2vec_view = map2vec

    for ilat in range(nlat):
        for ilon in range(nlon):
            bid = basin[ilat, ilon]
            if bid < 0:
                map2vec_view[ilat, ilon] = -9999
            else:
                map2vec_view[ilat, ilon] = vecid
            vecid += 1

    print(map2vec)
    print(map2vec_view)
    vec2lat = np.zeros([vecid], dtype=np.int32)
    cdef np.int32_t [:] vec2lat_view = vec2lat
    vec2lon = np.zeros([vecid], dtype=np.int32)
    cdef np.int32_t [:] vec2lon_view = vec2lon

    vecid = 0
    for ilat in range(nlat):
        for ilon in range(nlon):
            bid = basin[ilat, ilon]
            if bid < 0:
                # ocean
                continue
            vec2lat_view[vecid] = ilat
            vec2lon_view[vecid] = ilat
            vecid += 1
    print(vec2lat)
    print(vec2lat_view)
    return map2vec, vec2lat, vec2lon


#@cython.boundscheck(False)
#@cython.wraparound(False)
def vectorize_map2d(my_type[:, :] inputmap,
                    np.int32_t[:, :] map2vec, int nvec):
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
    if my_type == np.int32_t:
        DTYPE = np.int32
    elif my_type == float:
        DTYPE = np.float32
    vec = np.zeros([nvec], dtype=DTYPE)
    cdef my_type [:] vec_view = vec

    for ilat in prange(nlat, nogil=True):
        for ilon in prange(nlon):
            ivec = map2vec[ilat, ilon]
            if ivec < 0:
                continue
            else:
                vec_view[ivec] = inputmap[ilat, ilon]
    print(vec)
    print(vec_view)
    return vec


#@cython.boundscheck(False)
#@cython.wraparound(False)
def vectorize_map3d(my_type[:, :, :] inputmap,
                    np.int32_t[:, :] map2vec, int nvec):
    cdef int i
    cdef int nvar = inputmap.shape[0]
    if my_type == np.int32_t:
        DTYPE = np.int32
    elif my_type == float:
        DTYPE = np.float32
    vec = np.zeros([nvec], dtype=DTYPE)
    cdef my_type [:] vec_view = vec

    for i in range(nvar):
        vec_view[i] = vectorize_map2d(inputmap[i], map2vec, nvec)
    
    return vec