import numpy as np
cimport cython
cimport numpy as np

ctypedef fused my_type:
    int
    float


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
                # ocean
                continue
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