# cython: infer_types=True
import numpy as np
from cython.parallel import prange
cimport cython

ctypedef fused my_type:
    double
    float

@cython.boundscheck(False)
@cython.wraparound(False)
def get_storage_invertsely(my_type[:] outwth, my_type[:] rivwth,
                           my_type[:] rivlen, my_type[:] rivhgt,
                           my_type[:] rivshp, my_type[:] grarea, my_type[:, :] fldgrd,
                           int nvec, int nlfp, int undef):
    cdef int iv
    cdef my_type s
    cdef my_type wflw
    cdef my_type hflw
    cdef my_type wflp
    cdef my_type hflp
    cdef my_type wthpre
    cdef my_type dphpre
    cdef int layer
    cdef my_type wthinc
    cdef my_type Ariv
    cdef my_type Aflp
    cdef my_type Atmp

    if my_type is double:
        DTYPE = np.float64
    elif my_type is float:
        DTYPE = np.float32
    storage = np.zeros([2, nvec], dtype=DTYPE)
    cdef my_type [:, :] storage_view = storage
    for iv in prange(0, nvec, nogil=True):
    #for iv in range(0, nvec):
        if rivhgt[iv] == undef:
            continue
        if outwth[iv] < rivwth[iv]:
            # inbank
            s = rivshp[iv]
            wflw = outwth[iv]
            hflw = rivhgt[iv]*((wflw/rivwth[iv])**s)
            Ariv = wflw * hflw * (1-(1/(s+1)))
            Aflp = 0
        else:
            # outbank
            wflw = rivwth[iv]
            hflw = rivhgt[iv]
            s = rivshp[iv]
            Ariv = wflw * hflw * (1-(1/(s+1)))
            wflp = outwth[iv]
            wthpre = rivwth[iv]
            dphpre = 0
            layer = 0
            wthinc = grarea[iv]/(rivlen[iv]*nlfp)
            while wthpre < wflp:
                wthpre = wthinc + wthpre
                dphpre = fldgrd[layer, iv]*wthinc + dphpre
                layer  = 1 + layer
                if layer == nlfp-1:
                    break
            if layer == nlfp:
                # flow is over cell, assimilation may not be good in this grid.
                # leave it as it was or make it fldstomax.
                continue
            else:
                hflp = dphpre + (wflp-wthpre)*fldgrd[layer-1, iv]
            Atmp = (wflp + wflw) * hflp / 2.
            Ariv = Ariv + wflw*hflp
            Aflp = Atmp - wflw*hflp

        storage_view[0, iv] = rivlen[iv] * Ariv
        storage_view[1, iv] = rivlen[iv] * Aflp
    return storage
