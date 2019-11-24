import numpy as np
import os
import pandas as pd
from numba import jit
import dautils as du


# case-specific functions need to review if you change state variables settings
def rewrite_restart(modeldir, mapdir, exp, nlon, nlat, nt, eNum,
                    rivwth_fn, rivlen_fn, rivhgt_fn, rivshp_fn,
                    grarea_fn, fldgrd_fn, nlfp=10, dtype_f=np.float32):
    """
    re-write restart file (storage-only) to update initial condition
    after assimilation based on flow width.
    This inverting function is case-specific, thus if your assimilation
    variable changed, you need to review this.
    Args:
        modeldir (str): model directory
        mapdir (str): map directory
        exp (str): experiment name
        nlon (int): number of longitudinal grid cells
        nlat (int): number of latitudinal grid cells
        nt (int): number of time (first dimension) for sim. output files
        eNum (int): ensemble member id
        rivwth_fn (str): name of rivwth file
        rivlen_fn (str): name of rivlen file
        rivhgt_fn (str): name of rivhgt file
        rivshp_fn (str): name of rivshp file
        dtype_f (np object): data type for float in numpy object
    Returns:
        NoneType
    Notes:
        make sure that this function is called after your assimilation
        results are saved - this func. will read latest values from
        output files.
        nt is the number of layers your output have currently -
        meaning that if your output is currently shaped as [10, 440, 500]
        then nt=10. This number is usually the number of outer model loop
        passed.
    """
    outdir = os.path.join(modeldir, "out/{0}/{1:03d}".format(exp, eNum))
    # load data in vectorized format
    rivwth = du.load_data3d(os.path.join(mapdir, rivwth_fn),
                            1, nlat, nlon, dtype=np.float32)[0]
    rivlen = du.load_data3d(os.path.join(mapdir, rivlen_fn),
                            1, nlat, nlon, dtype=np.float32)[0]
    rivhgt = du.load_data3d(os.path.join(mapdir, rivhgt_fn),
                            1, nlat, nlon, dtype=np.float32)[0]
    grarea = du.load_data3d(os.path.join(mapdir, grarea_fn),
                            1, nlat, nlon, dtype=np.float32)[0]
    fldgrd = du.load_data3d(os.path.join(mapdir, rivhgt_fn),
                            nlfp, nlat, nlon, dtype=np.float32)[:]
    nvec = nlat*nlon
    # after assimilation file is saved
    rivshp = du.load_data3d(os.path.join(outdir, rivshp_fn),
                            1, nlat, nlon, dtype=np.float32)[0]
    outwth = du.load_data3d(os.path.join(outdir, "outwth.bin"),
                            nt, nlat, nlon, dtype=np.float32)[-1]
    storage = invert_storage(outwth, rivwth, rivlen, rivhgt,
                             nvec, rivshp, grarea, fldgrd,
                             nlfp=nlfp, undef=-9999)
    restart = np.zeros([2, nlat, nlon])
    restart[0] = du.vec2map(storage[0])
    restart[1] = du.vec2map(storage[1])
    restart.flatten().astype(dtype_f).tofile(
                                        os.path.join(outdir, "restart.bin")
                                        )


def get_storage_invertsely(outwth, rivwth, rivlen, rivhgt,
                           nvec, rivshp, grarea, fldgrd,
                           nlfp=10, undef=-9999):
    # Cython implementation later
    storage = np.zeros([2, nvec])
    for iv in range(0, nvec):
        if rivhgt[iv] == undef:
            continue
        if outwth[iv] < rivwth[iv]:
            # inbank
            s = rivshp[iv]
            wflw = outwth[iv]
            hflw = rivhgt[iv]*(wflw/rivwth[iv])**(1/s)
            Ariv = wflw*hflw*(1-(1/(s+1)))
            Aflp = 0
        else:
            # outbank
            wflw = rivwth[iv]
            hflw = rivhgt[iv]
            s = rivshp[iv]
            Ariv = wflw * hflw * (1-(1/(s+1)))

            wflp = outwth[iv]
            wthpre = 0
            dphpre = 0
            layer = 0
            wthinc = grarea[iv]*rivlen[iv]**(-1.)*nlfp**(-1)
            while wthpre < wflp:
                wthpre += wthinc
                dphpre += fldgrd[layer, iv]*wthinc
                layer += 1
                if layer == nlfp:
                    break
            if layer == nlfp:
                # flow is over cell, assimilation may not be good in this grid.
                # leave it as it was or make it fldstomax.
                continue
            else:
                hflp = dphpre + (wflp-wthpre)*fldgrd[layer, iv]
            Aflp = (wflp + wflw) * hflp / 2.
        storage[0, iv] = rivlen[iv] * Ariv
        storage[1, iv] = rivlen[iv] * Aflp
    return storage


def gain_perturbation(var, refmappath, nlat, nlon, eTot, dtype=np.float32):
    # read background prior information
    widthclass = pd.read_csv("./WidthsClass.csv", index_col=0)
    # convert from loged value to normal value
    widthMed = widthclass["50%"].apply(lambda x: np.exp(x)).tolist()
    widths2d = np.memmap(refmappath, dtype=dtype,
                         shape=(nlat, nlon), mode="r")
    if var == "rivman":
        widths2d = np.memmap(refmappath, dtype=dtype,
                             shape=(nlat, nlon), mode="r")
        nclass = pd.read_csv("./priorsNClass.csv", index_col=0)
        nLogMean = nclass["mean"].tolist()
        nLogStd = nclass["std"].tolist()
        outarray = get_map2d_from_lognormal(widths2d, widthMed,
                                            nLogMean, nLogStd, eTot)
    elif var == "rivshp":
        widths2d = np.memmap(refmappath, dtype=dtype,
                             shape=(nlat, nlon), mode="r")
        sclass = pd.read_csv("./priorsRClass.csv", index_col=0)
        sLogMean = sclass["mean"].tolist()
        sLogStd = sclass["std"].tolist()
        outarray = get_map2d_from_lognormal(widths2d, widthMed,
                                            nLogMean, nLogStd, eTot)
    elif var == "rivhgt":
        rivhgt2d = np.memmap(refmappath, dtype=dtype,
                             shape=(nlat, nlon), mode="r")
        outarray = get_rivhgt2d_from_lognormal(rivhgt2d, eTot, std=1.5)
    return outarray


@jit
def get_map2d_from_lognormal(widths2d, widthMed,
                             paramLogMean, paramLogStd, eTot, undef=-9999):
    nlat = width2d.shape[0]
    nlon = width2d.shape[1]
    outarray = np.zeros[[eTot, nlat, nlon]
    for ilat in range(nlat):
        for ilon in range(nlon):
            width = widths2d[ilat, ilon]
            if width = undef:
                outarray[:, ilat, ilon] = undef
                continue
            idx = np.argmin(np.array(widthMed) - width)
            mean = paramLogMean[idx]
            std = paramLogStd[idx]
            logwths = np.random.lognormal(mean, std, size=eTot)
            outarray[:, ilat, ilon] = np.exp(logwths)
    return outarray


@jit
def get_rivhgt2d_from_lognormal(rivhgt2d, eTot, std=1.5, undef=-9999):
    nlat = rivhgt2d.shape[0]
    nlon = rivhgt2d.shape[1]
    outarray = np.zeros[[eTot, nlat, nlon]
    for ilat in range(nlat):
        for ilon in range(nlon):
            if rivhgt[ilat, ilon] == undef:
                outarray[:, ilat, ilon] = undef
                continue
            mean = np.log(rivhgt[ilat, ilon])
            loghgts = np.random.lognormal(mean, std, size=eTot)
            outarray[:, ilat, ilon] = np.exp(loghgts)
    return outarray