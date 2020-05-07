import numpy as np
import os
import pandas as pd
import warnings
import subprocess
import datetime
import dautils as dau
import calc_storage

"""
a set of case-specific functions.
need to review if you change state variables settings.
If you change code, make sure to change following warning
message for future usage.

A wrapper called from AssimCama is:
    - gain_perturbation: returns generated array
    - make_vars: generate text file and returns path
    - update_states: overwrite results and returns void
                     make sure to rename (mv) your ${var}.bin to
                     ${var}_yyyymmdd.bin where yyyymmdd is your assimilated
                     date. This is to avoid CaMa overwrite your result in
                     the next execution (CaMa will try to write REC=tstep).
Keeping those wrapper names/functions would make code integration easy/clean.
"""

caseCode = "MSR-width"
svals = ["outwth", "rivman", "rivshp"]
string = "\nThis extention is desined for caseCode {0}.\n" +\
         "{0} assumes that state variables are:\n" +\
         "\t{1}, {2}, {3}\n" +\
         "If your state variables are different from above,\n" +\
         "you may need to edit this file " +\
         "and dependnt modules"
warnings.warn(string.format(caseCode, svals[0], svals[1], svals[2]))


# utilities-initialization functions
def gain_perturbation(var, outdir, mapdir, nlat, nlon, eTot,
                      dtype_f=np.float32):
    """
    get perturbated initial parameters and save those in outdir file.
    You should save every parameters in state variables in outdir.
    Args:
        var (str): variable name
        refmappath (str): path to reference 2d map parameter
        nlat (int): number of latitudinal grid cells
        nlon (int): number of longitudinal grid cells
        eTot (int): total number of ensemble members
        dtype_f (np.dtype): float data type you want to save;
                            must be similar to model data type.
    """
    # read background prior information
    widthclass = pd.read_csv("/home/yi79a/yuta/RiDiA/data/MS-RiDiA/rawdata/priorinfo/WidthsClass.csv", index_col=0)
    # convert from loged value to normal value
    widthMed = widthclass["50%"].apply(lambda x: np.exp(x)).tolist()
    widths2d = np.memmap(os.path.join(mapdir, "rivwth_gwdlr.bin"),
                         dtype=dtype_f, shape=(nlat, nlon), mode="r")
    if var == "rivman":
        nclass = pd.read_csv("/home/yi79a/yuta/RiDiA/data/MS-RiDiA/rawdata/priorinfo/priorsNClass.csv", index_col=0)
        nLogMean = nclass["mean"].tolist()
        nLogStd = nclass["std"].tolist()
        outarray = get_map2d_from_lognormal(widths2d, widthMed,
                                            nLogMean, nLogStd, 0.005, 1.0, eTot)
    elif var == "rivshp":
        sclass = pd.read_csv("/home/yi79a/yuta/RiDiA/data/MS-RiDiA/rawdata/priorinfo/priorsRClass.csv", index_col=0)
        sLogMean = sclass["mean"].tolist()
        sLogStd = sclass["std"].tolist()
        outarray = get_map2d_from_lognormal(widths2d, widthMed,
                                            sLogMean, sLogStd, 1, 20, eTot)
    # deprecated
    # elif var == "rivhgt":
    #     rivhgt2d = np.memmap(os.path.join(mapdir, "rivhgt.bin"), dtype=dtype_f,
    #                          shape=(nlat, nlon), mode="r")
    #     outarray = get_rivhgt2d_from_lognormal(rivhgt2d, eTot, pstd=20)
    #
    for e in range(eTot):
        odir = outdir.format(e)
        paramdir = os.path.join(odir, "param")
        if not os.path.exists(paramdir):
            os.makedirs(paramdir)
        bkupdir = os.path.join(odir, "init")
        if not os.path.exists(bkupdir):
            os.makedirs(bkupdir)
        fn = "{0}.bin".format(var)
        if e == 0:
            # copy original map
            subprocess.check_call(["cp", os.path.join(mapdir, "rivhgt.bin"),
                                   os.path.join(bkupdir, "rivhgt.bin")])
            subprocess.check_call(["cp", os.path.join(mapdir, "rivman.bin"),
                                   os.path.join(bkupdir, "rivman.bin")])
            subprocess.check_call(["cp", os.path.join(mapdir, "rivshp.bin"),
                                   os.path.join(bkupdir, "rivshp.bin")])
            subprocess.check_call(["cp", os.path.join(mapdir, "rivbta.bin"),
                                   os.path.join(bkupdir, "rivbta.bin")])
            subprocess.check_call(["cp", os.path.join(mapdir, "rivhgt.bin"),
                                   os.path.join(paramdir, "rivhgt.bin")])
            subprocess.check_call(["cp", os.path.join(mapdir, "rivman.bin"),
                                   os.path.join(paramdir, "rivman.bin")])
            subprocess.check_call(["cp", os.path.join(mapdir, "rivshp.bin"),
                                   os.path.join(paramdir, "rivshp.bin")])
            subprocess.check_call(["cp", os.path.join(mapdir, "rivbta.bin"),
                                   os.path.join(paramdir, "rivbta.bin")])
            continue
        if var == "rivhgt":
            get_rivhgt2d(mapdir, bkupdir, paramdir, fn, nlat, nlon,
                         hc_logmean=-2.3, hc_logstd=1.17, hp_min=0.3, hp_max=0.7)
        else:
            sf = outarray[e].flatten().astype(dtype_f)
            sf.tofile(os.path.join(paramdir, fn))
            sf.tofile(os.path.join(bkupdir, fn))
        print("output parameter file: {0}".format(os.path.join(paramdir, fn)))
        print("backup parameter file: {0}".format(os.path.join(bkupdir, fn)))

        if var == "rivshp":
            crivshp = os.path.join(paramdir, fn)
            crivbta = os.path.join(paramdir, "rivbta.bin")
            cnextxy = os.path.join(mapdir, "nextxy.bin")
            print("output parameter file: {0}".format(os.path.join(paramdir, "rivbta.bin")))
            subprocess.check_call(["/home/yi79a/yuta/RiDiA/srcda/MS-RiDiA/fsrc/calc_rivbta", crivshp, crivbta,
                                  cnextxy, str(nlon), str(nlat)])
            print("backup parameter file: {0}".format(os.path.join(bkupdir, "rivbta.bin")))
            subprocess.check_call(["cp", os.path.join(paramdir, "rivbta.bin"),
                                   os.path.join(bkupdir, "rivbta.bin")])


#@jit
def get_map2d_from_lognormal(widths2d, widthMed,
                             paramLogMean, paramLogStd, min, max, eTot, undef=-9999):
    """
    get purterbated 2dmap from lognormal distribution.
    """
    nlat = widths2d.shape[0]
    nlon = widths2d.shape[1]
    outarray = np.zeros([eTot, nlat, nlon])
    for ilat in range(nlat):
        for ilon in range(nlon):
            width = widths2d[ilat, ilon]
            if width == undef:
                outarray[:, ilat, ilon] = undef
                continue
            idx = np.argmin(np.absolute(np.array(widthMed) - width))
            mean = paramLogMean[idx]
            std = paramLogStd[idx]
            out = np.random.lognormal(mean, std, size=eTot)
            # out = np.exp(logwths)
            out[out > max] = max
            out[out < min] = min
            outarray[:, ilat, ilon] = out
    return outarray


def get_rivhgt2d(mapdir, bkupdir, paramdir, fn, nlat, nlon,
                 hc_logmean=-2.3, hc_logstd=-1.17, hp_min=0.4, hp_max=0.6):
    cnextxy = os.path.join(mapdir, "nextxy.bin")
    crivout = os.path.join(mapdir, "outclm.bin")
    crivhgt = os.path.join(paramdir, fn)
    hc = np.random.lognormal(hc_logmean, hc_logstd, 1)[0]
    hp = np.random.uniform(hp_min, hp_max, 1)[0]
    subprocess.check_call(["/home/yi79a/yuta/RiDiA/srcda/MS-RiDiA/fsrc/calc_rivhgt", crivout, crivhgt,
                          cnextxy, str(hc), str(hp),
                          str(nlon), str(nlat)])
    subprocess.check_call(["cp", os.path.join(paramdir, "rivhgt.bin"),
                           os.path.join(bkupdir, "rivhgt.bin")])


#@jit
def get_rivhgt2d_from_lognormal(rivhgt2d, eTot, pstd=20,
                                hgtlimit=10, undef=-9999):
    """
    deprecated
    """
    nlat = rivhgt2d.shape[0]
    nlon = rivhgt2d.shape[1]
    outarray = np.zeros([eTot, nlat, nlon])
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        for ilat in range(nlat):
            for ilon in range(nlon):
                if rivhgt2d[ilat, ilon] == undef:
                    outarray[:, ilat, ilon] = undef
                    continue
                mean = np.log(rivhgt2d[ilat, ilon])
                std = mean+pstd/1000
                loghgts = np.ones([eTot])*1e+20
                limit = np.log(rivhgt2d[ilat, ilon]*hgtlimit)
                while (loghgts > limit).any():
                    loghgts = np.random.lognormal(mean, std,
                                                  size=eTot)
                outarray[:, ilat, ilon] = np.exp(loghgts)
    return outarray


# multiprocessing; forwarding functions
def make_vars(modeldir, expname, rnofdir, simrange, eNum,
              ensrnof=False, restart=True):
    """
    generate vars.txt to load variables in CaMa gosh.

    Args:
        modeldir (str): model directory
        expname (str): experiment name
        rnofdir (str): rnof directory
        simrange (list): one batch sim. start/end date
        eNum (int): ensemble member id
        ensrnof (bool): True if you use ensemble runoff and need
                        string interpolation for paths.
        restart (bool): True if restart from previous time restart file
                        This is usualy True, only pass False when you
                        want to do initial spinups.

    Returns:
        str: path to the vars_${eNum}.txt
    """
    base = modeldir
    exp = expname
    rdir = os.path.join(base, "out/{0}/{1:02d}".format(exp, eNum))
    ysta = simrange[0].year
    smon = simrange[0].month
    sday = simrange[0].day
    edate = simrange[1] + datetime.timedelta(seconds=86400)  # by next date 00:00
    yend = edate.year
    emon = edate.month
    eday = edate.day
    if ensrnof:
        crofdir = rnofdir % eNum
    else:
        crofdir = rnofdir
    if restart:
        spinup = 1
    else:
        spinup = 0
    crivhgt = os.path.join(rdir, "param/rivhgt.bin")
    crivman = os.path.join(rdir, "param/rivman.bin")
    crivshp = os.path.join(rdir, "param/rivshp.bin")
    crivbta = os.path.join(rdir, "param/rivbta.bin")
    namelist = ["BASE", "EXP", "RDIR", "YSTA", "SMON", "SDAY", "SPINUP",
                "YEND", "EMON", "EDAY", "CROFDIR", "CRIVHGT",
                "CRIVMAN", "CRIVSHP", "CRIVBTA"]
    varlist = [base, exp, rdir, ysta, smon, sday, spinup,
               yend, emon, eday, crofdir, crivhgt,
               crivman, crivshp, crivbta]
    outpath = os.path.join(rdir, "vars_{0:02d}.txt".format(eNum))
    with open(outpath, "w") as f:
        for name, var in zip(namelist, varlist):
            f.write("{0}={1}\n".format(name, var))
    return outpath
#


# multiprocessing; post-processing functions
def update_states(xa_each, outdir, mapdir, nlon, nlat, nt, map2vec, vec2lat,
                  vec2lon, eNum, nlfp, edate, dtype_f=np.float32):
    """
    update parameters from assimilated state vectors

    Args:
        xa_each (np.ndarray): analysis array at time nt of eNum (nvars, nReach)
        outdir (str): model directory
        mapdir (str): map directory
        nlon (int): number of longitudinal grid cells
        nlat (int): number of latitudinal grid cells
        nt (int): number of time (first dimension) for sim. output files
        eNum (int): ensemble member id
        nlfp (int): number of flood plain layers
        dtype_f (np object): data type for float in numpy object

    Returns:
        None
    """
    save_updates(xa_each, outdir, nlon, nlat, nt, vec2lat, vec2lon, dtype_f)
    rewrite_restart(outdir, mapdir, nlon, nlat, nt, map2vec, vec2lat, vec2lon,
                    nlfp, dtype_f=dtype_f)
    # write new rivbta.bin based on new rivshp.bin
    crivshp = os.path.join(outdir, "param", "rivshp.bin")
    crivbta = os.path.join(outdir, "param", "rivbta.bin")
    cnextxy = os.path.join(mapdir, "nextxy.bin")
    subprocess.check_call(["/home/yi79a/yuta/RiDiA/srcda/MS-RiDiA/fsrc/calc_rivbta", crivshp, crivbta, cnextxy, str(nlon), str(nlat)])

    # add noise to avoid convergence
    add_noise(xa_each, outdir, nlon, nlat, nt, map2vec,
              vec2lat, vec2lon, dtype_f)

    # rename files
    for var in ["outflw.bin", "outwth.bin", "flddph.bin"]:
        outname = "{0}_{1}.bin".format(var.split(".")[0],
                                       edate.strftime("%Y%m%d"))
        subprocess.check_call(["mv", os.path.join(outdir, var),
                               os.path.join(outdir, outname)])


def rewrite_restart(outdir, mapdir, nlon, nlat, nt, map2vec, vec2lat, vec2lon,
                    nlfp=10, dtype_f=np.float32):
    """
    re-write restart file (storage-only) to update initial condition
    after assimilation based on flow width.
    This inverting function is case-specific, thus if your assimilation
    variable changed, you need to review this.

    Args:
        outdir (str): out directory
        mapdir (str): map directory
        nlon (int): number of longitudinal grid cells
        nlat (int): number of latitudinal grid cells
        nt (int): number of time (first dimension) for sim. output files
        nlfp (int): number of flood plain layers
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
    nvec = len(vec2lat)
    # load data in vectorized format
    rivwth = dau.load_data3d(os.path.join(mapdir, "rivwth_gwdlr.bin"),
                             1, nlat, nlon, map2vec, nvec, dtype=np.float32)[0]
    rivlen = dau.load_data3d(os.path.join(mapdir, "rivlen.bin"),
                             1, nlat, nlon, map2vec, nvec, dtype=np.float32)[0]
    rivhgt = dau.load_data3d(os.path.join(mapdir, "rivhgt.bin"),
                             1, nlat, nlon, map2vec, nvec, dtype=np.float32)[0]
    grarea = dau.load_data3d(os.path.join(mapdir, "ctmare.bin"),
                             1, nlat, nlon, map2vec, nvec, dtype=np.float32)[0]
    fldgrd = dau.load_data3d(os.path.join(mapdir, "fldgrd.bin"),
                             nlfp, nlat, nlon, map2vec, nvec,
                             dtype=np.float32)[:]
    # after assimilation file is saved
    rivshp = dau.load_data3d(os.path.join(outdir, "param/rivshp.bin"),
                             1, nlat, nlon, map2vec, nvec, dtype=np.float32)[0]
    outwth = dau.load_data3d(os.path.join(outdir, "outwth.bin"),
                             nt, nlat, nlon, map2vec, nvec, dtype=np.float32)[-1]
    storage = calc_storage.get_storage_invertsely(outwth, rivwth, rivlen, rivhgt,
                                                  rivshp, grarea, fldgrd, nvec,
                                                  nlfp=nlfp, undef=-9999)
    restart = np.zeros([2, nlat, nlon])
    restart[0] = dau.revert_map(storage[0].reshape(1, nvec),
                                vec2lat, vec2lon, nlat, nlon)[0]
    restart[1] = dau.revert_map(storage[1].reshape(1, nvec),
                                vec2lat, vec2lon, nlat, nlon)[0]
    restart.flatten().astype(dtype_f).tofile(
                                             os.path.join(outdir, "restart.bin")
                                             )


def save_updates(xa_each, outdir, nlon, nlat, nt, vec2lat, vec2lon, dtype_f):
    """
    save analysis onto file in outdir.

    Args:
        xa_each (np.ndarray): analysis array at time nt of eNum (nvars, nReach)
                              nvars are in order of:
                                [outwth, rivman, rivshp]
        outdir (str): out directory
        nlon (int): number of longitudinal grid cells
        nlat (int): number of latitudinal grid cells
        nt (int): number of time (first dimension) for sim. output files
        dtype_f (np object): data type for float in numpy object

    ToDo:
        Maybe add disturbance on parameters?
    """
    nvec = len(vec2lat)
    # update outwth
    data = np.memmap(os.path.join(outdir, "outwth.bin"), dtype=dtype_f,
                     shape=(nt, nlat, nlon), mode="r+")  # use carefully!
    tmp = dau.revert_map(xa_each[0, :].astype(dtype_f).reshape(1, nvec),
                         vec2lat, vec2lon, nlat, nlon)[0]
    undefloc = (tmp == 1e+20)
    tmp[undefloc] = 1
    tmp = np.exp(tmp)  # log
    tmp[undefloc] = 1e+20
    data[-1, :, :] = tmp[:, :]
    del data  # closing and flushing changes to disk

    data0 = np.memmap(os.path.join(outdir, "param/rivhgt.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="w+")  # use carefully!
    tmp = dau.revert_map(xa_each[1, :].astype(dtype_f).reshape(1, nvec),
                         vec2lat, vec2lon, nlat, nlon)[0]
    undefloc = (tmp == 1e+20)
    tmp[undefloc] = 1  # just to avoid overflow
    tmp = np.exp(tmp)
    tmp[tmp < 1] = 1
    tmp[undefloc] = -9999
    data0[:, :] = tmp[:, :]
    del data0

    data1 = np.memmap(os.path.join(outdir, "param/rivman.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="w+")  # use carefully!
    tmp = dau.revert_map(xa_each[2, :].astype(dtype_f).reshape(1, nvec),
                         vec2lat, vec2lon, nlat, nlon)[0]
    undefloc = (tmp == 1e+20)
    tmp[undefloc] = 1
    tmp = np.exp(tmp)
    tmp[tmp < 0.01] = 0.01
    tmp[undefloc] = -9999
    data1[:, :] = tmp[:, :]
    del data1

    data2 = np.memmap(os.path.join(outdir, "param/rivshp.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="w+")  # use carefully!
    tmp = dau.revert_map(xa_each[3, :].astype(dtype_f).reshape(1, nvec), vec2lat, vec2lon, nlat, nlon)[0]
    undefloc = (tmp == 1e+20)
    tmp[undefloc] = 1
    tmp = np.exp(tmp)
    tmp[tmp < 1] = 1
    tmp[undefloc] = -9999
    data2[:, :] = tmp[:, :]
    del data2


def multiply_normalnoise(vec, std, minv, maxv):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    noise = np.random.normal(1, std, 1)
    if noise < minv:
        noise = minv
    elif noise > maxv:
        noise = maxv
    print(noise, (vec*noise).min(), (vec*noise).max())
    return vec*noise


def add_noise(xa_each, outdir, nlon, nlat, nt, map2vec, vec2lat, vec2lon, dtype_f):
    """
    add noise for next loop.
    Tweaking needed.
    """
    nvec = len(vec2lat)
    xa_each = np.exp(xa_each)  # its log; xa is vector, so there is no undef.
    data0 = np.memmap(os.path.join(outdir, "param/rivhgt.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="r+")  # use carefully!
    next0 = multiply_normalnoise(xa_each[1, :], 0.25, 0.5, 1.5)
    tmp = dau.revert_map(next0.astype(dtype_f).reshape(1, nvec), vec2lat, vec2lon, nlat, nlon)[0]
    tmp[tmp < 0.5] = 0.5
    tmp[tmp == 1e+20] = -9999
    tmp[tmp > 20] = 20
    data0[:, :] = tmp[:, :]
    del data0

    data1 = np.memmap(os.path.join(outdir, "param/rivman.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="r+")  # use carefully!
    next1 = multiply_normalnoise(xa_each[2, :], 0.25, 0.5, 1.5)
    tmp = dau.revert_map(next1.astype(dtype_f).reshape(1, nvec), vec2lat, vec2lon, nlat, nlon)[0]
    tmp[tmp < 0.01] = 0.01
    tmp[tmp == 1e+20] = -9999
    tmp[tmp > 5] = 5
    data1[:, :] = tmp[:, :]
    del data1

    data2 = np.memmap(os.path.join(outdir, "param/rivshp.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="r+")  # use carefully!
    next2 = multiply_normalnoise(xa_each[3, :], 0.25, 0.5, 1.5)
    tmp = dau.revert_map(next2.astype(dtype_f).reshape(1, nvec), vec2lat, vec2lon, nlat, nlon)[0]
    tmp[tmp < 1] = 1
    tmp[tmp == 1e+20] = -9999
    tmp[tmp > 20] = 20
    data2[:, :] = tmp[:, :]
    del data2
#
