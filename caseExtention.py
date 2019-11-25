import numpy as np
import os
import pandas as pd
import warnings
from numba import jit
import dautils as du

"""
a set of case-specific functions.
need to review if you change state variables settings.
If you change code, make sure to change following warning
message for future usage.

A wrapper called from AssimCama is:
    - gain_perturbation: returns generated array
    - make_vars: generate text file and returns path
    - update_states: overwrite results and returns void
Keeping those wrapper names/functions would make code integration easy/clean.
"""

caseCode = "MSR-width"
svals = ["outwth", "rivman", "rivshp"]
string = "This extention is desined for caseCode {0}.\n" +\
         "{0} assumes that state variables are:\n" +\
         "\t{1}, {2}, {3}" +\
         "If your state variables are different from above,\n" +\
         "you may need to edit this file " +\
         "and dependnt modules".format(caseCode, svals[0],
                                       svals[1], svals[2])
warnings.warn(string)


# utilities-initialization functions
def gain_perturbation(var, refmappath, nlat, nlon, eTot, dtype_f=np.float32):
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
    widthclass = pd.read_csv("./WidthsClass.csv", index_col=0)
    # convert from loged value to normal value
    widthMed = widthclass["50%"].apply(lambda x: np.exp(x)).tolist()
    widths2d = np.memmap(refmappath, dtype=dtype_f,
                         shape=(nlat, nlon), mode="r")
    if var == "rivman":
        widths2d = np.memmap(refmappath, dtype=dtype_f,
                             shape=(nlat, nlon), mode="r")
        nclass = pd.read_csv("./priorsNClass.csv", index_col=0)
        nLogMean = nclass["mean"].tolist()
        nLogStd = nclass["std"].tolist()
        outarray = get_map2d_from_lognormal(widths2d, widthMed,
                                            nLogMean, nLogStd, eTot)
    elif var == "rivshp":
        widths2d = np.memmap(refmappath, dtype=dtype_f,
                             shape=(nlat, nlon), mode="r")
        sclass = pd.read_csv("./priorsRClass.csv", index_col=0)
        sLogMean = sclass["mean"].tolist()
        sLogStd = sclass["std"].tolist()
        outarray = get_map2d_from_lognormal(widths2d, widthMed,
                                            nLogMean, nLogStd, eTot)
    elif var == "rivhgt":
        rivhgt2d = np.memmap(refmappath, dtype=dtype_f,
                             shape=(nlat, nlon), mode="r")
        outarray = get_rivhgt2d_from_lognormal(rivhgt2d, eTot, std=1.5)
    return outarray


@jit
def get_map2d_from_lognormal(widths2d, widthMed,
                             paramLogMean, paramLogStd, eTot, undef=-9999):
    nlat = width2d.shape[0]
    nlon = width2d.shape[1]
    outarray = np.zeros([eTot, nlat, nlon])
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
    outarray = np.zeros([eTot, nlat, nlon])
    for ilat in range(nlat):
        for ilon in range(nlon):
            if rivhgt[ilat, ilon] == undef:
                outarray[:, ilat, ilon] = undef
                continue
            mean = np.log(rivhgt[ilat, ilon])
            loghgts = np.random.lognormal(mean, std, size=eTot)
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
    rdir = os.path.join(base, "out/{0}/{1:03d}".format(exp, eNum))
    ysta = simrange[0].year
    smon = simrange[0].month
    sday = simrange[0].day
    yend = simrange[1].year
    emon = simrange[1].month
    eday = simrange[1].day
    if ensrnof:
        crofdir = rnofdir % eNum
    else:
        crofdir = rnofdir
    if restart:
        spinup = 0
    else:
        spinup = 1
    crivhgt = os.path.join(rdir, "rivhgt.bin")
    crivman = os.path.join(rdir, "rivman.bin")
    crivshp = os.path.join(rdir, "rivshp.bin")
    crivbta = os.path.join(rdir, "rivbta.bin")
    namelist = ["BASE", "EXP", "RDIR", "YSTA", "SMON", "SDAY", "SPINUP"
                "YEND", "EMON", "EDAY", "CROFDIR", "CRIVHGT",
                "CRIVMAN", "CRIVSHP", "CRIVBTA"]
    varlist = [base, exp, rdir, ysta, smon, sday, spinup,
               yend, emon, eday, crofdir, crivhgt,
               crivman, crivshp, crivbta]
    outpath = os.path.join(rdir, "vars_{0:02d}".format(eNum))
    with open(outpath) as f:
        for name, var in zip(namelist, varlist):
            f.write("{0}={1}\n".format(name, var))
    return outpath
#


# multiprocessing; post-processing functions
def update_states(xa_each, outdir, mapdir, nlon, nlat, nt, eNum,
                  nlfp, dtype_f=np.float32):
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
    save_update(xa_each, outdir, nlon, nlat, nt, dtype_f)
    rewrite_restart(outdir, mapdir, nlon, nlat, nt, nlfp,
                    dtype_f=np.float32)
    # write new rivbta.bin based on new rivshp.bin
    crivshp = os.path.join(outdir, "rivshp.bin")
    crivbta = os.path.join(outdir, "rivbta.bin")
    subprocess.check_call("./fsrc/calc_rivbta", crivshp, crivbta, nlon, nlat)


def rewrite_restart(outdir, mapdir, nlon, nlat, nt,
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
    # load data in vectorized format
    rivwth = du.load_data3d(os.path.join(mapdir, "rivwth_gwdlr.bin"),
                            1, nlat, nlon, dtype=np.float32)[0]
    rivlen = du.load_data3d(os.path.join(mapdir, "rivlen.bin"),
                            1, nlat, nlon, dtype=np.float32)[0]
    rivhgt = du.load_data3d(os.path.join(mapdir, "rivhgt.bin"),
                            1, nlat, nlon, dtype=np.float32)[0]
    grarea = du.load_data3d(os.path.join(mapdir, "ctmare.bin"),
                            1, nlat, nlon, dtype=np.float32)[0]
    fldgrd = du.load_data3d(os.path.join(mapdir, "fldgrd.bin"),
                            nlfp, nlat, nlon, dtype=np.float32)[:]
    nvec = nlat*nlon
    # after assimilation file is saved
    rivshp = du.load_data3d(os.path.join(outdir, "rivshp.bin"),
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


def save_updates(xa_each, outdir, nlon, nlat, nt, dtype_f):
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
    """
    # update outwth
    data0 = np.memmap(os.path.join(outdir, "outwth.bin"), dtype=dtype_f,
                      shape=(nt, nlat, nlon), mode="w+")  # use carefully!
    data0[-1] = dau.vec2map(xa_each[0, :].astype(dtype_f))
    del data0  # closing and flushing changes to disk
    data1 = np.memmap(os.path.join(outdir, "rivman.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="w+")  # use carefully!
    data1 = dau.vec2map(xa_each[1, :].astype(dtype_f))
    del data1
    data2 = np.memmap(os.path.join(outdir, "rivshp.bin"), dtype=dtype_f,
                      shape=(nlat, nlon), mode="w+")  # use carefully!
    data2 = dau.vec2map(xa_each[2, :].astype(dtype_f))
    del data2
    # temporally; test purpose
    data = np.fromfile(os.path.join(outdir, "rivman.bin"), dtype=dtype_f).reshape(nlat, nlon)
    assert (data == dau.vec2map(xa_each[1, :].astype(dtype_f))), "not overwriten!"
    #
#
