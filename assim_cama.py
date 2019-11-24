import numpy as np
import h5py
import json
import datetime
import pytz
import subprocess
import os
from multiprocessing import Pool
from pyletkf import pyletkf
from pyletkf import exTool
import caseExtention as ext
import dautils as dau


class AssimCama(object):
    """
    CaMa data assimilation handler.
    Small modification of gosh/src file is needed:
     - sample gosh file: MSR_3min_wth_DA.sh
     - you need to cache d2fldstomax and d2fldgrd.
        - simply add following lines on cmf_ctrl_output_mod.F90
        - you may comment out those lines after you cached
          as these are time-independent.

    CodeFlow:
            register()  # register experiment information
                |
            initialize()  # define state vectors
                |         # initialize (ensemblize) variables
       |--------|  # map/skip spinup
       | ----------------
       | ||||(spinup)||||  # multi processing
       | ----------------
       |--------|  # join/map
         ----------------
         |||forwarding|||  # multi processing
         ----------------
                |  # join
            filterling  # data asslimation using pyletkf
          |||pyletkf|||  # multi processing in pyletkf
                |  # map
         ----------------
         ||||||||||||||||  # multi processing
         |postprocessing|  # calculate storage from assimilate values
         ||||||||||||||||  # re-write restart file to update initial conditions
         ----------------
                |  # join
        back to forwarding
    """

    def register(self, configjson):
        """
        registering model information from json file.
        Args:
            configjson (str): path to config file in json format
        Returns:
            None
        Notes:
            rather than using safeconfigparser, json format is better
            in language-wize IO.
        """
        with open(configjson) as f:
            varDict = json.load(f)
        self.expname = str(varDict["expname"])
        self.modeldir = str(varDict["modeldir"])
        self.outdir = os.path.join(self.modeldir, "out",
                                   self.expname, "{0:02d}")
        self.nlon = int(varDict["nlon"])
        self.nlat = int(varDict["nlat"])
        self.nvec = self.nlon * self.nlat
        self.mapdir = str(varDict["mapdir"])
        self.assimdatesPath = str(varDict["assimdatesPath"])
        self.west = float(varDict["west"])
        self.south = float(varDict["south"])
        self.res = float(varDict["res"])
        self.rnofdir = str(varDict["rnofdir"])
        self.assimdates = self.get_assimdates(self.assimdatesPath)
        self.eTot = int(varDict["eTot"])
        self.nCPUs = int(varDict["ncpus"])
        cachepath = str(varDict["cachepath"])
        if not os.path.exists(cachepath):
            use_cache = False
        self.patches = exTool.read_cache(cachepath)
        self.statevars = varDict(["statevals"])
        self.statetype = varDict(["statetype"])  # prognostic/parameter
        self.obsvars = varDict(["obsvars"])  # 1 if available 0 if not.
        self.check_consistency()  # check data consistency to avoid mistakes.
        self.assimconfig = varDict["assimconfig"]

        # instanciate pyletkf
        self.dacore = pyletkf.LETKF_core(self.assimconfig,
                                         mode="vector", use_cache=use_cache)
        self.dacore.initialize()

    def check_consistency(self):
        """
        checking data shapes to avoid mistakenly use data from
        unintentional source.
        """
        print("checking data consistency:")
        nvec = self.nlon * self.nlat
        print("checking local patch loaded...")
        assert len(self.patches) == nvec, "cached local patch size" +\
                                          "does not match with nlat/nlon."
        print("ok.")

    def get_nextSimDates(self, date, assimdates):
        """
        Based on the dates observation available,
        returns the simulation date range.
        Args:
            date (datetime.datetime): current date (simulation starts from)
                                      must be utc aware object
            assimdates (list): datetime.datetime objects, in utc posix form
        Returns:
            tuple: [start, end], datetime.datetime objects
        Notes:
            Make every datetime object aware for a timezone.
            Here datetime is always treated as UTC, which is the safest way.
        """
        assert date.tzinfo is not None\
            and date.tzinfo.utcoffset(date) is not None,\
            "date is not utc aware"
        # This is O(n), and binary search will be O(logn)
        # However, here len(assimdates) won't be that big,
        # and making for;if;else loop would be more costfull.
        nextAssimDate = [dd for dd in assimdates if date < dd][0]
        return [date, nextAssimDate]

    def get_assimdates(self, assimdatesPath):
        """
        read assimilation date information
        (when observation is available)
        Notes:
            format of file of assimdatePath:
              %Y%m%d
              19840101
              19840102
                ...
            dates should be in UTC.
        Flags:
            may be deprecated and use xarray with ncdf instead
        """
        with open(assimdatesPath, "r") as f:
            lines = f.read_lines()
            dates = [datetime.datetime.strptime(lines[0], l)
                     for l in lines[1::]]
        return dates

    def spinup(self, sdate, edate, ensrnof=False):
        """
        spiupping CaMa-Flood from zero storage.
        Args:
            sdate (datetime.datetime): spinup starts from this date
            edate (datetime.datetime): spinup ends at this date
            ensrnof (bool): True if you use ensemble runoff and need
                            string interpolation for paths.
        Returns:
            NoneType
        """
        simrange = [sdate, edate]
        print("spinupping state from {0} to {1}.".format(
                                            simrange[0].strftime("%Y%m%d%H"),
                                            simrange[1].strftime("%Y%m%d%H"))
              )
        p = Pool(self.nCPUs)
        argslist = [
                    [self.camagosh, self.modeldir, self.expname, self.rnofdir,
                     simrange, eNum, ensrnof, False]
                    for eNum in range(0, self.eTot)
                    ]
        p.map(run_CaMa_, argslist)
        p.close()

    def forward(self, date, nT, ensrnof=False, restart=True):
        """
        fowwarding a state until next observation is available.
        Args:
            date (datetime.datetime): current date in utc (aware object)
            ensrnof (bool): True if you use ensemble runoff and need
                        string interpolation for paths.
            restart (bool): True if restart from previous time restart file
                        This is usualy True, only pass False when you
                        want to do initial spinups.
        Returns:
            datetime.datetime: next initial date
        """
        simrange = self.get_nextSimDates(date, self.assimdates)
        print("forwarding state from {0} to {1}.".format(
                                            simrange[0].strftime("%Y%m%d%H"),
                                            simrange[1].strftime("%Y%m%d%H"))
              )
        p = Pool(self.nCPUs)
        argslist = [
                    [self.camagosh, self.modeldir, self.expname, self.rnofdir,
                     simrange, eNum, ensrnof, restart]
                    for eNum in range(0, self.eTot)
                    ]
        p.map(run_CaMa_, argslist)
        p.close()
        ndate = simrange[1] + datetime.timedelta(seconds=86400)
        nT += (ndate-date).days()
        print(date, ndate, nT)
        return ndate

    def filtering(self, date, nT):
        """
        LETKF at assmilation date
        Args:
            date (datetime.datetime): current date
            nT (int): number of time steps in output time
        """
        statevector = self.const_statevector(nT)
        obs, obserr = self.const_obs()
        xa = self.dacore.letkf(statevector, obs, obserr, obsvars,
                               nCPUs=1, smoother=False)
        self.parse_analysis(xa)
        self.update_states(xa)

    def const_statevector(self, nT):
        # create buffer array, this is used for concatenating memmap objects.
        buffer = np.memmap(self.dummyfile, dtype=np.float32, mode="w+",
                           shape=(len(self.statevars), self.eTot,
                                  nT, self.nvec)
                           )
        for idx, var in enumerate(self.statevars):  # not that many
            for eNum in range(self.eTot):  # not that many
                d = dau.load_data3d(os.path.join(self.outdir.format(eNum),
                                                 "{0}.bin".format(var)
                                                 ),
                                    nT, self.nlat, self.nlon, dtype=np.float32
                                    )
                buffer[idx, eNum, :, :] = d
        return buffer

    def const_obs(self, obs, date):
        """
        parse observation xarray and returns data at the date
        Args:
            obs (xarray.Dataset): observation dataset object
            date (datetime.datetime): date
        """
        obs_date = obs.sel(time=date).to_array().values()
        print(obs_date.shape)
        obs_date_values = obs_date[0]
        obs_date_errors = obs_date[1]
        return obs_date_values, obs_date_errors

    # These are only called at the very first time of the experiments.
    def initialize(self, date):
        """
        Initialize state variables.
        Once you call and create initial files, you may skip this.
        """
        for idx, var in enumerate(self.statevars):
            if self.statetype[idx] == "prognostic":
                continue
            elif self.statetype[idx] == "parameter":
                outarray = ext.gain_perturbation(var)
                for e in range(self.eTot):
                    odir = self.outdir.format(e)
                    bkupdir = os.path.join(odir, "init")
                    fn = "{0}.bin".format(var)
                    sf = outarray[e].flatten.astype(np.float32)
                    sf.tofile(os.path.join(odir, fn))
                    sf.tofile(os.path.join(bkupdir, fn))
            else:
                raise IndexError("type %s is " +
                                 "not defined".format(self.statetype[idx]))


# multiprocessing later
def run_CaMa_(args):
    """
    simple wrapper of run_CaMa for an usage in multiprocessing.
    Args:
        args (list): arguments passed to run_CaMa
    Returns:
        NoneType
    """
    run_CaMa(args[0], args[1], args[2], args[3], args[4], args[5],
             ensrnof=args[6], restart=args[7])


def run_CaMa(camagosh, modeldir, expname, rnofdir, simrange, eNum,
             ensrnof=False, restart=True):
    """
    execute cama-flood with in-code generated variable declaration
    Args:
        camagosh (str): path to the gosh file
        odeldir (str): model directory
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
        NoneType
    """
    varspath = make_vars(modeldir, expname, rnofdir, simrange, eNum,
                         ensrnof=ensrnof, restart=restart)
    subprocess.check_call([camagosh, varspath])


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
