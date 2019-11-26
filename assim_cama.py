import numpy as np
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

camaout_dtype = np.float32  # change if changed


class AssimCama(object):
    """
    CaMa-Flood [v395b] data assimilation handler.

    Notes:
        Small modification of gosh/src file is needed:
            - sample gosh file: MSR_3min_wth_DA.sh
            - you need to cache d2fldstomax and d2fldgrd in 2d map format.
            - simply add following lines on cmf_ctrl_output_mod.F90
            - you may comment out those lines after you cached
              as these are time-independent.
        This class should be almost-case-universal to avoid bugs.
            - those case-specific functions is collected in
              caseExtentions.py.
            - If you change system model design (e.g., state variables)
              please make sure to review the code, and edit for your case
              if applicable.
            - You may need to edit some class methods to match arguments in
              your extentions, but those editions would be appearent and clean.
        Watch out for byte precision when you save files to interect with CaMa.
            - In this code, especially pyletkf, uses double precision
                - np.float64, real*8
            - In current version of CaMa-Flood, outputs are single precision.
                - np.float32, real*4

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

    def register(self, configjson, initialize=True):
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
        # read and setup instance variables
        with open(configjson) as f:
            varDict = json.load(f)
        self.expname = str(varDict["expname"])
        self.modeldir = str(varDict["modeldir"])
        self.outdir = os.path.join(self.modeldir, "out",
                                   self.expname, "{0:02d}")
        self.nlon = int(varDict["nlon"])
        self.nlat = int(varDict["nlat"])
        self.mapdir = str(varDict["mapdir"])
        self.assimdatesPath = str(varDict["assimdatesPath"])
        self.west = float(varDict["west"])
        self.south = float(varDict["south"])
        self.res = float(varDict["res"])
        self.rnofdir = str(varDict["rnofdir"])
        self.ensrnof = bool(varDict["ensrnof"])
        self.eTot = int(varDict["eTot"])
        self.nCPUs = int(varDict["nCPUs"])
        self.statevars = varDict(["statevars"])
        self.statetype = varDict(["statetype"])  # prognostic/parameter
        self.obsvars = varDict(["obsvars"])  # 1 if available 0 if not.
        self.assimconfig = varDict["assimconfig"]
        self.cachepath = str(varDict["cachepath"])
        if not os.path.exists(self.cachepath):
            use_cache = False
        self.patches = exTool.read_cache(self.cachepath)
        self.assimdates = self.get_assimdates(self.assimdatesPath)
        self.nvec = len(vec2lat)
        # check data consistency to avoid mistakes.
        self.check_consistency()

        # instanciate pyletkf
        self.dacore = pyletkf.LETKF_core(self.assimconfig,
                                         mode="vector", use_cache=use_cache)
        self.dacore.initialize()

        # get perturbated state variables and save those in outdir
        if initialize:
            self.initialize()

    # main higher-API to start simulation
    def start(self, sdate, edate, spinup=False):
        """
        main driver to start experiment

        Args:
            sdate (datetime.datetime): start date, must be utc aware
            edate (datetime.datetime): end date, must be utc aware
            spinup (bool): True to spinup for a year
        """

        if spinup:
            utc = pytz.utc
            edate = datetime.datetime(sdate.year+1, 1, 1)
            edate = utc.localize(edate)
            self.spinup(sdate, edate, ensrnof=self.ensrnof)
        date = sdate
        nT = 0
        while date < edate:
            date, nT = self.driver(date, nT)

    def restart(self, edate):
        # read most rescent backup info
        with open(os.path.join(self.outdir.format(0), "ntlog.txt"), "r") as f:
            rescent_info = f.readlines()[-1].split(",")
        date = datetime.datetime.strptime(rescent_info[0], "%Y%m%d%H")
        nT = int(rescent_info[1])
        # copy backup restart files
        resfile = "restart_{0}.bin".format(date.strftime("%Y%m%d%H"))
        for eNum in range(self.eTot):
            outdir = self.outdir.format(eNum)
            respath = os.path.join(outdir, "restart", resfile)
            outpath = os.path.join(outdir, "restart.bin")
            subprocess.check_call("cp", respath, outpath)
        # restart simulation
        while date < edate:
            date, nT = self.driver(date, nT)

    def driver(self, date, nT):
        ndate, nT = self.forward(date, nT,
                                 ensrnof=self.ensrnof, restart=True)
        self.filtering(date, nT)
        if ndate.year > date.year:
            self.backup_restart(ndate)
            with open(os.path.join(self.outdir, "ntlog.txt"), "a") as f:
                f.write("{0}, {1}"
                        .format(ndate.strftime("%Y%m%d%H"), nT))
        return ndate, nT



    # utilities
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
        utc = pytz.utc
        with open(assimdatesPath, "r") as f:
            lines = f.read_lines()
            dates = [utc.localize(datetime.datetime.strptime(lines[0], l))
                     for l in lines[1::]]
        return dates

    def initialize(self):
        """
        Initialize parameters in state variables.
        Initial parameters will saved in self.outdir.
        In case of reuses, the same file is also save at self.outdir/init/.
        Once you call and create initial files, you may skip this.

        Notes:
            To tweak behavior of a perturbation, edit caseExtention.py
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
                    sf = outarray[e].flatten.astype(camaout_dtype)
                    sf.tofile(os.path.join(odir, fn))
                    sf.tofile(os.path.join(bkupdir, fn))
            else:
                raise IndexError("type %s is " +
                                 "not defined".format(self.statetype[idx]))
    #

    # spinup functions
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
        # backup spiup files
        self.backup_restart(edate + datetime.timedelta(seconds=86400))

    def backup_restart(self, date):
        for eNum in range(self.eTot):
            outdir = self.outdir.format(eNum)
            resdir = os.path.join(outdir, "restart")
            if not os.path.exists(resdir):
                os.makedirs(resdir)
            respath = os.path.join(outdir, "restart.bin")
            datestring = date.strftime("%Y%m%d%H")
            bpath = os.path.dir(resdir,
                                "restart_{0}.bin".format(datestring))
            subprocess.check_call(["cp", respath, bpath])
    #

    # forwarding functions
    def forward(self, date, nT, ensrnof=False, restart=True):
        """
        fowwarding a state until next observation is available.

        Args:
            date (datetime.datetime): current date in utc (aware object)
            ensrnof (bool): True if you use ensemble runoff and need
                        string interpolation for paths.
            restart (bool): True if restart from previous time restart file
                        This is usualy True, only pass False when you
                        want to start from zero storage for some reason.

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
        print(date, ndate, nT)  # check carefully
        return ndate, nT
    #

    # filtering functions
    def filtering(self, date, nT):
        """
        LETKF at assmilation date

        Args:
            date (datetime.datetime): current date
            nT (int): number of time steps in output time
        """
        statevector = self.const_statevector(nT)
        obs, obserr = self.const_obs()
        xa = self.dacore.letkf(statevector, obs, obserr, self.obsvars,
                               nCPUs=self.nCPUs, smoother=False)
        self.update_states(xa, nT)

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

    # postprocessing functions
    def update_states(self, xa, nT, dtype_f=camaout_dtype):
        """
        update current state based on assimilated results
        by saving files in outdir. Make sure that dtype_f
        is your model bytes. The saved files via this function
        will be used in the next step in the model.
        This is the most case-specific part in DA study.

        Args:
            xa (np.ndarray): analysis array [nvars, eTot, nT, nReach]
            nT (int): time step passed, most rescent t
            dtype_f (np.dtype): data type you want to save.
                                chose this carefully-this should be
                                your model byte precision.
        """
        argsmap = [[xa[:, eNum, -1, :], self.outdir.format(eNum), self.mapdir,
                   self.nlon, self.nlat, nT, eNum, self.nlfp, dtype_f]
                   for eNum in range(self.eTot)]
        p = Pool(self.nCPUs)
        p.map(submit_update_states, argsmap)
        p.close()
    #


# multiprocessing; forwarding functions
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
    varspath = ext.make_vars(modeldir, expname, rnofdir, simrange, eNum,
                             ensrnof=ensrnof, restart=restart)
    subprocess.check_call([camagosh, varspath])
#


# multiprocessing; postprocessing functions
def submit_update_states(args):
    """
    a wrapper to expand args.
    """
    ext.update_states(args[0], args[1], args[2], args[3],
                      args[4], args[5], args[6], args[7],
                      args[8])
