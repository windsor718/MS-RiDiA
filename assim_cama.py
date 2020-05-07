import numpy as np
import json
import datetime
import h5py
import xarray as xr
import pytz
import subprocess
import os
from distutils.util import strtobool
from multiprocessing import Pool
from pyletkf.pyletkf import pyletkf
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

    def register(self, configjson, initialize=True, use_cached_lp=False):
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
        self.camagosh = str(varDict["camagosh"])
        self.outdir = os.path.join(self.modeldir, "out",
                                   self.expname, "{0:02d}")
        self.nlon = int(varDict["nlon"])
        self.nlat = int(varDict["nlat"])
        self.mapdir = str(varDict["mapdir"])
        self.west = float(varDict["west"])
        self.south = float(varDict["south"])
        self.res = float(varDict["res"])
        self.nlfp = int(varDict["nlfp"])
        self.rnofdir = str(varDict["rnofdir"])
        self.ensrnof = bool(strtobool(varDict["ensrnof"]))
        self.eTot = int(varDict["eTot"])
        self.nCPUs = int(varDict["nCPUs"])
        self.statevars = varDict["statevars"]
        self.statedist = varDict["statedist"]
        self.statetype = varDict["statetype"]  # prognostic/parameter
        self.obsnames = varDict["obsnames"]
        self.obsdist = varDict["obsdist"]
        self.obsvars = varDict["obsvars"]  # 1 if available 0 if not.
        self.obsncpath = varDict["obsncpath"]
        self.assimconfig = varDict["assimconfig"]
        self.undef = int(varDict["undef"])
        self.dummyfile = "/home/yi79a/yuta/RiDiA/srcda/MS-RiDiA/buffer.bin"

        # read observations
        self.obs_dset = self.read_observation(self.obsncpath)
        self.assimdates = self.get_assimdates(self.obs_dset)
        vecmappath = str(varDict["vecmappath"])  # map_width.ipynb
        if os.path.exists(vecmappath):
            with h5py.File(vecmappath, "r") as f:
                self.map2vec = f["map2vec"][:]
                self.vec2lat = f["vec2lat"][:]
                self.vec2lon = f["vec2lon"][:]
                self.nvec = len(self.vec2lat)
        else:
            raise IOError("{0} does not exist. You may create this from "
                          "dautils.make_vectorized2dIndex, but make sure "
                          "to match with your observation data label.")

        # instanciate pyletkf;
        # if local patch is not cached, it will be generated.
        self.dacore = pyletkf.LETKF_core(self.assimconfig,
                                         mode="vector",
                                         use_cache=use_cached_lp)
        self.dacore.initialize()

        # check data consistency to avoid mistakes.
        self.check_consistency()

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
        utc = pytz.utc
        sdate = utc.localize(sdate)
        edate = utc.localize(edate)
        if spinup:
            sedate = datetime.datetime(sdate.year+1, 1, 1)
            sedate = utc.localize(sedate)
            self.spinup(sdate, sedate, ensrnof=self.ensrnof)
        date = sdate
        nT = 0
        while date < edate:
            date, nT = self.driver(date, self.obs_dset)

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

    def driver(self, date, obs_dset):
        test0 = np.fromfile(os.path.join(self.outdir.format(1), "param/rivhgt.bin"), np.float32).reshape(self.nlat, self.nlon)
        test1 = np.fromfile(os.path.join(self.outdir.format(2), "param/rivhgt.bin"), np.float32).reshape(self.nlat, self.nlon)
        print("max", test0.max(), test1.max())
        print((test0 - test1).sum(), (test0 - test1).max(), (test0 - test1).min())
        ndate, nT = self.forward(date,
                                 ensrnof=self.ensrnof, restart=True)
        adate = ndate - datetime.timedelta(seconds=86400)
        self.filtering(adate, nT, obs_dset)
        # if ndate.year > date.year:
        #     self.backup_restart(ndate)
        #     with open(os.path.join(self.outdir, "ntlog.txt"), "a") as f:
        #         f.write("{0}, {1}"
        #                 .format(ndate.strftime("%Y%m%d%H"), nT))
        return ndate, nT
    #

    # utilities
    def check_consistency(self):
        """
        checking data shapes to avoid mistakenly use data from
        unintentional source.
        """
        print("checking data consistency:")
        nvec = len(self.vec2lat)
        print("checking local patch loaded...")
        print(len(self.dacore.patches))
        assert len(self.dacore.patches) == nvec, "cached local patch size" +\
            "does not match with vector size."
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

    def read_observation(self, ncpath):
        """
        read observaiton netcdf. See data/MS-RiDiA/src/map_widths.ipynb
        to create this.

        Args:
            ncpath (str): netcdf path

        Returns:
            xarray.Dataset
        """
        return xr.open_dataset(ncpath)

    def get_assimdates(self, obsdset):
        """
        read assimilation date information
        (when observation is available)

        Args:
            obsdset (xr.Dataset): observation dataset

        Returns:
            list
        """
        utc = pytz.utc
        dates = obsdset["time"].values
        # dates.tolist() converts np.datetime64 to posix timestamp
        dtdates = [datetime.datetime.utcfromtimestamp(date/1e9) for date in
                   dates.tolist()]
        dates = [utc.localize(dtdate) for dtdate in dtdates]
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
                ext.gain_perturbation(var, self.outdir, self.mapdir,
                                      self.nlat, self.nlon,
                                      self.eTot)
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
            bpath = os.path.join(resdir,
                                "restart_{0}.bin".format(datestring))
            subprocess.check_call(["cp", respath, bpath])
    #

    # forwarding functions
    def forward(self, date, ensrnof=False, restart=True):
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
        nT = (ndate-date).days
        print(date, ndate, nT)  # check carefully
        return ndate, nT
    #

    # filtering functions
    def filtering(self, date, nT, obs):
        """
        LETKF at assmilation date

        Args:
            date (datetime.datetime): current date
            nT (int): number of time steps in output time
        """
        statevector = self.const_statevector(nT)
        obs, obserr = self.const_obs(obs, date)

        # pyletkf assumes double precision
        statevector = statevector.astype(np.float64)
        obs = obs.astype(np.float64)
        obserr = obs.astype(np.float64)
        xa, _ = self.dacore.letkf_vector(statevector, obs, obserr, self.obsvars,
                                         nCPUs=self.nCPUs, smoother=False)
        self.update_states(xa, nT, date)

    def const_statevector(self, nT):
        """
        construct statevector by reading files.
        variables are specified in register()

        Args:
            nT (int): number of time steps passed, or number of layers
            (REC in Fortran) in your file.

        Returns:
            np.ndarray-like: memory mapped array like object
        """
        # create buffer array, this is used for concatenating memmap objects.
        buffer = np.memmap(self.dummyfile, dtype=np.float32, mode="w+",
                           shape=(len(self.statevars), self.eTot,
                                  nT, self.nvec)
                           )
        for idx, var in enumerate(self.statevars):  # not that many
            for eNum in range(self.eTot):  # not that many
                if self.statetype[idx] == "prognostic":
                    d = dau.load_data3d(os.path.join(self.outdir.format(eNum),
                                                     "{0}.bin".format(var)
                                                     ),
                                         nT, self.nlat, self.nlon,
                                         self.map2vec, self.nvec, dtype=np.float32
                                         )
                else:  # parameter
                    d = dau.load_data3d(os.path.join(self.outdir.format(eNum),
                                                     "param/",
                                                     "{0}.bin".format(var)
                                                     ),
                                         1, self.nlat, self.nlon,
                                         self.map2vec, self.nvec, dtype=np.float32
                                         )
                if self.statedist[idx] == "log":
                    d[d==0] = 1e-8  # replace zero
                    buffer[idx, eNum, :, :] = np.log(d)
                elif self.statedist[idx] == "norm":
                    buffer[idx, eNum, :, :] = d
                else:
                    raise KeyError("undefined distribution: {0}".format(self.statedist[idx]))
        return buffer

    def const_obs(self, obsdset, date):
        """
        parse observation xarray and returns data at the date

        Args:
            obs (xarray.Dataset): observation dataset object
            date (datetime.datetime): date
        """
        obs_values_all = []
        obs_errors_all = []
        for idx, obsname in enumerate(self.obsnames):
            vecids = obsdset[obsname].vecid.values
            obs = obsdset[obsname].sel(time=date,
                                   kind="values").values
            obsall = np.ones([self.nvec], np.float64)*self.undef
            err = obsdset[obsname].sel(time=date,
                                   kind="errors").values
            errall = np.ones([self.nvec], np.float64)*self.undef
            if self.obsdist[idx] == "log":
                undefloc = (obs == self.undef)
                obs[undefloc] = 1.  # temporaly assign positive value incalse undef is negative.
                err[undefloc] = 1.
                logobs = np.log(obs)
                logerr = np.log(err)
                logobs[undefloc] = self.undef
                logerr[undefloc] = self.undef
                obsall[vecids] = logobs
                errall[vecids] = logerr
            elif self.obsdist[idx] == "norm":
                obsall[vecids] = obs
                errall[vecids] = err
            else:
                raise KeyError("undefined distribution: {0}".format(self.obsdist[idx]))
            obs_values_all.append(obsall.reshape(1, -1))
            obs_errors_all.append(errall.reshape(1, -1))
        obs_date_values = np.vstack(obs_values_all)
        obs_date_errors = np.vstack(obs_errors_all)
        return obs_date_values, obs_date_errors

    # postprocessing functions
    def update_states(self, xa, nT, edate, dtype_f=camaout_dtype):
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
        ens0 = xa[1, 1, -1, :]
        ens1 = xa[1, 2, -1, :]
        print((ens0-ens1).sum())
        argsmap = [[xa[:, eNum, -1, :], self.outdir.format(eNum), self.mapdir,
                   self.nlon, self.nlat, nT, self.map2vec, self.vec2lat,
                   self.vec2lon, eNum, self.nlfp, edate, dtype_f]
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
                      args[8], args[9], args[10], args[11], args[12])
