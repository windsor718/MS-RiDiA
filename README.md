# River Discharge reAnalysis via optical satellite images from Landsat  
## Missouri Case  
This is a repo. for a study of:  
**River Discharge reAnalysis using historical optical satellite images from Landsat from 1984.**  
Basic settings:
- Runoff: Princeton's hydrological reconstrunction; VIC outputs (Lin et al., 2019)
- Hydraulic model: CaMa-Flood (Yamazaki et al., 2012)
- Satellite images: Landsat optical images from 1984 to 2016.  
- state variables: width, Manning's n, Dingman's river cross sectional shape parameter  
- Data Assimilation algorithm: Local Ensemble Transformed Kalman Filter  

Expensive parts of those code sets are written in fortran and Cython (fully static typed; mumoryviewed; parallelized via openMP),
and most of IO part is lazy-loading using numpy memory mapping. Global Ready.  
  
## Code structure:  
- [pyletkf](https://github.com/windsor718/pyletkf): python/Cython implementation of LETKF.
- assim_cama.py: main interface for data assimilation using CaMa-Flood. This can be re-used for any kind of data assimilation study using pyletkf.  
- caseExtention.py: code collecting functions havily dependent on each experiment setting. Edit this file to make interface for your experiment.  
- dautils.py: functions used frequently in pyletkf interection. Maybe be included in pyletkf in future updates.  
- cysrc: cython source codes  
- fsrc: fortran90 source codes  
