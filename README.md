# ReMoTe-S. Residential Mobility of Tenants in Switzerland: an agent-based model

This repository contains the code for the paper "ReMoTe-S. Residential Mobility of Tenants in Switzerland: an agent-based model".


## Install
To install the required libraries, run the following commands
```
pip install numpy==1.18.3
pip install pandas==1.0.3
pip install scipy==1.6.3
pip install mesa==0.8.8.1
```

The use of a virtual environment (e.g. Anaconda) is not required, but recommended.
Python 3.8 has been used to run the simulations.

## Run
To execute the model, run the following commands inside the root directory of the project
```
cd model
python3 run.py
```

## Sensitivity analysis
To run the sensitivity analysis, execute:
```
cd model
python3 run_sa.py
```

## Output
The results of the simulations can be found in the directory "results". The subdirectory "SA" contains results for the sensitivity analysis, while the other folders contain the results for the different scenarios.

Results are generated as ".csv" files.


## Authors
Authors: Anna Pagani, Francesco Ballestrazzi, Emanuele Massaro, Claudia R. Binder
Laboratory of Human-Environment Relations in Urban Systems (HERUS) @ EPFL, Lausanne, Switzerland