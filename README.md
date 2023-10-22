# GmGM: The Gaussian Multi-Graphical Model

All figures in the paper were generated using one of the `*.ipynb` files in this repository.

```{bash}
# For our code and all experiments
conda env create --file environment.yml
conda activate GmGM

# You will want to compile the fortran subroutines
# for the algorithm
cd path/to/this/repo/...../python_frontend
# Run following command:
../build-tools/build-for-python.sh

# Necessary for the experiments comparing our
# algorithm to prior work, but otherwise
# not needed
cd path/to/matlab/...../Matlab/extern/engines/python
python -m pip install .
```

The repositories of prior work have been cloned into this repo under the folder `other_algs`; if you wish to build them, follow the instructions in those repositories.
