# Simulations for "When Will an Elevator Arrive?"
This repository gives the simulations used in the paper "When Will an Elevator Arrive?" https://arxiv.org/abs/2012.01471. All programs are written in Python, with optional Cython optimisation.

## Preparation
### Cython
To use Cython optimization, install Cython through https://cython.readthedocs.io/en/latest/src/quickstart/install.html.
Before running any of the programs, run the commend `python transport_setup.py build_ext --inplace` (once) in terminal under the same repository of the code. 
The files "transport.pyx" and "transport_setup.py" are part of the Cython implementation. To use native Python without Cython, simply delete the lines "from mytransport import transport"
And uncomment function definition of "transport".

### fast-histogram
For plotting histograms, we used fast-histogram package that can be installed from https://pypi.org/project/fast-histogram/. In the situation of plotting large amount of data, it significantly improves the efficiency compared to matplotlib.

## Simulation
The code imcludes all simulation used in the paper. The implementation of theretical calculation is not necessarily included. The model parameters are set as used in the paper, and can be modified by changing the values. 
### Infinite capacity
The file "elevator simulation finite capacity" contains both the iterative solution and the simulation result of interested distributions. 
### Finite capacity
The file "elevator simulation inite capacity" contains the simulation and outputs a key figure to show synchronization. All relevent data are stored in lists or numpy array for plotting and further analysis. The file "clearing probability" is a simplified version of the same simulation that only record the clearing probaility.

## Citation

`@article{ elevator2020,
  title={When Will an Elevator Arrive?},
  author={Zhijie Feng, Sidney Redner},
  journal={arXiv preprint},
  year={2020}`
}
