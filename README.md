# Simulations for "When Will an Elevator Arrive?"
This repository gives the simulations used in the paper "When Will an Elevator Arrive?" https://arxiv.org/abs/2012.01471. All programs are written in Python, with optional Cython optimisation.

## Preparation
### Cython
To use Cython optimization, install Cython through https://cython.readthedocs.io/en/latest/src/quickstart/install.html.
Before running any of the programs, type in the commend `python transport_setup.py build_ext --inplace` in terminal under the same repository of the code. 
The files "transport.pyx" and "transport_setup.py" are part of the Cython implementation. To use python without Cython, simply delete the lines "from mytransport import transport"
And uncomment function definition of "transport".

### fast-histogram
For plotting histograms, we used fast-histogram package that can be installed from https://pypi.org/project/fast-histogram/. In the situation of plotting large amount of data, it significantly improves the efficiency compared to matplotlib.

