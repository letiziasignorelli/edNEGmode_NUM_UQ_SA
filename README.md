# Efficient uncertainty quantification and sensitivity analysis of electrodiffusive neuron models

This code was used to produce the results presented in Signorelli, L. (2023): \
[Efficient Uncertainty Quantification and Sensitivity Analysis of Electrodiffusive Neuron Models](https://hdl.handle.net/10589/211026).

The model studied was presented in Sætra et al., *PLoS Computational Biology*, 17(7), e1008143 (2021):\
[An electrodiffusive neuron-extracellular-glia model for exploring
the genesis of slow potentials in the brain](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008143). 

The code was developed on top of the followings: 
- https://github.com/CINPLA/edNEGmodel_analysis
- https://github.com/CINPLA/edNEGmodel

# Installation

To install the code, clone or download the repo, navigate to the top directory of the repo and enter the following command
in the terminal: 
```bash
python setup.py install
```

The code was run with Windows 10 Home and Python 3.8.16.

If you have problems reading the initial_values.npz-file, install git lfs and try `git-lfs pull`.
