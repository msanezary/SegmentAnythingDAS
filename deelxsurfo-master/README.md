#hpmv #surfo #deel

# Context
Examples and tutorials to read & process SURFO HDF5 datasets using XDAS.
XDAS is a public Python library for processing Distributed Acoustic Sensing data.

This repo contains minimal examples, please refer to XDAS if you wish to pursue using it:<br>
https://xdas.readthedocs.io/en/latest/ <br>
https://github.com/xdas-dev/xdas <br>
https://github.com/xdas-dev/tutorials


# How to run this repo:
### With a regular conda env

1. Create and activate a barebone Python 3.12 environement

```bash
# Example with conda
conda create -n MyEnv python=3.11
conda activate MyEnv
```

2. Install project's dependencies

```bash
pip install -r env/requirements.txt
python -m pip install -e .
```

3. Install jupyterlab (or another IDE)
```bash
conda install -c conda-forge jupyterlab
```

4. Explore the tutorial notebook in the `./app` folder.


# TODOs:
- [ ] README
- [x] Reading single files with XDAS
- [x] Reading multiple files with XDAS
- [x] XDAS slicing with time/space coordinates (tutorial)
- [x] XDAS single transform (tutorial)
- [x] XDAS pipelining (tutorial)
- [x] Add minimum metadata using HPMV pipeline
- [ ] Generate sample data from HPMV pipeline
