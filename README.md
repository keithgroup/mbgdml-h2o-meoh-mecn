# Many-body machine learning potentials for water, acetonitrile, and methanol

![GitHub repo size](https://img.shields.io/github/repo-size/keithgroup/mbgdml-h2o-meoh-mecn)

Data sets, scripts, and analyses of many-body machine learning (mbML) potentials for water, acetonitrile, and methanol.

## Manifest

Our workflow takes advantage of both NumPy `npz` and `exdir` files along with open-source software developed by members of our group (i.e., [mbgdml](https://github.com/keithgroup/mbGDML) and [reptar](https://github.com/aalexmmaldonado/reptar)).
Below is a short explanation of what resides in each directory which contains other `README` files when necessary.

- `data`: contains (almost) all data pertaining towards the development and application of mbML potentials considered here.
This includes GFN2-xTB MD simulations for $n$-body sampling; MP2/def2-TZVP energies and forces of $n$-body structures and isomers; MD simulations driven by MP2 and various mbML potentials.
Due to [GitHub file size limitations](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#file-size-limits), the large MD trajectory files are [archived on Zenodo](https://doi.org/10.5281/zenodo.7112198).
- `training-logs`: contains [GDML](https://github.com/keithgroup/mbGDML), [GAP](https://libatoms.github.io/GAP/), and [SchNet](https://schnetpack.readthedocs.io/en/stable/) training scripts and logs for 1-, 2-, and 3-body models for the solvents considered here.
The resulting models are [archived on Zenodo](https://doi.org/10.5281/zenodo.7112163).
- `scripts`: all Python scripts used to prepare the manuscript.
This includes scripts to train models, run molecular dynamics simulations, convert file types, analyze model predictions, create plots, etc.
All Python scripts that generate figures with matplotlib are labeled with a `figure-` prefix.
- `analysis`: mainly figures used for the results and discussion along with some postprocessing data.

## Reproducibility

All information, data, and figures presented in our manuscript can be directly reproduced with the relevant code and data stored in this repository.
Obviously there are inherent limitations on reproducibility such as different environments, computers, and long-term hosting of these repositories.
We cannot do much about the last two, but we do provide a `requirements.txt` file the specifies packages and versions we used with Python 3.10.4 on Ubuntu 20.04.

As mentioned previously, we were not able to fit every file into this repository.
[Trained models](https://doi.org/10.5281/zenodo.7112163) and [MD simulation data](https://doi.org/10.5281/zenodo.7112198) are archived separately on Zenodo.
These repositories would need to be downloaded and extracted in the same directory as this repository.
You may need to adjust the relative paths to data in the provided scripts.

If you are trying to reproduce this work, we thank you for your service to the academic community!
We tried to be as transparent about our data, code, and analyses as possible; however, please contact the corresponding authors with any questions or difficulties.

## License

[![CC BY 4.0][cc-by-4.0-shield]][cc-by-4.0]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by-4.0].

[cc-by-4.0]: http://creativecommons.org/licenses/by/4.0/
[cc-by-4.0-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
