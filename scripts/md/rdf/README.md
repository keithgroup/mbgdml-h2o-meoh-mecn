# Radial distribution functions

Periodic NVT MD simulation were performed to evaluate the dynamic accuracy of our many-body machine learning potentials.
Only mbGDML was considered as it showed consistent accuracy across system sizes for all three solvents.

## Computation

Radial distribution functions (rdfs) were computed using the `mbgdml.analysis.rdf.RDF` class.
Python scripts to compute each rdf curve are given in the `compute` directory.
Because MD simulations are commonly restarted, they require a single `npy` file that contains all coordinates.
These files can become rather large&mdash;on the order of a few hundred MB.
Since GitHub's [file size limit is 100 MB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#file-size-limits) we exclude these files.

These files can be created (and ignored by git) with the `write-combined-R-from-exdir.py` script in the `utils` directory.
Settings for each solvent MD simulation are shown below.

- **Water**

    ```python
    exdir_path = 'h2o/137h2o-mbgdml-md.exdir'
    group_keys = ['1-nvt', '2-nvt']
    npy_path = 'h2o/137h2o-mbgdml-nvt_1_2.npy'
    ```

- **Acetonitrile**

    ```python
    exdir_path = 'mecn/67mecn-mbgdml-md.exdir'
    group_keys = ['1-nvt-298', '2-nvt-298', '3-nvt-298']
    npy_path = 'mecn/67mecn-mbgdml-nvt_1_2_3-298.npy'
    ```

- **Methanol**

    ```python
    exdir_path = 'meoh/61meoh-mbgdml-md.exdir'
    group_keys = ['1-nvt', '2-nvt', '3-nvt']
    npy_path = 'meoh/61meoh-mbgdml-nvt_1_2_3.npy'
    ```

## Plotting

Once the rdf computation script is completed, there will be a `npz` file that contains bins and $g(r)$.
Scripts in the `plot` directory read data in this `npz` file and plot the computed rdf in comparison to literature sources stored in the `data/external/` directory.
