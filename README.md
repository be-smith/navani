# navani

Navani is a Python module for processing and plotting electrochemical data from battery cyclers, combining other open source libraries to create pandas dataframes with a normalized schema across multiple cycler brands. It is intended to be easy to use for those unfamiliar with programming.
Contains functions to compute dQ/dV and dV/dQ.

Full documentation can be found [here](https://be-smith.github.io/navani/).

Currently supports:

- BioLogic MPR (`.mpr`)
- Arbin res files (`.res`)
- Simple `.txt` and Excel `.xls`/`.xlsx` formats produced by e.g., Arbin, Ivium and Lanhe/Lande
- Neware NDA and NDAX (`.nda`, `.ndax`)

The main dependencies are:

- pandas
- [galvani](https://github.com/echemdata/galvani) (BioLogic MPR) 
- [mdbtools](https://github.com/mdbtools/mdbtools) (for reading Arbin's .res files with galvani).
- [NewareNDA](https://github.com/Solid-Energy-Systems/NewareNDA) (for reading Neware's NDA and NDAx formats).

Navani is released under the terms of the MIT license. 

> [!WARNING]
> The [galvani](https://github.com/echemdata/galvani) dependency is available under the terms of [GPLv3 License](https://github.com/echemdata/galvani/blob/master/LICENSE). We believe this usage to be valid following the GPLv3 interpretation of the [copyright holder for galvani](https://github.com/echemdata/galvani/issues/51#issuecomment-701500053). The galvani library is not distributed with Navani, but installing Navani from PyPI will also install GPL-licensed dependencies. Users are responsible for GPL compliance of any downstream projects in this regard.


## Installation

You will need Python 3.10 or higher to use Navani.

Navani can now be installed using pip:

```shell
pip install navani
```

However it is still advised to install navani using [uv](https://docs.astral.sh/uv/), to manage dependencies.

To install Navani and its dependencies, clone this repository and use uv to setup a virtual environment with the dependencies:

```shell
git clone git@github.com/be-smith/navani
cd navani
uv venv
uv sync
```

You should now have an environment you can activate with all the required dependencies (except mdbtools, which is covered later).

To activate this environment simply run from the navani folder:

```shell
source .venv/bin/activate
```

If you would like to contribute to navani it is recommended to install the dev dependencies, this can be done simply by:

```shell
uv sync --all-extras --dev
```

If don't want to use uv it is still strongly recommended to use a fresh Python environment to install navani, using e.g., `conda create` or `python -m venv <chosen directory`.
To install navani, either clone this repository and install from your local copy:

```shell
git clone git@github.com/be-smith/navani
cd navani
pip install .
```

The additional non-Python mdbtools dependency to `galvani` that is required to read Arbin's `.res` format can be installed on Ubuntu via `sudo apt install mdbtools`, with similar instructions available for other Linux distributions and macOS [here](https://github.com/mdbtools/mdbtools).

## Usage

The main entry point to navani is the `navani.echem.echem_file_loader` function, which will do file type detection and return a pandas dataframe.
Many different plot types are then available, as shown below:

```python
import pandas as pd
import navani.echem as ec

df = ec.echem_file_loader(filepath)
fig, ax = ec.charge_discharge_plot(df, 1)
```
<img src="https://github.com/be-smith/navani/raw/main/docs/Example_figures/Graphite_charge_discharge_plot.png" alt="Graphite charge discharge plot example" width="50%" height="50%">

Also included are functions for extracting dQ/dV from the data:

```python
for cycle in [1, 2]:
    mask = df['half cycle'] == cycle
    voltage, dqdv, capacity = ec.dqdv_single_cycle(df['Capacity'][mask], df['Voltage'][mask],
                                                   window_size_1=51,
                                                    polyorder_1=5,
                                                    s_spline=0.0,
                                                    window_size_2=51,
                                                    polyorder_2=5,
                                                    final_smooth=True)
    plt.plot(voltage, dqdv)

plt.xlim(0, 0.5)
plt.xlabel('Voltage / V')
plt.ylabel('dQ/dV / mAhV$^{-1}$')
```

<img src="https://github.com/be-smith/navani/raw/main/docs/Example_figures/Graphite_dqdv.png" alt="Graphite dQ/dV plot example" width="50%" height="50%">

And easily plotting multiple cycles:
```python
fig, ax = ec.multi_dqdv_plot(df, cycles=cycles,
                    colormap='plasma',
                    window_size_1=51,
                    polyorder_1=5,
                    s_spline=1e-7,
                    window_size_2=251,
                    polyorder_2=5,
                    final_smooth=True)
```
<img src="https://github.com/be-smith/navani/raw/main/docs/Example_figures/Si_dQdV.png" alt="Si dQ/dV plot example" width="50%" height="50%">

Simple jupyter notebooks and Colab notebooks can be found [here](https://github.com/be-smith/navani/blob/main/Simple%20example%20jupyter.ipynb) for Jupyter and [here](https://github.com/be-smith/navani/blob/main/Simple_examples_colab.ipynb) for Colab.

Whilst a more detailed Colab notebook can be found [here](https://github.com/be-smith/navani/blob/main/Detailed_colab_tutorial.ipynb).
