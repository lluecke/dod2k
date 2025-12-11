# Installation

This page explains how to set up the **DoD2k** Python package and its dependencies.

---

## Requirements

- Python 3.9+
- Conda
- JupyterLab or Jupyter Notebook
- Required Python packages (see `dod2k-env.yml`)

---

## Setup

1. **Clone the repository**
   
    ```bash
    git clone https://github.com/lluecke/dod2k.git
    cd dod2k
    ```

2. **Create the Conda environment from the YAML file:**
   
    ```bash
   conda env create -f dod2k-environment.yml
    ```

3. **Activate the environment:**
   
    ```bash
    conda activate dod2k
    ```

4. **Launch JupyterLab or Jupyter Notebook:**
   
    ```bash
    jupyter lab
    ```
    or 
    ```bash
    jupyter notebook
    ```


## Next Steps

Once the environment is set up:

- Explore the [Quickstart Tutorial](quickstart.md)
- Check out the [Tutorials](../tutorial/index.md) to e.g.
    - Load and visualise DoD2k and use for data analysis
    - Generate DoD2k from scratch
    - Run a duplicate detection workflow on the merged database
- Run example notebooks:
    - [df_info.ipynb](../notebooks/df_info.ipynb)
    - [df_plot_dod2k.ipynb](../notebooks/df_plot_dod2k.ipynb)
- Check the [API Reference](../api/index.md) for detailed module documentation



