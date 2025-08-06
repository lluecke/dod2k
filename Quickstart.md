How to get started with the dod2k environment, functions, notebooks and products.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

0.0 Get the project: in a working directory,

```
git clone https://github.com/lluecke/dod2k.git
```

1.0. Create the environment: in dod2k/, 

```
conda create --name dod2k-env -c conda-forge
conda activate dod2k-env
conda install pandas numpy matplotlib cartopy scipy=1.6.0 geopy xarray
pip install lipd tqdm rasterio pyproj netcdf4 cfr scikit-learn rioxarray legacy-cgi
pip install git+https://github.com/sylvia-dee/PRYSM.git
```

1.1. To export the environment as a .yml file:

```
conda env export > dod2k-env.yml
```

1.2. To add it as a virtual environment in jupyterhub: in a jupyterhub-username terminal window,

```
pip install --user virtualenv
virtualenv dod2k-env
pip install --user ipykernel
python -m ipykernel install --user --name=dod2k-env --display-name "Python (dod2k-env)"
```

2.0. Test load scripts:

```
pages2k/load_pages2k_v2.ipynb
fe23/load_fe23.ipynb
iso2k/load_iso2k.ipynb
sisal/load_sisal_v7.ipynb
ch2k/load_ch2k_v4.ipynb
```

3.0. test filtering, info, plotting, detection/decision/removal, analysis notebooks:

```
df_filter.ipynb
df_info.ipynb
df_plot_dod2k_v1.ipynb
dup_detection.ipynb 
dup_decision.ipynb
dup_removal.ipynb
M_analysis_v0.3.ipynb
MT_analysis_v9.3.ipynb
T_analysis_v9.3.ipynb
```

4.0. To run ```S_analysis_v1.6.ipynb``` you will first need to create the directory ```speleothem_modeling_inputs```, and download into it data from their source urls:

```
mkdir speleothem_modeling_inputs
cd speleothem_modeling_inputs
wget https://wateriso.utah.edu/waterisotopes/media/ArcGrids/GlobalPrecip.zip
unzip GlobalPrecip.zip
wget https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.07/cruts.2304141047.v4.07/tmp/cru_ts4.07.1901.2022.tmp.dat.nc.gz
gunzip cru_ts4.07.1901.2022.tmp.dat.nc.gz
```



