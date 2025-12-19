# **Applications**


These notebooks are tailored to analyse the database filtered for specific variables, such as `interpretation_variable`, to investigate the climatic properties of certain archive types. 


## PCA on database filtered for `interpretation_variable`


### Moisture and moisture-temperature records only

This notebook performs a principal component analysis (PCA) on moisture and moisture-temperature sensitive records in DoD 2k.

For each subset, the following algorithm is being used:

#### **1. Loads the filtered database for moisture and moisture-temperature records only**
   
```python title='python3/Jupyter'
db_name = 'dod2k_v2.0_filtered_M_TM'

# load dataframe
df = utf.load_compact_dataframe_from_csv(db_name)
print(df.info())
df.name = db_name
```


#### **2. Filter `archive_type` and `paleoData_proxy`**
Filter (defines proxy subset) and produces summary plots of the data, in particular regarding: coverage, resolution and length of the records. This gives us information for the next step, in which we need to choose the parameters for the PCA

!!! example

    For `archiveType` Wood and `paleoData_proxy` ring width, do:

    ```python title='python3/Jupyter'
    # (1) filter for archiveType and/or paleoData_proxy: 
    
    at = 'Wood'
    pt = 'ring width'
    key = '%s_%s'%(at, pt)
    ```

    
```python title='python3/Jupyter'
# filter records for specific archive and proxy type
df_proxy = df.copy().loc[(df['archiveType']==at)].loc[(df['paleoData_proxy']==pt)] 

n_recs = len(df_proxy) # number of records
print('n_records   : ', n_recs)

print('archive type: ', set(df_proxy['archiveType']))
print('proxy type:   ', set(df_proxy['paleoData_proxy']))

# (2) plot the spatial distribution of records
geo_fig, col = uplt.geo_plot(df_proxy, return_col=True)

# (3) plot the coverage for proxy types and plot resolution

uta.convert_subannual_to_annual_res(df_proxy)

df_proxy = uta.add_auxvars_plot_summary(df_proxy, key, col=col[at])
```
    

   
#### **3. Define proxy specific parameters for the PCA**
- period (start and end year): choose a period of sufficient data density (all records chosen for the analysis need to at least overlap during this period)
- minimum resolution: records exceeding this resolution are being excluded from the analysis. Records with higher resolution will be subsampled to create homogeneous resolution across all the records.
- record length: records shorter than the record length are being excluded from the analysis.
- The choice of parameters will determine the success of the PCA. There is a trade-off between the number of records included and the quality (i.e. period/record length/resolution).
- Summary figures are being produced for the filtered data
- z-scores added to dataframe (mean=0 and std=1 over the entire record) as 'paleoData_zscores'
- note: z-scores may be biased if records are only partly overlapping in time, or increase in availability over time, or both.
  
!!! example

    For `archiveType` Wood and `paleoData_proxy` ring width, do:
   
    ```python title='python3/Jupyter'
    #========================= PROXY SPECIFIC: Wood ring width =========================
    minres    = 1                         # homogenised resolution
    mny       = 1000                      # start year of homogenised time coord
    mxy       = 2000                      # end year of homogenised time coord
    nyears    = np.min([600, mxy-mny])    # minimum length of each record
    #====================================================================
    ```


```python title='python3/Jupyter'

# filter for record length during target period
df_proxy = uta.filter_record_length(df_proxy, nyears, mny, mxy)

# filter for resolution
df_proxy = uta.filter_resolution(df_proxy, minres)

# plot coverage and resolution
uplt.plot_coverage_analysis(df_proxy, np.arange(mny, mxy+minres, minres), key, col[at])
uplt.plot_resolution(df_proxy, key, col=col[at])
uplt.plot_length(df_proxy, key, col=col[at])

n_recs = len(df_proxy) # final number of records


print(df_proxy[['miny', 'maxy', 'originalDatabase']])

pca_rec[key] = df_proxy['datasetId']

# add 'z-scores' to dataframe and plot z-scores and values
df_proxy = uta.add_zscores_plot(df_proxy, key, plot_output=True)

```
   
#### **4. Homogenise data dimensions across the records**
- defines a homogenised time variable over the target period and with the target resolution (as defined in the last step), which is saved as a new column in the dataframe named 'years_hom'
- creates a data matrix with dimensions n_records x n_time which is saved as a new column in df, named `paleoData_values_hom` and `paleoData_zscores_hom`.
- Note that this data is formatted as a np.ma.masked_array, where missing data is set to zero and masked out.
  
```python title='python3/Jupyter'
# define new homogenised time coordinate
df_proxy, years_hom = uta.homogenise_time(df_proxy, mny, mxy, minres)
time[key] = years_hom

# assign the paleoData_values to the non-missing values in the homogenised data array

out = uta.homogenise_data_dimensions(df_proxy, years_hom, key, plot_output=False) # returns list of homogenised paleoData_values and paleoData_zscores
paleoData_values_hom, paleoData_zscores_hom, year_hom_avbl, zsco_hom_avbl = out

# define new columns in df_filt
new_columns = {'year_hom': [years_hom]*n_recs, 
               'year_hom_avbl': year_hom_avbl, 
               'paleoData_values_hom': [paleoData_values_hom[ii, :] for ii in range(n_recs)], 
               'paleoData_zscores_hom': [paleoData_zscores_hom[ii, :] for ii in range(n_recs)], 
               'paleoData_zscores_hom_avbl': zsco_hom_avbl}
df_proxy = df_proxy.assign(**new_columns)

print('Real intersect after homogenising resolution: ')
intrsct = uta.find_shared_period(df_proxy, minmax=(mny, mxy), time='year_hom_avbl', 
                             data='paleoData_zscores_hom_avbl')
data[key] = paleoData_zscores_hom
```
  
      
#### **5. PCA**
- obtain covariance matrix of `paleoData_zscores_hom` (note that for every two records the covariance is calculated over their intersect of data availability)
- obtain eigenvectors and eigenvalues via SVD composition
- obtain and plots fraction of explained variance, first two PCs and load for first two EOFs vs ordering in the data frame.
      
```python title='python3/Jupyter'
covariance, overlap = uta.calc_covariance_matrix(df_proxy)

eigenvalues, eigenvectors = uta.PCA(covariance)

foev[key] = uta.fraction_of_explained_var(covariance, eigenvalues, n_recs, key, df.name)

PCs[key], EOFs[key] = uplt.plot_PCs(years_hom, eigenvectors, paleoData_zscores_hom, key, df.name)
```


#### **6. Collect the information for several proxy types**
Repeat the first steps for several proxy types, collect the data in a dictionary, then generate summary figures:


<figure markdown="span">
  ![PCA](../assets/images/MT_PCA.png){ width="800" }
  <figcaption>Figure 1: Summary figure of the PCA.</figcaption>
</figure>

<figure markdown="span">
  ![EOF 1 load](../assets/images/MT_spatial_EOF1.png){ width="800" }
  <figcaption>Figure 2: EOF 1 load.</figcaption>
</figure>

<figure markdown="span">
  ![EOF 2 load](../assets/images/MT_spatial_EOF2.png){ width="800" }
  <figcaption>Figure 3: EOF 2 load.</figcaption>
</figure>




See the notebook [analysis_moisttemp.ipynb](../notebooks/analysis_moisttemp.ipynb)



------

## PSM on database filtered for `archiveType` and `paleoData_proxy`

### Speleothem $\delta^{18}O$ calcite records


This notebook reads the DoD2k, filters for late 20th century speleothem $\delta^{18}O$ records, and uses CRUTS4.07 gridded temperature and precipitated amount-weighted mean annual $\delta^{18}O$ of precipitation estimated by Bowen and Ravenaugh (2003, updated) to simulate o18 of speleothem calcite and compare results across a spatial gradient, to observations.

Note that to execute this notebook, you will need to first download and unzip source input data from Harris et al (2020) and Bowen and Ravenough (2003, updated) in the directory speleothem_modeling_inputs.  See the Bibliography cell at bottom of this notebook and the file Quickstart.md for URLs and more information.



For further information see the notebook [analysis_speleothem.ipynb](../notebooks/analysis_speleothem.ipynb)



