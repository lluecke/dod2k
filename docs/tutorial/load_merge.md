# **Load the original databases and merge to a common dataframe.**

This tutorial shows you how to load the original databases from source and merge them to a common dataframe (ready for duplicate detection).


## **Load the original databases from source**

The original databases are each loaded via their individual *load notebooks* (`load_ds.ipynb`, `ds` as below). These notebooks load each database from source and create a standardised 'compact' `pandas` dataframe, with the following columns:

  - ```archiveType```
  - ```dataSetName```
  - ```datasetId```
  - ```geo_meanElev```
  - ```geo_meanLat```
  - ```geo_meanLon```
  - ```geo_siteName```
  - ```interpretation_direction``` (new in v2.0)
  - ```interpretation_variable```
  - ```interpretation_variableDetail```
  - ```interpretation_seasonality``` (new in v2.0)
  - ```originalDataURL```
  - ```originalDatabase```
  - ```paleoData_notes```
  - ```paleoData_proxy```
  - ```paleoData_sensorSpecies```
  - ```paleoData_units```
  - ```paleoData_values```
  - ```paleoData_variableName```
  - ```year```
  - ```yearUnits```

The intermediate data is saved in the `data` directory, where each database has its own subdirectory. For any database `db`, the output is saved in `data/db` under:

- `db_compact.pkl`: `pickle` file for easy python processing, read with ```pd.DataFrame(pd.read_pickle(db_compact.pkl))```
- `db_compact_metadata.csv`: `csv` file which includes all the metadata associated with each record
- `db_compact_year.csv`: `csv` file which includes the time coordinates for each record
- `db_compact_paleoData_values.csv`: `csv` file which includes the actual record data for each record

Further information about each individual load notebook can be found here:



??? info  "Load PAGES 2k data from source"


    The interactive notebook to load PAGES 2k can be found here: [load_pages2k.ipynb](../notebooks/load_pages2k.ipynb).
    
    This notebook loads PAGES 2k data from LiPDverse, currently version 2.2.0, and creates a standardised 'compact dataframe'. 

    In order to load the data from source please activate the following cell:
    
    ```python title='python3/Jupyter'
    # Download the file
    
    !wget -O data/pages2k/Pages2kTemperature2_2_0.pkl https://lipdverse.org/Pages2kTemperature/current_version/Pages2kTemperature2_2_0.pkl
    ```
    
    This will download the pickle file `Pages2kTemperature2_2_0.pkl` into `data/pages2k`.

    Subsequently, run the interactive notebook to create a set of `csv` files containing the compact dataframe.


??? info "Load FE23 data from source"
    
    The interactive notebook to load FE23 can be found here: [load_fe23.ipynb](../notebooks/load_fe23.ipynb).
    
    This notebook loads FE23 from the NCEI and creates a standardised 'compact dataframe'. 

    In order to load the data from source please activate the following cell:
    
    ```python title='python3/Jupyter'
    # download and unzip FE23 
    !wget -O /data/fe23/franke2022-fe23.nc https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/franke2022/franke2022-fe23.nc
    fe23_full  = xr.open_dataset('fe23/franke2022-fe23.nc')
    
    # save slice of FE23 with only relevant variables as netCDF (fe23_full is 25GB)
    fe23_slice = fe23_full[vars]
    fe23_slice.to_netcdf('data/fe23/franke2022-fe23_slice.nc')
    ```
    
    This will download the netCDF file `franke2022-fe23.nc` into `data/fe23`. Note that this is a very large dataset (around 25GB) so it might be useful for the operator to slice a small list of variables of the large file and save it instead.

    Subsequently, run the interactive notebook to create a set of `csv` files containing the compact dataframe.




??? info "Load Iso2k data from source"

    The interactive notebook to load Iso2k can be found here: [load_iso2k.ipynb](../notebooks/load_iso2k.ipynb).
    
    This notebook loads Iso2k from the LiPDverse and creates a standardised 'compact dataframe'. 

    In order to load the data from source please activate the following cell:
    
    ```python title='python3/Jupyter'
    # Download the file (use -O to specify output filename)
    !wget -O data/iso2k/iso2k1_1_2.zip https://lipdverse.org/iso2k/current_version/iso2k1_1_2.zip
    
    # Unzip to the correct destination
    !unzip data/iso2k/iso2k1_1_2.zip -d data/iso2k/iso2k1_1_2
    ```
    
    This will download the zip file `iso2k1_1_2.zip` into `data/iso2k` and unzip into the directory `data/iso2k/iso2k1_1_2`.

    Subsequently, run the interactive notebook to create a set of `csv` files containing the compact dataframe.




??? info "Load SISAL data from source"

    The interactive notebook to load SISAL can be found here: [load_sisal.ipynb](../notebooks/load_sisal.ipynb).
    
    This notebook loads SISAL from a set of `csv` files. 

    The current version (v3) was made available to the authorsd but is not currently uploaded on the LiPDverse homepage yet. We will update the notebook and documentation accordingly once there are updates on the public availability. 

    Subsequently, run the interactive notebook to create a set of `csv` files containing the compact dataframe.



??? info "Load CoralHydro2k data from source"


    The interactive notebook to load CoralHydro2k can be found here: [load_ch2k.ipynb](../notebooks/load_ch2k.ipynb).
    
    This notebook loads CoralHydro2k from the LiPDverse and creates a standardised 'compact dataframe'. 
    
    In order to load the data from source please activate the following cell:
    
    ```python title='python3/Jupyter'
    # Download the file (use -O to specify output filename)
    !wget -O data/ch2k/CoralHydro2k1_0_1.zip https://lipdverse.org/CoralHydro2k/current_version/CoralHydro2k1_0_1.zip
    
    # Unzip to the correct destination
    !unzip data/ch2k/CoralHydro2k1_0_1.zip -d data/ch2k/ch2k_101
    ```
    
    This will download the zip file `CoralHydro2k1_0_1.zip` into `data/ch2k` and unzip into the directory `data/ch2k/ch2k_101`.

    Subsequently, run the interactive notebook to create a set of `csv` files containing the compact dataframe.



## **Merge the databases to a common dataframe.**

The individual databases can be merged into a common dataframe by loading all the compact dataframes, created in the *load notebooks* (see previous section). 

Simply run the notebook [merge_databases.ipynb](../notebooks/merge_databases.ipynb). This notebook loads the compact dataframes and concatenates them via 


```python
# read compact dataframes from all the single databases

dataset_names = ['pages2k', 'fe23', 'ch2k', 'iso2k', 'sisal' ]

print(dataset_names[0])
df = utf.load_compact_dataframe_from_csv(dataset_names[0])
print('length: ', len(df))

for ii, dn in enumerate(dataset_names[1:]):
    print(f'add {dn}')
    new_df = utf.load_compact_dataframe_from_csv(dn)
    df = pd.concat([df, new_df])
    print('length: ', len(df))

print('---------------')
print('RESULT:')
print(df.info())
```

which creates the following dataframe:
```text title="Output"
<class 'pandas.core.frame.DataFrame'>
Index: 5879 entries, 0 to 545
Data columns (total 21 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   archiveType                    5879 non-null   object 
 1   dataSetName                    5879 non-null   object 
 2   datasetId                      5879 non-null   object 
 3   geo_meanElev                   5780 non-null   float32
 4   geo_meanLat                    5879 non-null   float32
 5   geo_meanLon                    5879 non-null   float32
 6   geo_siteName                   5879 non-null   object 
 7   interpretation_direction       5879 non-null   object 
 8   interpretation_seasonality     5879 non-null   object 
 9   interpretation_variable        5879 non-null   object 
 10  interpretation_variableDetail  5879 non-null   object 
 11  originalDataURL                5879 non-null   object 
 12  originalDatabase               5879 non-null   object 
 13  paleoData_notes                5879 non-null   object 
 14  paleoData_proxy                5879 non-null   object 
 15  paleoData_sensorSpecies        5879 non-null   object 
 16  paleoData_units                5879 non-null   object 
 17  paleoData_values               5879 non-null   object 
 18  paleoData_variableName         5879 non-null   object 
 19  year                           5879 non-null   object 
 20  yearUnits                      5879 non-null   object 
dtypes: float32(3), object(18)
memory usage: 941.6+ KB
None
```

This dataframe is saved in the directory `data/all_merged`, and can be used as input for [duplicate detection](duplicate.md).



