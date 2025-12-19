# **Loading the DoD2k Database**

This tutorial shows you how to load and explore the DoD2k paleoclimate database.

## **What is DoD2k?**

DoD2k (Database of Databases 2k) integrates five major paleoclimate databases:


| Database | Version | Data | Reference | Records | Archives
|----------|---------|---------|---------------|---------------|---------------|
| **PAGES 2k** | v2.2.0 | [LiPDverse](https://lipdverse.org/Pages2kTemperature/current_version/) | [PAGES 2k Consortium 2017](https://www.nature.com/articles/sdata201788]) | 1364| multi-proxy |
| **SISAL** | v3 | [ORA](https://ora.ox.ac.uk/objects/uuid:1e91e2ac-ca9f-46e5-85f3-8d82d4d3cfd4) | [Kaushal et al. 2024](https://essd.copernicus.org/articles/16/1933/2024/) | 546 | speleothems |
| **Iso2k** | v1.1.2 | [LiPDverse](https://lipdverse.org/iso2k/current_version/) | [Konecky et al. 2020](https://essd.copernicus.org/articles/12/2261/2020/) |435 | multi-proxy|
| **CoralHydro2k** | v1.0.1 | [LiPDverse](https://lipdverse.org/CoralHydro2k/current_version/) | [Walter et al. 2023](https://essd.copernicus.org/articles/15/2081/2023/) |221| corals|
| **FE23** | - | [NCEI](https://www.ncei.noaa.gov/access/paleo-search/study/36773) | [Evans et al. 2022](https://cp.copernicus.org/articles/18/2583/2022/) |2754| tree-rings |



Since these databases may share a number of records, these databases were subject to a **duplicate detection and removal process**. The resulting output is **DoD2k**.


## **The database**

The database is saved in `root_dir/data/dod2k` and provided in two data formats:

- pickle format (fast, python only):
  ```
  dod2k_compact.pkl
  ```
- comma seperated value format (portable, readable)
  ```
  dod2k_compact_year.csv
  dod2k_compact_metadata.csv
  dod2k_compact_paleoData_values.csv
  ```

In this directory we also provide a text file 
```
dod2k_compact_README.txt
```
which shows the details of the duplicate screening process (date, operator credentials, notes).

**Which Format Should I Use?**

- CSV: Best for interoperability and inspecting data manually
- Pickle: Faster loading, preserves numpy arrays without conversion - python only!!!


Here we explain how to load these files into a pandas dataframe using python.

## **Loading the database from csv**


### Step 1: Set up your environment
Start in the repository root directory (`dod2k/`). From here import

```python title='python3/Jupyter'
import sys
from pathlib import Path

# Add dod2k to path
dod2k = Path().resolve().parent
sys.path.insert(0, str(dod2k))
print(dod2k)
from dod2k_utilities.ut_functions import load_compact_dataframe_from_csv
```
The function `load_compact_dataframe_from_csv` imports the different csv files and stitches them together to form a dataframe. See [`load_compact_dataframe_from_csv()`](../api/ut_functions.md) for details.

### Step 2: Load the data from csv
```python title='python3/Jupyter'
# Load the duplicate-free database
df = load_compact_dataframe_from_csv('dod2k')

# Check the shape
print(f"Database contains {len(df)} records")
print(f"Columns: {', '.join(df.columns)}")
```


## Alternatively: Load the database from the pickle
For faster loading and if you only need python access, use the pickle format.
Make sure you start in the repository root directory (`dod2k`). From here import
```python title='python3/Jupyter'
import pandas as pd

# Load the duplicate-free database
df = pd.read_pickle('/data/dod2k_dupfree_dupfree/dod2k_compact.pkl')

print(f"Database contains {len(df)} records")
```


## **Explore the dataframe and visualise the data**

### Step 3: Explore the dataframe column by column

```python title='python3/Jupyter'
import pandas as pd
import numpy as np
```

Under `dod2k/notebooks` you can find the notebook `df_info.ipynb`. This notebook goes through the dataframe column by column and shows you a quick summary of the entries. 

The key features of this notebook are:

```python title='python3/Jupyter' title="Input"
print(df.info())
```

```text title="Output"
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4516 entries, 0 to 4515
Data columns (total 19 columns):
 #   Column                                Non-Null Count  Dtype  
---  ------                                --------------  -----  
 0   archiveType                           4516 non-null   object 
 1   climateInterpretation_variable        4516 non-null   object 
 2   climateInterpretation_variableDetail  4516 non-null   object 
 3   dataSetName                           4516 non-null   object 
 4   datasetId                             4516 non-null   object 
 5   duplicateDetails                      4516 non-null   object 
 6   geo_meanElev                          4433 non-null   float32
 7   geo_meanLat                           4516 non-null   float32
 8   geo_meanLon                           4516 non-null   float32
 9   geo_siteName                          4516 non-null   object 
 10  originalDataURL                       4516 non-null   object 
 11  originalDatabase                      4516 non-null   object 
 12  paleoData_notes                       4516 non-null   object 
 13  paleoData_proxy                       4516 non-null   object 
 14  paleoData_sensorSpecies               4516 non-null   object 
 15  paleoData_units                       4516 non-null   object 
 16  paleoData_values                      4516 non-null   object 
 17  year                                  4516 non-null   object 
 18  yearUnits                             4516 non-null   object 
dtypes: float32(3), object(16)
memory usage: 617.6+ KB
None
```
The interactive notebook then goes through each column and shows the entries, for example

```python title='python3/Jupyter' title="Input"
# archiveType
key = 'archiveType'
print('%s: '%key)
print(np.unique(df[key]))
print(np.unique([str(type(dd)) for dd in df[key]]))
```

```text title="Output"
archiveType: 
['bivalve' 'borehole' 'coral' 'documents' 'glacier ice' 'ground ice'
 'hybrid' 'lake sediment' 'marine sediment' 'mollusk shells'
 'sclerosponge' 'speleothem' 'terrestrial sediment' 'tree']
["<class 'str'>"]
```

```python title='python3/Jupyter' title="Input"
# paleoData_proxy
key = 'paleoData_proxy'
print('%s: '%key)
print(np.unique([kk for kk in df[key]]))
print(np.unique([str(type(dd)) for dd in df[key]]))
```

```text title="Output"
paleoData_proxy: 
['BSi' 'Documentary' 'MXD' 'Mg/Ca' 'Sr/Ca' 'TEX86' 'TRW' 'alkenone'
 'borehole' 'calcification' 'calcification rate' 'chironomid'
 'chrysophyte' 'd13C' 'd18O' 'd2H' 'diatom' 'dynocist MAT' 'foram Mg/Ca'
 'foram d18O' 'foraminifera' 'growth rate' 'historic' 'hybrid' 'melt'
 'midge' 'pollen' 'reflectance' 'sed accumulation' 'varve property'
 'varve thickness']
["<class 'str'>"]
```


For further guidance see the [interactive notebook](../notebooks/df_info.ipynb).


### Step 4: Visualise the dataframe

Under `dod2k/notebooks` you can find the notebook `df_plot_dod2k.ipynb`. This notebook visualises the dataframe and produces summary figures of the database. It also reproduces the manuscript figures. 

Import the python libraries
```python title='python3/Jupyter' 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature 
from matplotlib.gridspec import GridSpec as GS
from copy import deepcopy as dc

from dod2k_utilities import ut_functions as utf # contains utility functions
from dod2k_utilities import ut_plot as uplt # contains plotting functions
```

After loading the dataframe, start off by counting the number of records in each archive type:

```python title='python3/Jupyter'
# count archive types
archive_count = {}
for ii, at in enumerate(set(df['archiveType'])):
    archive_count[at] = df.loc[df['archiveType']==at, 'archiveType'].count()
```

Now count the number of records for each proxy type, depending on the archive type:
```python title='python3/Jupyter'
archive_proxy_count = {}
archive_proxy_ticks = []
for ii, at in enumerate(set(df['archiveType'])):
    proxy_types   = df['paleoData_proxy'][df['archiveType']==at].unique()
    for pt in proxy_types:
        cc = df['paleoData_proxy'][(df['paleoData_proxy']==pt)&(df['archiveType']==at)].count()
        archive_proxy_count['%s: %s'%(at, pt)] = cc
        archive_proxy_ticks += [at+': '+pt]
```

For each archive type, specify colours for each archive, but also distinguish between major archives (which have the most records) and minor archives (rare ones, only including less than ten records):
```python title='python3/Jupyter'
archive_colour = {'other': cols[-1]}
other_archives = []
major_archives = []

sort = np.argsort([cc for cc in archive_count.values()])
archives_sorted = np.array([cc for cc in archive_count.keys()])[sort][::-1]

for ii, at in enumerate(archives_sorted):
    print(ii, at, archive_count[at])
    if archive_count[at]>10:
        major_archives     +=[at]
        archive_colour[at] = cols[ii]
    else:
        other_archives     +=[at]
        archive_colour[at] = cols[-1]
```
Now plot a bar chart of the major archives using [`plot_count_proxy_by_archive_short()`](../api/ut_plot.md)

```python title='python3/Jupyter'
uplt.plot_count_proxy_by_archive_short(df, archive_proxy_count, archive_proxy_ticks, archive_colour) 
```

<figure markdown="span">
  ![Archive barchart](../assets/images/dod2k_barchart.png){ width="800" }
  <figcaption>Figure 1: Number of records for each proxy type, by archive.</figcaption>
</figure>

Next plot a spatial plot of all the proxy records:

```python title='python3/Jupyter'
#%% plot the spatial distribution of all records
proxy_lats = df['geo_meanLat'].values
proxy_lons = df['geo_meanLon'].values

# plots the map
fig = plt.figure(figsize=(15, 12), dpi=350)
grid = GS(1, 3)

ax = plt.subplot(grid[:, :], projection=ccrs.Robinson()) # create axis with Robinson projection of globe

ax.add_feature(cfeature.LAND, alpha=0.5) # adds land features
ax.add_feature(cfeature.OCEAN, alpha=0.6, facecolor='#C5DEEA') # adds ocean features
ax.coastlines() # adds coastline features

ax.set_global()

# loop through the data to generate a scatter plot of each data record:
# 1st loop: go through archive types individually (determines marker type)
# 2nd loop: through paleo proxy types attributed to the specific archive, which is colour coded


mt = 'ov^s<>pP*XDdh'*10 # generates string of marker types

archive_types = major_archives+other_archives
# archive_types = [aa for aa in archive_types if aa!='other']


ijk=0
for jj, at in enumerate(archive_types):
    arch_mask = df['archiveType']==at
    arch_proxy_types = np.unique(df['paleoData_proxy'][arch_mask])
    for ii, pt in enumerate(arch_proxy_types):
        pt_mask = df['paleoData_proxy']==pt
        at_mask = df['archiveType']==at
        label = at+': '+pt+' ($n=%d$)'% df['paleoData_proxy'][(df['paleoData_proxy']==pt)&(df['archiveType']==at)].count()
        marker = mt[ii] if at in major_archives else mt[ijk]
        plt.scatter(proxy_lons[pt_mask&at_mask], proxy_lats[pt_mask&at_mask], 
                    transform=ccrs.PlateCarree(), zorder=999,
                    marker=marker, color=archive_colour[at], 
                    label=label,#.replace('marine sediment:', 'marine sediment:\n'), 
                    lw=.3, ec='k', s=200)
        if at not in major_archives: ijk+=1
            
plt.legend(bbox_to_anchor=(-0.01,-0.01), loc='upper left', ncol=3, fontsize=13.5, framealpha=0)
grid.tight_layout(fig)

utf.save_fig(fig, f'{df.name}_spatial_all', dir=df.name)
```

Which creates this plot

<figure markdown="span">
  ![Archive map](../assets/images/overview_map.png){ width="800" }
  <figcaption>Figure 2: Spatial distribution of records by archive and proxy type.</figcaption>
</figure>


For further guidance see the [interactive notebook](../notebooks/df_plot_dod2k.ipynb).


## **Filter the dataframe**

Under `dod2k/notebooks` you can find the notebook `df_filter.ipynb`. This notebook let's you filter the dataframe for a specific metadata type, e.g. for moisture sensitive records, or for tree/TRW type records, etc.

This notebook then saves the filtered dataframe as a compact dataframe under `dod2k/data`, from which it can be loaded by other notebooks (e.g. `df_plot_dod2k.ipynb`).


Start by loading the dataframe, then filter using e.g. 
```python title='python3/Jupyter'
# # filter for >>exclusively moisture<< sensitive records only (without t+m)
df_filter = df.loc[(df['climateInterpretation_variable']=='moisture')]
```

The resulting dataframe can then be saved and used as input for `df_info.ipynb` or `df_plot_dod2k.ipynb`, or you can add this line to another notebook if you prefer. 


For further guidance see the [interactive notebook](../notebooks/df_filter.ipynb).


