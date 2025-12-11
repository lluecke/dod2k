#!/usr/bin/env python
# coding: utf-8

# # Duplicate detection - step 1: find the potential duplicates

# This notebook runs the first part of the duplicate detection algorithm on a dataframe with the following columns:
# 
# - `archiveType`       (used for duplicate detection algorithm)
# - `dataSetName`
# - `datasetId`
# - `geo_meanElev`      (used for duplicate detection algorithm)
# - `geo_meanLat`       (used for duplicate detection algorithm)
# - `geo_meanLon`       (used for duplicate detection algorithm)
# - `geo_siteName`      (used for duplicate detection algorithm)
# - `interpretation_direction`
# - `interpretation_seasonality`
# - `interpretation_variable`
# - `interpretation_variableDetails`
# - `originalDataURL`
# - `originalDatabase`
# - `paleoData_notes`
# - `paleoData_proxy`   (used for duplicate detection algorithm)
# - `paleoData_units`
# - `paleoData_values`  (used for duplicate detection algorithm, test for correlation, RMSE, correlation of 1st difference, RMSE of 1st difference)
# - `paleoData_variableName`
# - `year`              (used for duplicate detection algorithm)
# - `yearUnits`
# 
# The key function for duplicate detection is `find_duplicates` in `f_duplicate_search.py`
# 
# The output is saved as csvs in the directory `data/DATABASENAME/dup_detection`, which are used again for step 2 (`dup_decisions.py`):
# - `pot_dup_correlations_DATABASENAME.csv`
#    - matrix of correlations between each pair      
# - `pot_dup_distances_km_DATABASENAME.csv`
#    - matrix of distances between each pair 
# - `pot_dup_IDs_DATABASENAME.csv`
#    - saves the IDs of each pair
# - `pot_dup_indices_DATABASENAME.csv`
#    - saves the dataframe indices of each pair
# 
# Summary figures of the potential duplicate pairs are created and the plots are saved in the same directory, following:
# duplicatenumber_ID1_ID2_index1_index2.jpg
# 
# Updates:
# - 06/11/2025 by LL: Tidied up and updated for DoD2k v2.0
# - 27/11/2024 by LL: Fixed a bug in find_duplicates (in f_duplicate_search) and relaxed site criteria.
# 
# 27/9/2024 created by LL
# 
# Author: Lucie J. Luecke

# ## Set up working environment

# Make sure the repo_root is added correctly, it should be: your_root_dir/dod2k
# This should be the working directory throughout this notebook (and all other notebooks).

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import sys
import os
from pathlib import Path

# Add parent directory to path (works from any notebook in notebooks/)
# the repo_root should be the parent directory of the notebooks folder
current_dir = Path().resolve()
# Determine repo root
if current_dir.name == 'dod2k': repo_root = current_dir
elif current_dir.parent.name == 'dod2k': repo_root = current_dir.parent
else: raise Exception('Please review the repo root structure (see first cell).')

# Update cwd and path only if needed
if os.getcwd() != str(repo_root):
    os.chdir(repo_root)
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

print(f"Repo root: {repo_root}")
if str(os.getcwd())==str(repo_root):
    print(f"Working directory matches repo root. ")


# In[2]:


import pandas as pd
import numpy as np

from dod2k_utilities import ut_functions as utf # contains utility functions
from dod2k_utilities import ut_duplicate_search as dup # contains utility functions


# ## Load dataset

# Define the dataset which needs to be screened for duplicates. Input files for the duplicate detection mechanism need to be compact dataframes (`pandas` dataframes with standardised columns and entry formatting). 
# 
# The function `load_compact_dataframe_from_csv` loads the dataframe from a `csv` file from `data\DB\`, with `DB` the name of the database. The database name (`db_name`) can be 
# - `pages2k`
# - `ch2k`
# - `iso2k`
# - `sisal`
# - `fe23`
# 
# for the individual databases, or 
# 
# - `all_merged`
# 
# to load the merged database of all individual databases, or can be any user defined compact dataframe.

# In[3]:


# load dataframe
db_name='all_merged' 
# db_name='ch2k' 
df = utf.load_compact_dataframe_from_csv(db_name)

print(df.info())
df.name = db_name


# # Duplicate Detection

# ### Find duplicates

# Now run the first part of the duplicate detection algorithm, which goes through each candidate pair and evaluates the pairs for the following criteria:
# 
# - **metadata criteria**:
#   - archive types (`archiveType`) must be identical
#   - proxy types (`paleoData_proxy`) must be identical
# - **geographical criteria**:
#   - elevation (`geo_meanElev`) similar, within defined tolerance (use kwarg `elevation_tolerance`, defaults to 0)
#   - latitude and longtitude (`geo_meanLat` and `geo_meanLon`) similar, within defined tolerance in km (use kwarg `dist_tolerance_km`, defaults to 8 km)
# - **overlap criterion**:
#   - time must overlap for at least $n$ points (use kwarg `n_points_thresh` to modify, defaults to $n=10$) unless at least one of the record is shorter than `n_points_thresh` 
# - **site criterion**:
#   - there must be some overlap in the site name (`geo_siteName`)
# - **correlation criteria**:
#   - correlation between the overlapping period must be greater than defined threshold (use `corr_thresh` to modify, defaults to 0.9) or correlation of first difference must be greater than defined threshold (use `corr_diff_thresh` to modify, defaults to 0.9)
#   - RMSE of overlapping period must be smaller than defined threshold (use `rmse_thresh` to modify, defaults to 0.1) or RMSE of first difference must be smaller than defined threshold (use `rmse_diff_thresh` to modify, defaults to 0.1)
# - **URL criterion**:
#   - URLs (`originalDataURL`) must be identical if both records originate from the same database (`originalDatabase` must be identical)
# 
# 
# -----
# 
#  **A potential duplicate candidate pair is flagged, if all of these criteria are satisfied OR the correlation between the candidates is particularly high (>0.98), while there is sufficient overlap (as defined by the overlap criterion).**
# 
#  ----
# 
# The output for a database named `DB` is saved under `data/DB/dup_detection/dup_detection_candidates_DB.csv`.
#  

# In[4]:


## run the find duplicate algorithm
dup.find_duplicates_optimized(df, n_points_thresh=10)


# ## Plot duplicate candidate pairs

# *OPTIONAL*: plot the duplicate candidate pairs, which were flagged by the duplicate detection algorithm. 
# The function `plot_duplicates` loads the flagged candidate pairs for a database named `DB` from csv (`data/DB/dup_detection/dup_detection_candidates_DB.csv`) and produces summary figures of the potential duplicates, which are saved in the directory `figs/DB/dup_detection/`.
# 
# **Note that the same summary figures are being used for the duplicate decision process (`dup_decisions.ipynb`).**

# In[5]:


dup.plot_duplicates(df, save_figures=False)


# In[6]:


fn = utf.find(f'dup_detection_candidates_{df.name}.csv',  f'data/{df.name}/dup_detection')


# In[7]:


if fn != []:
    print('----------------------------------------------------')
    print('Sucessfully finished the duplicate detection process!'.upper())
    print('----------------------------------------------------')
    print('Saved the detection output file in:')
    print()
    print('%s.'%', '.join(fn))
    print()
    print('You are now able to proceed to the next notebook: dup_decision.ipynb')
else:
    print('Final output file is missing.')
    print()
    print('Please re-run the notebook to complete duplicate detection process.')

