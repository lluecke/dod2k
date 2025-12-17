# **Run the duplicate detection workflow to generate a duplicate free dataframe**

This workflow runs a duplicate detection, decision and removal algorithm to generate a duplicate free dataframe. 

## Required columns
The input dataframe must have the following columns:
<div class="grid" markdown style="grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.5rem;">
- `archiveType`       (used for duplicate detection algorithm)
- `dataSetName`
- `datasetId`
- `geo_meanElev`      (used for duplicate detection algorithm)
- `geo_meanLat`       (used for duplicate detection algorithm)
- `geo_meanLon`       (used for duplicate detection algorithm)
- `geo_siteName`      (used for duplicate detection algorithm)
- `interpretation_direction`
- `interpretation_seasonality`
- `interpretation_variable`
- `interpretation_variableDetails`
- `originalDataURL`
- `originalDatabase`
- `paleoData_notes`
- `paleoData_proxy`   (used for duplicate detection algorithm)
- `paleoData_units`
- `paleoData_values`  (used for duplicate detection algorithm, test for correlation, RMSE, correlation of 1st difference, RMSE of 1st difference)
- `paleoData_variableName`
- `year`              (used for duplicate detection algorithm)
- `yearUnits`
</div>

!!! Info "Output Location"

    All outputs are saved as `csv` in the directory `data/DATABASENAME/dup_detection`.

----------------
## **Step 1: Duplicate detection (`dup_detection.ipynb`)**

**Notebook:** [`dup_detection.ipynb`](../notebooks/dup_detection.ipynb)

This interactive notebook (`dup_detection.ipynb`) runs a duplicate detection algorithm for a specific database. 

### 1.1 Set up working environment
Make sure the repo_root is added correctly: `your_root_dir/dod2k`
This should be the working directory throughout this notebook (and all other notebooks).
The following libraries are required to run this notebook

```python title="python3/Jupyter"
import pandas as pd
import numpy as np

from dod2k_utilities import ut_functions as utf # contains utility functions
from dod2k_utilities import ut_duplicate_search as dup # contains utility functions
```

### 1.2 Load the compact dataframe
Define the dataset which needs to be screened for duplicates. Input files for the duplicate detection mechanism need to be compact dataframes (`pandas` dataframes with standardised columns and entry formatting). 

The function `load_compact_dataframe_from_csv` loads the dataframe from a `csv` file from `data\DB\`, with `DB` the name of the database. The database name (`db_name`) can be 
- `pages2k`
- `ch2k`
- `iso2k`
- `sisal`
- `fe23`

for the individual databases, or 

- `all_merged`

to load the merged database of all individual databases, or can be any user defined compact dataframe.

Load the dataframe using
```python title="python3/Jupyter"
db_name='all_merged' 
df = utf.load_compact_dataframe_from_csv(db_name)
```




### 1.3 Run the duplicate detection algorithm

Now run the first part of the duplicate detection algorithm, which goes through each candidate pair and evaluates the pairs according to a defined set of criteria.


```python title="python3/Jupyter"
dup.find_duplicates_optimized(df, n_points_thresh=10)
```

**Output:** `data/DB/dup_detection/dup_detection_candidates_DB.csv`


!!! abstract "Detection Criteria"
    - **metadata criteria**:
      - archive types (`archiveType`) must be identical
      - proxy types (`paleoData_proxy`) must be identical
    - **geographical criteria**:
      - elevation (`geo_meanElev`) similar, within defined tolerance (use kwarg `elevation_tolerance`, defaults to 0)
      - latitude and longtitude (`geo_meanLat` and `geo_meanLon`) similar, within defined tolerance in km (use kwarg `dist_tolerance_km`, defaults to 8 km)
    - **overlap criterion**:
      - time must overlap for at least $n$ points (use kwarg `n_points_thresh` to modify, defaults to $n=10$) unless at least one of the record is shorter than `n_points_thresh` 
    - **site criterion**:
      - there must be some overlap in the site name (`geo_siteName`)
    - **correlation criteria**:
      - correlation between the overlapping period must be greater than defined threshold (use `corr_thresh` to modify, defaults to 0.9) or correlation of first difference must be greater than defined threshold (use `corr_diff_thresh` to modify, defaults to 0.9)
      - RMSE of overlapping period must be smaller than defined threshold (use `rmse_thresh` to modify, defaults to 0.1) or RMSE of first difference must be smaller than defined threshold (use `rmse_diff_thresh` to modify, defaults to 0.1)
    - **URL criterion**:
      - URLs (`originalDataURL`) must be identical if both records originate from the same database (`originalDatabase` must be identical)

!!! warning "Flagging Logic"

    **A potential duplicate candidate pair is flagged, if all of these criteria are satisfied OR the correlation between the candidates is particularly high (>0.98), while there is sufficient overlap (as defined by the overlap criterion).**


???+ tip "Tip for large databases"

    The duplicate detection algorithm can take a while to run, especially for large databases (such as the merged database with over 5000 records). 
    Instead of running this notebook interactively, it might therefore be better to execute it as a python script via the command line.
    
    In order to do this, run
    
    ```bash title='bash'
    cd ~/dod2k_v2.0/dod2k
    mkdir -p scripts
    jupyter nbconvert --to python notebooks/dup_detection.ipynb --stdout | \
      sed 's/^get_ipython()/# get_ipython()/' | \
      sed 's/^\([[:space:]]*\)%/\1# %/' > scripts/dup_detection.py
    ```
    
    This generates a script `dup_detection.py` from the command line. Make sure you have modified this file to load the correct database before executing. Then run 
    ```bash title='bash'
    python scripts/dup_detection.py
    ```
<!-- The output for a database named `DB` is saved under `data/DB/dup_detection/dup_detection_candidates_DB.csv`.

```python title="python3/Jupyter"
dup.find_duplicates_optimized(df, n_points_thresh=10)
``` -->


### 1.4 Optional: display the duplicate candidate summary figures

<!-- *OPTIONAL*: plot the duplicate candidate pairs, which were flagged by the duplicate detection algorithm. 
The function `plot_duplicates` loads the flagged candidate pairs for a database named `DB` from csv (`data/DB/dup_detection/dup_detection_candidates_DB.csv`) and produces summary figures of the potential duplicates, which are saved in the directory `figs/DB/dup_detection/`.

**Note that the same summary figures are being used for the duplicate decision process (`dup_decisions.ipynb`).**

Run
```python title="python3/Jupyter"
dup.plot_duplicates(df, save_figures=True)
```
 -->

*Optional:* Plot flagged candidate pairs. Figures are saved to `figs/DB/dup_detection/`.

```python title="python3/Jupyter"
dup.plot_duplicates(df, save_figures=True)
```

!!! note
    These same figures are used in the duplicate decision process.


## **Step 2: Duplicate decisions (`dup_decision.ipynb`)**

**Notebook:** [`dup_decision.ipynb`](../notebooks/dup_decision.ipynb)

This interactive notebook (`dup_decision.ipynb`) runs a duplicate decision algorithm for a specific database, following the identification of the potential duplicate candidate pairs. The algorithm walks the operator through each of the detected duplicate candidate pairs from `dup_detection.ipynb` and runs a decision process to decide whether to keep or reject the identified records.  


### 2.1 Initialisation

To set up the working directory and load the compact dataframe, please follow the instructions detailed in  steps 1.1 (set up working directory) and 1.2 (load compact dataframe). 

In addition, the operator is asked to provide their credentials along with the decision process. Please fill in your details:

```python title='python3/Jupyter'

initials = 'FN'
fullname = 'Full Name'
email    = 'name@email.ac.uk'
operator_details = [initials, fullname, email]
```

!!! info "Why Credentials?"
    - Initials label intermediate output files
    - Name and email ensure transparency and traceability

<!-- The initials are used to label the intermediate output files for the next steps of the workflow, and the name and email address will be saved in the final duplicate free dataframe to ensure transparency and traceability. -->

### 2.2 Hierarchy for duplicate removal for identical duplicates

For automated decisions, which apply to *identical* duplicates, we have defined a hierarchy of databases, which decides which record should be kept.

The hierarchy is assigned to the original databases, from 1 the highest value (should always be kept) to the lowest value n (the number of original databases). The hierarchy is added to the dataframe as an additional column (`Hierarchy`) for the decision process. Note that this parameter is not migrated to the final duplicate free database. 

```python title='python3/Jupyter'
# implement hierarchy for automated decisions for identical records

df = dup.define_hierarchy(df, hierarchy='default')
```

The default hierarchy is given as

PAGES2k v2.2.0 < FE23 (Breitenmoser et al. (2014)) < CoralHydro2k v1.0.1 < Iso2k v1.1.2 < SISAL v3.

The hierarchy can be changed by providing a dictionary to the `hierarchy` kwarg:
```
hierarchy={'pages2k': pages2k_value, 'fe23': fe23_value, 'iso2k': iso2k_value, 'ch2k': ch2k_value, 'sisal': sisal_value}
```

!!! note
    This hierarchy is not saved in the final duplicate-free database.

In order to reduce the operator workload, you also have the option to implement an automatic choice for specific database combinations. Please also specify a reason when doing so!

This is meant to be for any records which do not satisfy the hierarchy criterion, i.e. records with different data but identical metadata, such as updated records. 

If you do not wish to do this, delete `automate_db_choice` from kwargs or set to `False` (default).

For example we have set 

```python title='python3/Jupyter'
automate_db_choice = {'preferred_db': 'FE23 (Breitenmoser et al. (2014))', 
                      'rejected_db': 'PAGES 2k v2.2.0', 
                      'reason': 'conservative replication requirement'}
```




### 2.3 Duplicate decision process

Run the decision algorithm:
```python title='python3/Jupyter'
dup.duplicate_decisions_multiple(df, operator_details=operator_details, choose_recollection=True, 
                                 remove_identicals=True, backup=True, comment=True, automate_db_choice=automate_db_choice)
```
**Decision options for each pair:**

- keep both records
- keep just one record
- delete both records
- create composite of both records.


!!! tip "Automated Decisions"
    - Recollections/updates: automatically selected
    - Identical duplicates: highest hierarchy record kept automatically
    - Automate db choice: as described previously
    
<!-- Recollections and updates of duplicates are automatically selected (`choose_recollection=True`), as well as identical duplicates following the hierarchy defined above (`remove_identicals=True`).  -->


<!-- The duplicate decision algorithm is then initiated via:

```python title='python3/Jupyter'
dup.duplicate_decisions(df, operator_details=operator_details, choose_recollection=True, 
                        remove_identicals=True)
```
This function goes one-by-one through the individual duplicate candidate pairs and scans the data and metadata. For identical duplicates (defined as very high correlation and identical metadata, or perfect correlation), the algorithm makes an automated decision, in which the database with the highest hierarchical position is being chosen to keep (decision `KEEP`), while the other record will be removed (decision `REMOVE`). For other duplciates, the operator needs to provide manual input, based on a summary figure (see Figure 1).  -->

<!-- <figure markdown="span">
  ![Duplicate summary figure](../assets/images/dup_detection/duplicate_candidate_pair.jpg){ width="800" }
  <figcaption>Figure 1: Summary figure of a potential duplicate candidate pair, for which the operator is asked to make a decision.</figcaption>
</figure>

The operator is first presented with the option to leave a note on the decision process: 

```
**Decision required for this duplicate pair (see figure above).**
Before inputting your decision. Would you like to leave a comment on your decision process?
```

This adds transparency on the decision process and justifies the operators choices. 

Next, the operator is asked to make the following decision, for example:

```
Keep record 1 (pages2k_50, black) [1], record 2 (FE23_northamerica_canada_cana091, blue) [2], keep both [b], keep none [n] or create a composite of both records [c]?  [Type 1/2/b/n/c]
```

For candidate pairs which have a very high correlation and.or for which the metadata is identical, we have implemented an automated choice, in which the data record from the most recently published database is automatically chosen. This means less user inout is required and is especially helpful for very large databases with hundreds of duplicates. 


!!! Note

    The operator has the option to restart the decision process from a backup file in the directory `data/DB/dup_detection`. This can be especially useful should the connection be interrupted during the process. If such a file exists, the operator would be asked wether they want to use this file to restart the decision process. The decision process will then restart where the backup file cut out. This works for multiple interruptions.

??? question "Can I reverse a decision?"

    There is currently no option to reverse a decision while running the duplicate decisions. However should the operator want to revise a previous decision they have two options: 

    1. If this concerns the most recent decision, the operator should interrupt the decision process and remove the last line from the backup file (`data/DB/dup_detection/dup_decisions_DB_INITIALS_BACKUP.csv`). They could then restart the decision process and the process will restart from the last line in the backup file, so the operator has the opportunity to revise their last decision.
    
    2. They could interrupt the decision process and directly edit the backup file. The backup file (`data/DB/dup_detection/dup_decisions_DB_INITIALS_BACKUP.csv`) is a spreasheet which can be edited directly and the decisions can be revised under 'Decision 1' and 'Decision 2' (columns X and Y in Microsoft Excel). Note that the operator should keep to the standard terminology of input to avoid problems further downstream, i.e. use `KEEP`, `REMOVE` or `COMPOSITE` only. 

    3. The operator could complete the decision process and manually edit the final decision output file. Please make sure to use correct terminology as in option #2.
    

The final output is saved in `data/DB/dup_detection/dup_decisions_dod2k_dupfree_INITIALS_DATE.csv`

Summary figures are saved in the directory `figs/dup_detection/DB`, and are also linked in the output csv file. -->

**Example prompts:**
<figure markdown="span">
  ![Duplicate summary figure](../assets/images/dup_detection/duplicate_summary.png){ width="800" }
  <figcaption>Figure 1: Summary figure of a potential duplicate candidate pair, for which the operator is asked to make a decision.</figcaption>
</figure>

```
**Decision required for this duplicate pair (see figure above).**
Before inputting your decision.
Would you like to leave a comment on your decision process?
**COMMENT** Please type your comment here and/or press enter.
```
```

 **DECISION** Keep record 1 (pages2k_50, blue circles) [1],
record 2 (FE23_northamerica_canada_cana091, red crosses) [2],
keep both [b], keep none [n] or create a composite of both records [c]?
Note: only overlapping timesteps are being composited. [Type 1/2/b/n/c]:
```


**Output:** `data/DB/dup_detection/dup_decisions_dod2k_dupfree_INITIALS_DATE.csv`

**Figures:** `figs/dup_detection/DB/` (linked in output CSV)

!!! note "Backup & Resume"
    The process creates backup files in `data/DB/dup_detection/`. If interrupted, you can resume from the backup.

??? question "Can I Reverse a Decision?"

    There is currently no option to reverse a decision while running the duplicate decisions. 
    However should the operator want to revise a previous decision they have two options: 

    1. **Most recent decision:** Interrupt the process, remove the last line from the backup file (`data/DB/dup_detection/dup_decisions_DB_INITIALS_BACKUP.csv`), then restart.
    
    2. **Any decision:** Interrupt and directly edit the backup file columns 'Decision 1' and 'Decision 2'. Use only: `KEEP`, `REMOVE`, or `COMPOSITE`.

    3. **After completion:** Manually edit the final output file with correct terminology.


!!! info "Handling of multiple duplicates"
    The decision process is currently not optimised for handling of multiple duplicates (i.e. records which have more than one potential duplicate candidate), going through the duplicates on a pair-by-pair basis. However, `dup.duplicate_decisions_multiple` includes improved handling of multiple duplicates. For any records which are associated with multiple duplicates, all the other duplicate candidates are shown alongside the summary figure for the duplicate candidate pair. Any previous decisions, when available, are shown besides the `datasetId`, `archiveType`, `paleoData_proxy` etc.:
    
    ```
    ***ATTENTION*** THIS RECORD IS ASSOCIATED WITH MULTIPLE DUPLICATES! 
    PLEASE PAY SPECIAL ATTENTION WHEN MAKING DECISIONS FOR THIS RECORD!
    The potential duplicates also associated with this record are:
     Dataset ID          : iso2k_786
         - URL                 : https://www.ncdc.noaa.gov/paleo/study/1856
    ```
    
    <figure markdown="span">
      ![Multiple summary figure](../assets/images/dup_detection/duplicate_multiples.png){ width="800" }
      <figcaption>Figure 2: Summary figure for multiple duplicates.</figcaption>
    </figure>
    
    The operator can then make an informed decision for each candidate pair.

    
---


## **Step 3: Duplicate removal (`dup_removal.ipynb`)**

**Notebook:** [`dup_removal.ipynb`](../notebooks/dup_removal.ipynb)

This notebook removes duplicates based on decisions from Step 2.

<!-- This interactive notebook (`dup_removal.ipynb`) removes the duplicates flagged in `dup_detection.ipynb`, following the decisions made in `dup_decision.ipynb`. The decisions include
- removal of redundant duplicates
- creation of composites

 To view the interactive notebook see [dup_removal.ipynb](../notebooks/dup_removal.ipynb).  -->


### 3.1 Initialisation

To set up the working directory and load the compact dataframe, please follow the instructions detailed in steps 1.1 (set up working directory), 1.2 (load compact dataframe) and 2.1 (provide operator credentials).

In addition, `datasetId` needs to be set as dataframe index to reliably identify the duplicates later on:
```python title='python3/Jupyter'
df.set_index('datasetId', inplace = True)
df['datasetId']=df.index
```

### 3.2 Load duplicate decisions from csv
In order to load the duplicate decisions from csv, the operator *initials* and the *date* need to be specified, to match the desired decision output file.

Accordingly, the decision output file is loaded from `data/DBNAME/dup_detection/dup_decisions_DBNAME_INITIALS_DATE.csv`:

```python title='python3/Jupyter'
filename      = f'data/{df.name}/dup_detection/dup_decisions_{df.name}_{initials}_{date}'
data, header  = dup.read_csv(filename, header=True)
df_decisions  = pd.read_csv(filename+'.csv', header=5)
```

`dup.read_csv` provides the `header`, which details the operator's details as provided in the decision file, along with any comments on the general decision process. Later in the notebook, `header` is written into a metadata file which should be provided alongside the duplicate free dataset. `df_decisions` is a `pandas` dataframe which is populated with the decision data, record by record, and will be used to implement the decisions to create a duplicate free dataset.

### 3.3 Implement duplicate decisions

From `df_decisions` we extract a dictionary which includes all decisions for each individual record:

```python title='python3/Jupyter'
decisions = {}
for ind in df_decisions.index:
    id1, id2   = df_decisions.loc[ind, ['datasetId 1', 'datasetId 2']]
    dec1, dec2 = df_decisions.loc[ind, ['Decision 1', 'Decision 2']]
    for id, dec in zip([id1, id2], [dec1, dec2]):
        if id not in decisions: decisions[id] = []
        decisions[id]+=[dec]
```
This dictionary can be used to identify and track multiple decisions, as well as to review the choices made.


We also extract details of each decisions, which will later be used to populate the field `duplicateDetails` in the final dataframe (the output of this notebook):


```python title='python3/Jupyter'
dup_details = dup.provide_dup_details(df_decisions, header)
```


!!! Note

    Note that any one record can appear more than once and have multiple decisions associated with it (e.g. `'REMOVE'`, `'KEEP'` or `'COMPOSITE'`).
    
    In order to remove the duplicates we therefore implement the following steps:
    
    1. **Remove** all records from the dataframe which are associated with the decision `'REMOVE'` or `COMPOSITE` -> `df_cleaned`
    2. **Create composites** of the `COMPOSITE` records -> `df_composite`
    3. **Check** for records which have multiple decisions associated. These are potentially remaining duplicates.

#### 3.3.1. Records to be removed
First simply remove all the records to which the decision `REMOVE` or `COMPOSITE` applies to and store in `df_cleaned`, while all `'REMOVE'` or `'COMPOSITE'` type records are stored in `df_duplica_rmv` (for later inspection).

```python title='python3/Jupyter'
# load the records TO BE REMOVED
remove_IDs  = list(df_decisions['datasetId 1'][np.isin(df_decisions['Decision 1'],['REMOVE', 'COMPOSITE'])])
remove_IDs += list(df_decisions['datasetId 2'][np.isin(df_decisions['Decision 2'],['REMOVE', 'COMPOSITE'])])
remove_IDs  = np.unique(remove_IDs)

df_duplica     =  df.loc[remove_IDs, 'datasetId'] # df containing only records which were removed
df_cleaned =  df.drop(remove_IDs) # df freed from 'REMOVE' type duplicates

print(f'Removed {len(df_duplica)} REMOVE or COMPOSITE type records.')
print(f'REMOVE type duplicate free dataset contains {len(df_cleaned)} records.')
print('Removed the following IDs:', remove_IDs)
```

`df_cleaned` then contains all data apart from records which are marked as `REMOVE` or `COMPOSITE`. Thus, it only keeps the records which either were never marked as duplicates or where the operator had decided to keep a duplicate. 


Note that the `duplicateDetails` need to be added to `df_cleaned` via


```python title='python3/Jupyter'
df_cleaned['duplicateDetails']='N/A'
for ID in dup_details:
    if ID in df_cleaned.index: 
        if df_cleaned.at[ID, 'duplicateDetails']=='N/A': 
            df_cleaned.at[ID, 'duplicateDetails']=dup_details[ID]
        else: df_cleaned.at[ID, 'duplicateDetails']+=dup_details[ID]
```

#### 3.3.2. Records to be composited
Now identify all the records to which the decision `'COMPOSITE'` applies to, create composites and store in `df_composite`. 
For differences in the numerical metadata we use the average (e.g. `geo_meanLat`, `geo_meanLon`, ...), while for string types we merge the strings to form a composite. The `datasetId` is created from both original values to `'f{df.name}_composite_z_{ID_1}_{ID_2}'`, with `ID_1` and `ID_2` the original `datasetId` for each record. The data is being composited by averaging the z-scores of the original data. 

```python title='python3/Jupyter'
# add the column 'duplicateDetails' to df, in case it does not exist
if 'duplicateDetails' not in df.columns: df['duplicateDetails']='N/A'

# load the records to be composited
comp_ID_pairs = df_decisions[(df_decisions['Decision 1']=='COMPOSITE')&(df_decisions['Decision 2']=='COMPOSITE')]

# create new composite data and metadata from the pairs
# loop through the composite pairs and check metadata
df_composite = dup.join_composites_metadata(df, comp_ID_pairs, df_decisions, header)
```
The function `join_composites_metadata` also creates summary figures of the composites in order to supervise the composition process. 

#### 3.3.3. Check for multiple duplicate records with different decisions

In order to obtain the duplicate free dataframe we merge `df_cleaned` and `df_composite`:

```python title='python3/Jupyter'
tmp_df_dupfree = pd.concat([df_cleaned, df_composite])
tmp_df_dupfree.index = tmp_df_dupfree['datasetId']
tmp_decisions = decisions.copy()
```

This dataframe initiates a loop in which the records which are associated with multiple decisions are fed into another round of duplicate detection, decisions and removal. This is necessary to ensure that no duplicates remain in the merged dataframe because of combined decisions.

!!! info "Example"

    * `REMOVE`/`KEEP` and `COMPOSITE`:
      - duplicate pair `a` and `b` have had the decisions assigned: `a`-> `REMOVE`, `b` -> `KEEP`
      * duplicate pair `a` and `c` have had the decisions assigned: `a` -> `COMPOSITE`, `c` -> `COMPOSITE`.
      * In this case, `b` and `ac` (the composite record of `a` and `c`) would be <span style="color:red">**duplicates in the merged dataframe**</span>   
    * `REMOVE`/`KEEP` & `REMOVE`/`KEEP`
      * duplicate pair `a` and `b` have had the decisions assigned: `a`-> `REMOVE`, `b` -> `KEEP`
      * duplicate pair `a` and `c` have had the decisions assigned: `a` -> `REMOVE`, `c` -> `KEEP`.
      * In this case, `a` would be removed, but `b` and `c` will be kept and would be <span style="color:red">**duplicates in the merged dataframe**</span>. 
    * `COMPOSITE` x 2
      * duplicate pair `a` and `b` have had the decisions assigned: `a`-> `COMPOSITE`, `b` -> `COMPOSITE`
      * duplicate pair `a` and `c` have had the decisions assigned: `a` -> `COMPOSITE`, `c` -> `COMPOSITE`.
      * In this case, `ab` and `ac` would be <span style="color:red">**duplicates in the merged dataframe**</span>.

The loop iterates for a maximum of ten, but stops as soon as no duplicates are detected anymore in the dataframe subset. Note that this loop only checks among the records associated with more than one decision. In each iteration, the operator also has the opportunity to end the duplicate search. Note also that it is not advised to create multiple iterations of composites. 


```python title='python3/Jupyter'
# Simple composite tracking for debugging only
composite_log = []

for ii in range(10): 
    tmp_df_dupfree.set_index('datasetId', inplace = True)
    tmp_df_dupfree['datasetId']=tmp_df_dupfree.index
    
    print('-'*20)
    print(f'ITERATION # {ii}')
    
    multiple_dups = []
    for id in tmp_decisions.keys():
        if len(tmp_decisions[id]) > 1:
            if id not in multiple_dups:
                multiple_dups.append(id)
    
    if len(multiple_dups) > 0:
        # Check which of the multiple duplicate IDs are still in the dataframe
        multiple_dups_new = []
        current_ids = set(tmp_df_dupfree.index)  # Get all current IDs as a set
        
        for id in multiple_dups:
            if id in current_ids:  # Simple membership check
                multiple_dups_new.append(id)
        
        if len(multiple_dups_new) > 0:
            print(f'WARNING! Decisions associated with {len(multiple_dups_new)} multiple duplicates in the new dataframe.')
            print('Please review these records below and run through a further duplicate detection workflow until no more duplicates are found.')
        else:
            print('No more multiple duplicates found in current dataframe.')
            print('SUCCESS!!')
            break
    else:
        print('No more multiple duplicates.')
        print('SUCCESS!!')
        break
    
    # Now we create a small dataframe which needs to be checked for duplicates.
    df_check = tmp_df_dupfree.copy()[np.isin(tmp_df_dupfree['datasetId'], multiple_dups_new)]
    print('Check dataframe: ')
    df_check.name = 'tmp'
    df_check.index = range(len(df_check))
    print(df_check.info())
    # We then run a brief duplicate detection algorithm on the dataframe. Note that by default the composited data has the highest value in the hierarchy.
    pot_dup_IDs = dup.find_duplicates_optimized(df_check, n_points_thresh=10, return_data=True)
    if len(pot_dup_IDs)==0:
        print('SUCCESS!! NO MORE DUPLICATES DETECTED!!')
        break
    else:
        yn=''
        while yn not in ['y', 'n']:
            yn = input('Do you want to continue with the decision process for duplicates? [y/n]')
        if yn=='n': break
    
    df_check = dup.define_hierarchy(df_check)
    dup.duplicate_decisions_multiple(df_check, operator_details=operator_details, choose_recollection=True, 
                            remove_identicals=False, backup=False, comment=False)
    # implement the decisions
    tmp_df_decisions  = pd.read_csv(f'data/{df_check.name}/dup_detection/dup_decisions_{df_check.name}_{initials}_{date}'+'.csv', header=5)
    tmp_dup_details   = dup.provide_dup_details(tmp_df_decisions, header)

    
    # decisions
    tmp_decisions = {}
    for ind in tmp_df_decisions.index:
        id1, id2   = tmp_df_decisions.loc[ind, ['datasetId 1', 'datasetId 2']]
        dec1, dec2 = tmp_df_decisions.loc[ind, ['Decision 1', 'Decision 2']]
        for id, dec in zip([id1, id2], [dec1, dec2]):
            if id not in tmp_decisions: tmp_decisions[id] = []
            tmp_decisions[id]+=[dec]
    
    df_check.set_index('datasetId', inplace = True)
    df_check['datasetId']=df_check.index
    
    #drop all REMOVE or COMPOSITE types
    tmp_remove_IDs  = list(tmp_df_decisions['datasetId 1'][np.isin(tmp_df_decisions['Decision 1'],['REMOVE', 'COMPOSITE'])])
    tmp_remove_IDs += list(tmp_df_decisions['datasetId 2'][np.isin(tmp_df_decisions['Decision 2'],['REMOVE', 'COMPOSITE'])])
    tmp_remove_IDs = np.unique(tmp_remove_IDs)#[id for id in np.unique(tmp_remove_IDs) if id not in tmp_remove_IDs]
    tmp_df_cleaned = tmp_df_dupfree.drop(tmp_remove_IDs) # df freed from 'REMOVE' type duplicates
    
    # # composite the 
    tmp_comp_ID_pairs = tmp_df_decisions[(tmp_df_decisions['Decision 1']=='COMPOSITE')&(tmp_df_decisions['Decision 2']=='COMPOSITE')]
    
    if len(tmp_comp_ID_pairs) > 0:
        for _, pair in tmp_comp_ID_pairs.iterrows():
            id1, id2 = pair['datasetId 1'], pair['datasetId 2']
            # Log what was composited
            composite_log.append({
                'iteration': ii,
                'composited': [id1, id2],
                'new_id': f"{id1}_{id2}_composite"  # or however you generate it
            })
    # # create new composite data and metadata from the pairs
    # # loop through the composite pairs and check metadata
    tmp_df_composite = dup.join_composites_metadata(df_check, tmp_comp_ID_pairs, tmp_df_decisions, header)

    tmp_df_dupfree = pd.concat([tmp_df_cleaned, tmp_df_composite])
    print('--'*20)
    print('Finished iteration.')
    
    print('NEW DATAFRAME:')
    print(tmp_df_dupfree.info())

    print('--'*20)
    print('--'*20)
    if ii==19: print('STILL DUPLICATES PRESENT AFTER MULTIPLE ITERATIONS! REVISE DECISION PROCESS!!')

    print('--'*20)

print(f"Created {len(composite_log)} composites across all iterations")


```

As soon as no more duplicates are detected among the remaining candidates, the loop outputs:

```
No more multiple duplicates.
SUCCESS!!
```


### 3.4 Check entire dataframe for remaining duplicates

In order to check that all duplicates have definitely been removed from the dataframe, we run another round of duplicate detection, decisions and removal, using a similar workflow as in the previous step:

```python title='python3/Jupyter'
tmp_df_dupfree.set_index('datasetId', inplace = True)
tmp_df_dupfree['datasetId']=tmp_df_dupfree.index

# Now we create a  dataframe which needs to be checked for duplicates.
df_check = tmp_df_dupfree.copy()
df_check.name = 'tmp'
df_check.index = range(len(df_check))
# We then run a brief duplicate detection algorithm on the dataframe. Note that by default the composited data has the highest value in the hierarchy.
pot_dup_IDs = dup.find_duplicates_optimized(df_check, n_points_thresh=10, return_data=True)
if len(pot_dup_IDs)==0:
    print('SUCCESS!! NO MORE DUPLICATES DETECTED!!')
else:
    df_check = dup.define_hierarchy(df_check)
    dup.duplicate_decisions_multiple(df_check, operator_details=operator_details, choose_recollection=True, 
                            remove_identicals=False, backup=False)
    # implement the decisions
    tmp_df_decisions  = pd.read_csv(f'data/{df_check.name}/dup_detection/dup_decisions_{df_check.name}_{initials}_{date}'+'.csv', header=5)
    tmp_dup_details   = dup.provide_dup_details(tmp_df_decisions, header)
    
    
    # decisions
    tmp_decisions = {}
    for ind in tmp_df_decisions.index:
        id1, id2   = tmp_df_decisions.loc[ind, ['datasetId 1', 'datasetId 2']]
        dec1, dec2 = tmp_df_decisions.loc[ind, ['Decision 1', 'Decision 2']]
        for id, dec in zip([id1, id2], [dec1, dec2]):
            if id not in tmp_decisions: tmp_decisions[id] = []
            tmp_decisions[id]+=[dec]
    
    df_check.set_index('datasetId', inplace = True)
    df_check['datasetId']=df_check.index
    
    #drop all REMOVE or COMPOSITE types
    tmp_remove_IDs  = list(tmp_df_decisions['datasetId 1'][np.isin(tmp_df_decisions['Decision 1'],['REMOVE', 'COMPOSITE'])])
    tmp_remove_IDs += list(tmp_df_decisions['datasetId 2'][np.isin(tmp_df_decisions['Decision 2'],['REMOVE', 'COMPOSITE'])])
    tmp_remove_IDs = np.unique(tmp_remove_IDs)#[id for id in np.unique(tmp_remove_IDs) if id not in tmp_remove_IDs]
    tmp_df_cleaned = tmp_df_dupfree.drop(tmp_remove_IDs) # df freed from 'REMOVE' type duplicates
    
    # # composite the 
    tmp_comp_ID_pairs = tmp_df_decisions[(tmp_df_decisions['Decision 1']=='COMPOSITE')&(tmp_df_decisions['Decision 2']=='COMPOSITE')]
    
    # # create new composite data and metadata from the pairs
    # # loop through the composite pairs and check metadata
    tmp_df_composite = dup.join_composites_metadata(df_check, tmp_comp_ID_pairs, tmp_df_decisions, header)
    
    tmp_df_dupfree = pd.concat([tmp_df_cleaned, tmp_df_composite])
    
    print('Finished last round of duplicate removal.')
    print('Potentially run through this cell again to check for remaining duplicates.')
```

!!! warning

    This step runs an entire duplicate detection and thus can take a substantial amount of time, as previously. Alternatively, you can skip this step, output the dataframe and feed it back into `dup_detection.ipynb` and repeat the duplicate workflow.

    

### 3.5 Save duplicate free dataframe 

Once the operator is satisfied that no more duplicates remain, the final dataframe can be created


```python title='python3/Jupyter'
df_dupfree = tmp_df_dupfree
print(df_dupfree.info())
```

and saved via 

```python title='python3/Jupyter'
df_dupfree = df_dupfree[sorted(df_dupfree.columns)]
df_dupfree.name =f'{df.name}_{initials}_{date}_dupfree'
os.makedirs(f'data/{df_dupfree.name}/', exist_ok=True)


utf.write_compact_dataframe_to_csv(df_dupfree)
```

In order to provide the associated operator's information (such as details, date of creation and operator's comments), we also create the README file:

```python title='python3/Jupyter'
# write header with operator information as README txt file
file = open(f'data/{df_dupfree.name}/{df_dupfree.name}_dupfree_README.txt', 'w')
for line in header:
    file.write(line+'\n')
file.close()
```


!!! success "Workflow Complete"
    The duplicate detection workflow is now finished!

!!! info

    For more details on the interactive notebooks, see 
    1. [dup_detection.ipynb](../notebooks/dup_detection.ipynb)
    2. [dup_decision.ipynb](../notebooks/dup_decision.ipynb)
    3. [dup_removal.ipynb](../notebooks/dup_removal.ipynb)


