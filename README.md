# DoD 2k

A Python toolkit for integrating and standardizing global paleoclimate proxy databases to create a duplicate-free, quality-controled database of databases

## Documentation

For full documentation, please visit: [https://lluecke.github.io/dod2k/](https://lluecke.github.io/dod2k/)

## Quick Start

How to get started with the dod2k environment, functions, notebooks and products.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


### For database use (DoD2k)

0. Get the project: in a working directory,

    ```bash
    git clone https://github.com/lluecke/dod2k.git
    ```

1. Create and activate the python environment: in dod2k/, 

    ```bash
    conda env create -n dod2k-env -f dod2k-env.yml
    conda activate dod2k-env
    ```

2. Explore DoD2k: use the notebooks
       
    ```
    notebooks/df_info.ipynb
    notebooks/df_plot_dod2k.ipynb
    notebooks/df_filter.ipynb
    ```

3. Applications of DoD2k
   
   1. For analysis of moisture/temperature/moisture and temperature sensitive records use
      
        ```
        notebooks/analysis_M.ipynb
        notebooks/analysis_MT.ipynb
        notebooks/analysis_T.ipynb
        ```
        
    2. For speleothem analysis:
       
       To run ```notebooks/S_analysis_v1.6.ipynb``` you will first need to create the directory ```data/speleothem_modeling_inputs```, and download into it data from their source urls:
    
    ```
    mkdir speleothem_modeling_inputs
    cd speleothem_modeling_inputs
    wget https://wateriso.utah.edu/waterisotopes/media/ArcGrids/GlobalPrecip.zip
    unzip GlobalPrecip.zip
    wget https://crudata.uea.ac.uk/cru/data/hrg/cru_ts_4.07/cruts.2304141047.v4.07/tmp/cru_ts4.07.1901.2022.tmp.dat.nc.gz
    gunzip cru_ts4.07.1901.2022.tmp.dat.nc.gz
    ```










--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### For toolkit use (DT2k)


0. Get the project: in a working directory,
    
    ```bash
    git clone https://github.com/lluecke/dod2k.git
    ```

1. Create and activate the python environment: in dod2k/, 

    ```bash
    conda env create -n dod2k-env -f dod2k-env.yml
    conda activate dod2k-env
    ```

2. Create a common dataframe from source databases (OPTIONAL)
   
    1. Load scripts for input databases:
    
        ```
        data/pages2k/load_pages2k.ipynb
        data/fe23/load_fe23.ipynb
        data/iso2k/load_iso2k.ipynb
        data/sisal/load_sisal.ipynb
        data/ch2k/load_ch2k.ipynb
        ```
        
    2. Merge databases
    
        ```
        data/dod2k/merge_databases.ipynb
        ```

    Note: these notebooks serve for creating compact dataframes from source data and for creating a common dataframe by merging all the databases into one dataframe. If you are not interested in this step, it can be skipped and you can use the compact dataframes as provided in the directories (`csv` or `pkl` files). For altering the source data (e.g. updating a database or adding one), you can add/edit these notebooks accordingly.


4. Run duplicate workflow

    The following steps recreate the complete duplicate workflow.

    1. Duplicate detection: If you have altered any source data, run:
        
        ```
        notebooks/dup_detection.ipynb 
        ```

        This notebook goes through each pair of records to identify potential duplicate candidates. Careful, this will be computationally heavy and may take some time to run!
       The notebooks outputs the file
       ```
       root/data/dod2k/dup_detection/dup_detection_candidates_dod2k.csv
       ```
       This file will be used for the decision process (next step). If you have not changed any source data, you may skip this step and proceed with the next step.
       
    2. Duplicate decision process: run

        ```
        notebooks/dup_decision.ipynb
        ```
         This file walks you through all the potential duplicate candidates and asks for decisions on certain duplicate candidate pairs. The decisions are saved in
       
        ```
        root/data/dod2k/dup_detection/dup_decisions_dod2k_{INITIALS}_{DATECREATED}.csv
        ```
       
        Note: The decision process may be lengthy and get interrupted by server issues.
        However a backup file is created during the workflow and it should be possible to restart where you left off when running the file.
        However in order for this to work it is required that your initials and the date match the backup file!!
        If you restart on another day, it is necessary that you alter the date of the backup file accordingly.
        The backup file can be found here:
        ```
        root/data/dod2k/dup_detection/dup_decisions_dod2k_{INITIALS}_{DATECREATED}_BACKUP.csv
        ```

    3. Duplicate removal process: run
      
        ```
        notebooks/dup_removal.ipynb
        ```
        to implement all the decisions and to create a duplicate free compact dataframe.

5. Rerun the duplicate process (check for remaining duplicates) for `dod2k_dupfree`. Creates `dod2k_dupfree_dupfree` (which is published as DoD2k)
    
6. Explore output (see step #2 for database use)

    If you want to see your own output you will need to alter the `key` for loading according to your initials and the date of the file created:

    ```python

    db_name = 'dod2k_dupfree_dupfree'
    path = 'data/dod2k/'
    file = 'dod2k_dupfree_{INITIALS}_{DATECREATED}_dupfree'
    # load dataframe
    df = utf.load_compact_dataframe_from_csv(db_name, readfrom=(path, filename))
    print(df.info())
    df.name = db_name
    ```
