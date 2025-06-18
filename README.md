This is DoD2k version 1.0.

DOI for the source code: https://doi.org/10.5281/zenodo.15676256

DOI for the data: https://www.ncei.noaa.gov/access/paleo-search/study/41981

How to get started with the dod2k environment, functions, notebooks and products: Quickstart.md
 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This project reads the following five databases (as of 11/12/2024):

- PAGES 2k (v2.0.0: https://springernature.figshare.com/collections/A_global_multiproxy_database_for_temperature_reconstructions_of_the_Common_Era/3285353 (accessed 15/6/23), with Palmyra update (Dee et al. 2020))
- FE 23 (https://cp.copernicus.org/articles/18/2583/2022/ (accessed 28/10/24), https://www.ncei.noaa.gov/access/paleo-search/study/36773)
- SISAL v3 (https:ora.ox.ac.uk/objects/uuid:1e91e2ac-ca9f-46e5-85f3-8d82d4d3cfd4 (accessed 03/06/2024), Kaushal et al (2024) https://essd.copernicus.org/articles/16/1933/2024/ )
- Iso2k (v1.0.1: https://www.ncei.noaa.gov/pub/data/paleo/reconstructions/iso2k/ (accessed 5/11/24), Konecky et al (2020) https://essd.copernicus.org/articles/12/2261/2020/ )
- CoralHydro2k v1.0.0 (https://lipdverse.org/CoralHydro2k/current_version/  (accessed 24/5/2024) https://essd.copernicus.org/articles/15/2081/2023/   https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/noaa-coral-35453/html#)

and creates compact dataframes (with standardised columns and in part standardised metadata):
 - ch2k/load_ch2k.ipynb        which creates   ch2k/ch2k_compact_metadata.csv, ch2k/ch2k_compact_paleoData_values.csv, ch2k/ch2k_compact_year.csv
 - sisal/load_sisal.ipynb      which creates   sisal/sisal_compact_metadata.csv, sisal/sisal_compact_paleoData_values.csv, sisal/sisal_compact_year.csv
 - pages2k/load_pages2k.ipynb  which creates   pages2k/pages2k_compact_metadata.csv, pages2k/pages2k_compact_paleoData_values.csv, pages2k/pages2k_compact_year.csv
 - fe23/load_fe23.ipynb        which creates   fe23/fe23_compact_metadata.csv, fe23/fe23_compact_paleoData_values.csv, fe23/fe23_compact_year.csv
 - iso2k/load_iso2k.ipynb      which creates   iso2k/iso2k_compact_metadata.csv, iso2k/iso2k_compact_paleoData_values.csv, iso2k/iso2k_compact_year.csv
 
these files are concatenated to dod2k via
  - dod2k/load_dod2k.ipynb     which creates    dod2k/dod2k_compact_metadata.csv, dod2k/dod2k_compact_paleoData_values.csv, dod2k/dod2k_compact_year.csv
 
however dod2k is subject to duplicates which need to be found and removed.
The duplicate detection occurs via:
  - dup_detection.ipynb       which creates   dod2k/dup_detection/dup_detection_candidates_dod2k.csv
        = the purpose of this notebook is to identify all the potential duplicate candidates based on certain criteria
        = the output is a list of all the detected potential candidate pairs
  - dup_decisions.ipynb      which creates   dod2k/dup_detection/dup_decisions_dod2k_INITIALS_DATE.csv
        = the purpose of this notebook is to go through each potential duplicate candidates and either automatically decide whether they are TRUE duplicates, or, for the trickier cases, present a summary figure to the operator and ask them for their input.
        = the output is a list of all the decisions associated with the duplicate candidates
  - dup_removal.ipynb        which creates   dod2k/dod2k_INITIALS_DATE_dup_free_metadata.csv, dod2k/dod2k_INITIALS_DATE_dup_free_paleoData_values.csv, dod2k/dod2k_INITIALS_DATE_dup_free_year.csv, dod2k/dod2k_INITIALS_DATE_dup_free_README.csv
        = the purpose of this notebook is to remove the duplicates from the database
        = for the final version of changes, the operator can decide to save a copy as
              dod2k_dupfree/dod2k_dupfree_compact_metadata.csv, dod2k_dupfree/dod2k_dupfree_compact_paleoData_values.csv, dod2k_dupfree/dod2k_dupfree_compact_year.csv, dod2k_dupfree/dod2k_dupfree_compact_README.txt
        = this dod2k_dupfree can itself be used as input for duplicate detection, which creates dod2k_dupfree_dupfree etc., to check for any residual duplicates

The output is presented via the following notebooks:
- df_info.ipynb            which displays the input variables of the dataframe
- df_plot_dod2k.ipynb      which creates a series of plots of the dod2k data

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
This work was supported by a Royal Society of London/Wolfson Visiting Fellowship grant Award \RSWVF\R1\221018 to MNE, which partly supported LJL, US NSF/P4CLIMATE Award AGS2303530 to MNE, which supported KJF and FZ, and by the University of Maryland, College Park and the University of Edinburgh, School of Geosciences, for funding and hosting MNE during a sabbatical year visit.
    
