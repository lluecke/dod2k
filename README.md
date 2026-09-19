# Nivi Munjal, Ethan Hartz, Sarah Bransky, Jennifer Ding, Kevin Fan

Simepl and deep clustering of tree ring width archive data

## Databases used


DoD2k (Database of Databases 2k) integrates five major paleoclimate databases:


| Database | Version | Data | Reference | Records | Archives
|----------|---------|---------|---------------|---------------|---------------|
| **PAGES 2k** | v2.2.0 | [LiPDverse](https://lipdverse.org/Pages2kTemperature/current_version/) | [PAGES 2k Consortium 2017](https://www.nature.com/articles/sdata201788]) | 1364| multi-proxy |
| **SISAL** | v3 | [ORA](https://ora.ox.ac.uk/objects/uuid:1e91e2ac-ca9f-46e5-85f3-8d82d4d3cfd4) | [Kaushal et al. 2024](https://essd.copernicus.org/articles/16/1933/2024/) | 546 | speleothems |
| **Iso2k** | v1.1.2 | [LiPDverse](https://lipdverse.org/iso2k/current_version/) | [Konecky et al. 2020](https://essd.copernicus.org/articles/12/2261/2020/) |435 | multi-proxy|
| **CoralHydro2k** | v1.0.1 | [LiPDverse](https://lipdverse.org/CoralHydro2k/current_version/) | [Walter et al. 2023](https://essd.copernicus.org/articles/15/2081/2023/) |221| corals|
| **FE23** | - | [NCEI](https://www.ncei.noaa.gov/access/paleo-search/study/36773) | [Franke, Evans et al. 2022](https://cp.copernicus.org/articles/18/2583/2022/) |2754| tree-rings |

## Structure

Final_Project_Coding_SimpleClustering_CMSC472.ipynb - Simple clustering off geographical data, shows spatial and frequency plots

dod2k_v2.0_Wood.pkl
dod2k_v2.0_Wood_compact_metadata.csv
dod2k_v2.0_Wood_compact_year.csv
dod2k_v2.0_Wood_compact_paleoData_values.csv - All associated data and metadata used in building out clustering programs

load_trw.ipynb - deep clustering notebook.