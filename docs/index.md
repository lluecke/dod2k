<!-- # DoD2k Documentation -->

<!-- 
---
hide:
  - navigation
---
 -->
 <h1 style="text-align: center;">DoD2k: a Database of Databases for the Common Era</h1>

<div style="text-align: center; margin: 2em 0;">
  <p style="font-size: 1.2em; color: var(--md-default-fg-color--light);">
    A Python toolkit for integrating and standardizing global paleoclimate proxy databases
  </p>
</div>


<figure markdown="span">
  ![Global coverage](assets/images/overview_map.png){ width="100%" style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" }
  <figcaption>Global distribution of paleoclimate proxy records in DoD2k v2.0</figcaption>
</figure>

## Welcome!

This documentation allows you to:

- Access and process DoD2k v2.0 or its input databases
- Produce DoD2k from source using the DT2k toolkit
- Merge several databases into a database using a set of standardised metadata
- Screen a databases for duplicates and remove duplicates
- Perform data filtering and plotting.
- Run analysis workflows through Jupyter notebooks (e.g., MT_analysis, T_analysis).

<!-- !!! info "New in DoD2k v2.0"
    - Enhanced duplicate detection algorithm
    - Standardized metadata across all databases
    - Improved documentation and API reference -->

---
<!-- ## Main Features

- **Duplicate detection**: Identify and remove duplicate records across databases ([see the tutorial on duplicate detection](tutorial/duplicate.md))
- **Standardized format**: Consistent metadata structure across all sources 
- **Analysis notebooks**: Reproducible workflows for filtering, plotting, and analysis ([explore DoD2k here](tutorial/use_dod2k.md))
- **Python utilities**: Functions for data processing and visualization ([see the API](api/index.md))

--- -->
<!-- ## At a Glance

<figure markdown="span">
  ![Global coverage](assets/images/overview_map.pdf){ width="100%" style="border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);" }
  <figcaption>Global distribution of paleoclimate proxy records in DoD2k v2.0</figcaption>
</figure>
--- -->

<!-- ## Getting Started

For instructions on setting up the environment, loading or generating DoD2k, and running the notebooks: see the [Quickstart Guide](tutorial/quickstart.md)

!!! tip "Quick Start in 3 Steps"
    1. **Install**: Set up the environment → [Installation Guide](getting_started/installation.md)
    2. **Load**: Access DoD2k v2.0 or generate from source via the duplicate detection workflow
    3. **Visualize/analyze**: Filter for certain metadata and visualise as you wish
 -->


## Get Started

<div class="grid cards" markdown>

-   :fontawesome-solid-rocket:{ .lg } __Quickstart Guide__

    ---
    
    Set up your environment and run your first analysis
    
    [:octicons-arrow-right-24: Get started](getting_started/quickstart.md)

-   :fontawesome-solid-book:{ .lg } __Tutorials__

    ---
    
    Step-by-step workflows for common tasks
    
    [:octicons-arrow-right-24: View tutorials](tutorial/index.md)

-   :fontawesome-solid-code:{ .lg } __API Reference__

    ---
    
    Complete documentation of all functions
    
    [:octicons-arrow-right-24: Browse API](api/index.md)

-   :fontawesome-brands-github:{ .lg } __Source Code__

    ---
    
    Explore and contribute on GitHub
    
    [:octicons-arrow-right-24: View repository](https://github.com/lluecke/dod2k)

</div>

<!-- !!! tip "Quick Start in 3 Steps"
    1. **Install**: Set up the environment → [Installation Guide](installation.md)
    2. **Load**: Access DoD2k v2.0 or generate from source via the duplicate detection workflow
    3. **Visualize/analyze**: Filter for certain metadata and visualise as you wish -->


<!-- ## Key Features

- Duplicate detection and removal notebooks (`dup_detection.ipynb`, `dup_decision.ipynb`, `dup_removal.ipynb`).
- Plotting and summary notebooks of the databases (`df_info.ipynp`, `df_plot_dod2k_v1.ipynb`) for reproducible figures.
- Filtering of the databases for different metadata (`df_filter.ipynb`)
- Utility functions for data filtering and analysis (`ut_analysis.py`, `ut_duplicate_search.py`, `ut_plot.py`, `ut_functions.py`). -->

---

<!-- ## Quick Links

- [Installation](installation.md) – Environment setup
- [Tutorials](tutorial/quickstart.md) – Step-by-step guides
- [API Reference](api/index.md) – Detailed module and function documentation -->


<!-- ## Integrated Databases

- **PAGES2k** (v2.2.0)
    - Data: [https://lipdverse.org/Pages2kTemperature/current_version/](https://lipdverse.org/Pages2kTemperature/current_version/)
    - Article: PAGES 2k Consortium 2017 ([https://www.nature.com/articles/sdata201788](https://www.nature.com/articles/sdata201788]))
- **SISAL** (v3)
    - Data: [https://ora.ox.ac.uk/objects/uuid:1e91e2ac-ca9f-46e5-85f3-8d82d4d3cfd4](https://ora.ox.ac.uk/objects/uuid:1e91e2ac-ca9f-46e5-85f3-8d82d4d3cfd4)
    - Article: Kaushal et al. 2024 ([https://essd.copernicus.org/articles/16/1933/2024/](https://essd.copernicus.org/articles/16/1933/2024/))
- **ISO2k** (v1.1.2)
    - Data: [https://lipdverse.org/iso2k/current_version/](https://lipdverse.org/iso2k/current_version/)
    - Article: Konecky et al. 2020 ([https://essd.copernicus.org/articles/12/2261/2020/](https://essd.copernicus.org/articles/12/2261/2020/))
- **CoralHydro2k** (v1.0.1)
    - Data: [https://lipdverse.org/CoralHydro2k/current_version/](https://lipdverse.org/CoralHydro2k/current_version/)
    - Article: Walter et al. 2023 ([https://essd.copernicus.org/articles/15/2081/2023/](https://essd.copernicus.org/articles/15/2081/2023/))
- **FE23** 
    - Data: [https://www.ncei.noaa.gov/access/paleo-search/study/36773](https://www.ncei.noaa.gov/access/paleo-search/study/36773)
    - Article: Evans et al. 2022 ([https://cp.copernicus.org/articles/18/2583/2022/](https://cp.copernicus.org/articles/18/2583/2022/))

## Integrated Databases

| Database | Version | Data | Reference | Records | Archives
|----------|---------|---------|---------------|---------------|---------------|
| **PAGES 2k** | v2.2.0 | [LiPDverse](https://lipdverse.org/Pages2kTemperature/current_version/) | [PAGES 2k Consortium 2017](https://www.nature.com/articles/sdata201788]) | 1364| multi-proxy |
| **SISAL** | v3 | [ORA](https://ora.ox.ac.uk/objects/uuid:1e91e2ac-ca9f-46e5-85f3-8d82d4d3cfd4) | [Kaushal et al. 2024](https://essd.copernicus.org/articles/16/1933/2024/) | 546 | speleothems |
| **Iso2k** | v1.1.2 | [LiPDverse](https://lipdverse.org/iso2k/current_version/) | [Konecky et al. 2020](https://essd.copernicus.org/articles/12/2261/2020/) |435 | multi-proxy|
| **CoralHydro2k** | v1.0.1 | [LiPDverse](https://lipdverse.org/CoralHydro2k/current_version/) | [Walter et al. 2023](https://essd.copernicus.org/articles/15/2081/2023/) |221| corals|
| **FE23** | - | [NCEI](https://www.ncei.noaa.gov/access/paleo-search/study/36773) | [Evans et al. 2022](https://cp.copernicus.org/articles/18/2583/2022/) |2754| tree-rings |


--- -->

## Documentation

- [Quickstart Guide](getting_started/quickstart.md) - Environment setup and first steps
- [Tutorials](tutorial/index.md) - Step-by-step workflows
- [API Reference](api/index.md) - Function documentation

---



**Source code:** [github.com/lluecke/dod2k](https://github.com/lluecke/dod2k)  
**Citation:** [DOI: 10.5281/zenodo.15676256](https://doi.org/10.5281/zenodo.15676256)