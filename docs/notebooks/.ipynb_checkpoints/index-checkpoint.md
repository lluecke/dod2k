# Notebooks

Interactive Jupyter notebooks for working with DoD2k. Each notebook provides a complete workflow for specific tasks.

---

## Loading Individual Databases

<div class="grid cards" markdown>

-   :material-file-download:{ .lg .middle } __Load CoralHydro2k__

    ---

    Load and process the CoralHydro2k database
    
    [:octicons-arrow-right-24: Open notebook](load_ch2k.ipynb)

-   :material-file-download:{ .lg .middle } __Load FE23__

    ---

    Load and process the FE23 database
    
    [:octicons-arrow-right-24: Open notebook](load_fe23.ipynb)

-   :material-file-download:{ .lg .middle } __Load Iso2k__

    ---

    Load and process the Iso2k database
    
    [:octicons-arrow-right-24: Open notebook](load_iso2k.ipynb)

-   :material-file-download:{ .lg .middle } __Load PAGES2k__

    ---

    Load and process the PAGES2k database
    
    [:octicons-arrow-right-24: Open notebook](load_pages2k.ipynb)

-   :material-file-download:{ .lg .middle } __Load SISAL__

    ---

    Load and process the SISAL speleothem database
    
    [:octicons-arrow-right-24: Open notebook](load_sisal.ipynb)

-   :material-merge:{ .lg .middle } __Merge Databases__

    ---

    Combine multiple databases into DoD2k
    
    [:octicons-arrow-right-24: Open notebook](merge_databases.ipynb)

</div>

---

## Duplicate Detection Workflow

<div class="grid cards" markdown>

-   :material-file-search:{ .lg .middle } __Step 1: Detect Duplicates__

    ---

    Identify potential duplicate records across databases
    
    [:octicons-arrow-right-24: Open notebook](dup_detection.ipynb)

-   :material-checkbox-marked:{ .lg .middle } __Step 2: Review Duplicates__

    ---

    Review and classify detected duplicate candidates
    
    [:octicons-arrow-right-24: Open notebook](dup_decision.ipynb)

-   :material-delete:{ .lg .middle } __Step 3: Remove Duplicates__

    ---

    Remove confirmed duplicates from the database
    
    [:octicons-arrow-right-24: Open notebook](dup_removal.ipynb)

</div>

---

## Visualization & Analysis

<div class="grid cards" markdown>

-   :material-information:{ .lg .middle } __Explore Database Info__

    ---

    View summary statistics and metadata of the compact dataframe
    
    [:octicons-arrow-right-24: Open notebook](df_info.ipynb)

-   :material-chart-line:{ .lg .middle } __Plot DoD2k__

    ---

    Create visualizations of proxy records and spatial distributions
    
    [:octicons-arrow-right-24: Open notebook](df_plot_dod2k.ipynb)

-   :material-filter:{ .lg .middle } __Filter Database__

    ---

    Filter records by metadata criteria (archive type, location, etc.)
    
    [:octicons-arrow-right-24: Open notebook](df_filter.ipynb)

</div>

---

## Quick Access

**Typical workflows:**

1. **Load data** → Use individual load notebooks or merge_databases.ipynb
2. **Clean data** → Run duplicate detection workflow (3 steps)
3. **Explore data** → Use df_info.ipynb and df_plot_dod2k.ipynb
4. **Filter data** → Use df_filter.ipynb for targeted analysis