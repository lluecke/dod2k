# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:42:11 2023

Author: Lucie Luecke

Utility functions for loading, saving and cleaning up datasets.
"""
import os
import numpy as np
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import csv
import pandas as pd
import xarray as xr



def write_csv(data, filename, header=False, cols=False):
    
    """
    Write data to a CSV file.
    
    Parameters
    ----------
    data : array-like
        2D array or list of rows to write to CSV. Each row should be an
        iterable of values.
    filename : str
        Output filename without the '.csv' extension.
    
    Returns
    -------
    None
    
    Notes
    -----
    The file will be created in the current working directory with '.csv'
    extension automatically appended.
    """

    
    with open(filename+'.csv', 'w', newline='') as f:
        
        writer = csv.writer(f)
        if header is not False:
            writer.writerows(header)
        if cols is not False:
            writer.writerows(cols)
            
        writer.writerows(data)
    return
    
def write_dataframe_columns_to_csv(data, header, filename, path, ID=False):
    """
    Write dataframe column(s) to a CSV file with optional ID column.
    
    Creates the output directory if it doesn't exist and writes data with
    a header row. Optionally prepends an ID column to each row.
    
    Parameters
    ----------
    data : array-like
        Iterable of column values to write. Each element can be a single value
        (str, float, np.float64) or an iterable of values representing a row.
    header : list of str
        Column names for the CSV header row.
    filename : str
        Output filename without the '.csv' extension.
    path : str
        Relative path from current working directory where file will be saved.
        Should start with '/'.
    ID : array-like, optional
        Array of ID values to prepend to each row. Must have same length as
        data. If False (default), no ID column is added.
    
    Returns
    -------
    None
    
    Notes
    -----
    - Directory structure is created automatically if it doesn't exist
    - Single values (str, float, np.float64) are automatically converted to lists
    - The file is saved as: `cwd + path + filename + '.csv'`
    
    Examples
    --------
    >>> data = [25.5, 30.2, 28.1]
    >>> header = ['ID', 'Temperature']
    >>> IDs = ['Site_A', 'Site_B', 'Site_C']
    >>> write_dataframe_columns_to_csv(data, header, 'temps', '/output', ID=IDs)
    
    >>> # Without ID column
    >>> data = [[1, 2], [3, 4], [5, 6]]
    >>> header = ['X', 'Y']
    >>> write_dataframe_columns_to_csv(data, header, 'coords', '/output')
    """
    # write dataframe column(s) as csv
    if not os.path.exists(os.getcwd()+path):
        os.makedirs(os.getcwd()+path)
        
    with open(os.getcwd()+path+'/'+filename+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for ii, row in enumerate(data):
            if (type(row) is str) or (type(row) is np.float64) or (type(row) is float):
                row=[row]
            if ID is not False:
                writer.writerow([ID[ii]]+ list(row))
            else:
                writer.writerow(list(row))
    return

def  write_compact_dataframe_to_csv(df, saveto='df.name'):
    """
    Save a compact dataframe to three separate CSV files.
    
    Splits a dataframe into metadata, paleoData_values, and year components,
    saving each as a separate CSV file.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing 'datasetId', 'paleoData_values', 'year' columns
        and additional metadata columns.
    saveto : str or tuple, optional
        If 'df.name' (default), uses df.name for path and filename.
        If tuple, should be (path, filename_template) where filename_template
        contains '%s' placeholder for component name.
    
    Returns
    -------
    None
    
    Notes
    -----
    Creates three CSV files:
    - *_compact_metadata.csv : All columns except paleoData_values and year
    - *_compact_paleoData_values.csv : datasetId and paleoData_values columns
    - *_compact_year.csv : datasetId and year columns
    
    Examples
    --------
    >>> write_compact_dataframe_to_csv(df)  # Uses df.name
    >>> write_compact_dataframe_to_csv(df, saveto=('/output', 'data_%s'))
    """

    # sort dataframe alphabetically, with 'datasetId' first:
    other_keys = [kk for kk in df.keys() if kk!='datasetId']
    other_keys.sort()
    keys_sorted = ['datasetId']+other_keys
    # print(keys_sorted)
    df_name = df.name
    df = df[keys_sorted]
    df.name=df_name
    # print(df.info())
    # save to a list of csv files (metadata, data, year)
    for fn in ['metadata', 'paleoData_values', 'year']:
        if saveto=='df.name':
            path = f'/data/{df.name}'
            filename = f'{df.name}_compact_%s'
        else:
            path=saveto[0]
            filename=saveto[1]
        if fn in ['paleoData_values', 'year']:
            header = ['datasetId', fn]
            write_dataframe_columns_to_csv(df[fn].values, 
                                             header, 
                                             filename%fn, path, 
                                             ID=df['datasetId'].values)
        else:
            keys = [key for key in df.columns if key not in ['paleoData_values', 'year']]
            print('METADATA: %s'%', '.join(keys))
            write_dataframe_columns_to_csv(df[keys].values, 
                                             keys, filename%fn, path,
                                             ID=False)
    print(f'Saved to {os.getcwd()+path}/{filename}.csv')
    return

def read_compact_dataframe_columns_from_csv(key, filename, path):
    """
    Read a single column from a compact dataframe CSV file.
    
    Parameters
    ----------
    key : str
        Column name for the data being read (e.g., 'paleoData_values' or 'year').
    filename : str
        Filename without the '.csv' extension.
    path : str
        Relative path from current working directory.
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with index set to IDs from first column and a single column
        named `key` containing the data.
    
    Notes
    -----
    Expects CSV format with first column as ID and remaining columns as data.
    First row is treated as header and skipped.
    """
    ID = []
    data = []
    with open(os.getcwd()+path+'/'+filename+'.csv', 'r', newline='') as f:
        reader = csv.reader(f)
        for ii, row in enumerate(reader):
            if ii==0:
                header=row
                continue
            ID   += [row[0]]
            data += [[row[1:]]]
    df = pd.DataFrame(np.array(data, dtype=object), index=ID, columns=[key])
    return df

def load_compact_dataframe_from_csv(df_name, readfrom='df.name', index_col=0):
    """
    Load a compact dataframe from three CSV files.
    
    Reconstructs a complete dataframe from separate metadata, paleoData_values,
    and year CSV files created by write_compact_dataframe_to_csv.
    
    Parameters
    ----------
    df_name : str
        Name of the dataframe (used for path and filename construction).
    readfrom : str or tuple, optional
        If 'df.name' (default), uses df_name for path and filename.
        If tuple, should be (path, filename_template) where filename_template
        contains '%s' placeholder for component name.
    index_col : int, optional
        Column number to use as index when reading metadata CSV. Default is 0.
    
    Returns
    -------
    pandas.DataFrame
        Complete dataframe with all metadata columns plus paleoData_values
        and year columns. Data types are automatically converted to appropriate
        types (float32 for numeric arrays, str for text fields).
    
    Notes
    -----
    The function:
    - Joins metadata, paleoData_values, and year dataframes on datasetId
    - Converts array strings to numpy float32 arrays
    - Enforces specific data types for standard columns
    - Resets index to sequential integers
    
    Examples
    --------
    >>> df = load_compact_dataframe_from_csv('dod2k')
    >>> df = load_compact_dataframe_from_csv('pages2k', readfrom=('/data', 'p2k_%s'))
    """
    # load the compact dataframe from three csv files 
    if readfrom=='df.name':
        path = f'/data/{df_name}'
        filename = df_name+'_compact_%s'
    else:
        path=readfrom[0]
        filename=readfrom[1]
    
    df_meta = pd.read_csv(os.getcwd()+path+'/'+filename%'metadata'+'.csv', index_col=index_col, keep_default_na=False)
    df_data = read_compact_dataframe_columns_from_csv('paleoData_values', filename%'paleoData_values', path)
    df_year = read_compact_dataframe_columns_from_csv('year', filename%'year', path)
    
    df_csv = df_meta.join(df_data).join(df_year)
    
    df_csv['datasetId'] = df_csv.index
    df_csv.index=range(len(df_csv))
    

    df_csv = df_csv[sorted(df_csv.columns)]

    
    df_csv['year'] = df_csv['year'].map(parse_array_string)
    df_csv['paleoData_values'] = df_csv['paleoData_values'].map(parse_array_string)
    
    # df_csv['year'] = df_csv['year'].map(lambda x: np.array(x, dtype = np.float32))
    # df_csv['paleoData_values'] = df_csv['paleoData_values'].map(lambda x: np.array(x, dtype = np.float32))

    df_csv = df_csv.astype({'archiveType': str, 
                            'dataSetName': str, 
                            'datasetId': str, 
                            'geo_meanElev': np.float32, 
                            'geo_meanLat': np.float32, 
                            'geo_meanLon': np.float32, 
                            'geo_siteName': str, 
                            'interpretation_direction': str,
                            'interpretation_seasonality': str,
                            'interpretation_variable': str,
                            'interpretation_variableDetail': str,
                            'originalDatabase': str, 
                            'originalDataURL': str, 
                            'paleoData_notes': str, 
                            'paleoData_proxy': str,
                            'paleoData_sensorSpecies': str,
                            'paleoData_units': str, 
                            'paleoData_variableName': str, 
                            'yearUnits': str})

    df_csv.name = df_name
    return df_csv


def read_csv(filename, dtype=str, header=False, last_header_row=0):
    """
    Read data from a CSV file with optional header extraction.
    
    Parameters
    ----------
    filename : str
        Filename without the '.csv' extension.
    dtype : type or None, optional
        Data type for the output array. If None, returns list. Default is str.
    header : bool, optional
        If True, extracts header rows (lines starting with '#' or before
        last_header_row). Default is False.
    last_header_row : int, optional
        Index of the last header row (0-indexed). Default is 0.
    
    Returns
    -------
    data : numpy.ndarray or list
        The CSV data as array or list depending on dtype.
    header : list, optional
        List of header lines (only returned if header=True).
    
    Examples
    --------
    >>> data = read_csv('myfile')  # Returns array of strings
    >>> data, hdr = read_csv('myfile', header=True, last_header_row=2)
    >>> data = read_csv('myfile', dtype=None)  # Returns list
    """
    # reads csv file
    if header:
        header = []
        data   = []
        with open(filename+'.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for irow, row in enumerate(reader):
                if row[0].startswith('#') or irow<=last_header_row:
                    # header.append(row[0].replace('#',''))
                    if row[0].startswith('#'):
                        row[0] = row[0].replace('#','')
                    header.append(','.join(row))  # Changed from row[0] to row
                else:
                    data.append(row)
        if not dtype:
            return np.array(data, dtype=dtype), header
        else:
            return data, header
    else:
        data = []
        with open(filename+'.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        if not dtype:
            return np.array(data, dtype=dtype)
        else:
            return data
        
def find(pattern, path):
    """
    Find all files containing a pattern in their name within a directory tree.
    
    Parameters
    ----------
    pattern : str
        String pattern to search for in filenames.
    path : str
        Root directory path to search recursively.
    
    Returns
    -------
    list of str
        List of full file paths for all matching files.
    
    Examples
    --------
    >>> csv_files = find('.csv', '/data/output')
    >>> metadata_files = find('metadata', '.')
    """
    import datetime

    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if pattern in name:
                result.append(os.path.join(root, name))
    return result
    
def figsave(fig, name, trans=False, add='/', fc='white',
            form='pdf', close=False, addfigs='/figs/'):
    """
    Save a matplotlib figure to file with optional formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    name : str
        Filename without extension.
    trans : bool, optional
        If True, save with transparent background. Default is False.
    add : str, optional
        Additional subdirectory path within addfigs. Default is '/'.
    fc : str, optional
        Face color for non-transparent backgrounds. Default is 'white'.
    form : str, optional
        File format ('pdf', 'png', 'jpg', etc.). Default is 'pdf'.
    close : bool, optional
        If True, close the figure after saving. Default is False.
    addfigs : str, optional
        Base directory for saving figures. Default is '/figs/'.
    
    Returns
    -------
    None
    
    Notes
    -----
    - Creates directory structure if it doesn't exist
    - If format is 'pdf', also saves a JPG version at 100 dpi
    - Saves with tight bounding box and no padding
    """
    # from datetime import date
    # day = date.today().isoformat()
    # ddir = day+'/'+add if save_date else add
    ddir = add
    if not os.path.exists(os.getcwd()+addfigs+ddir):
        os.makedirs(os.getcwd()+addfigs+ddir)
    
    fig.savefig(os.getcwd()+addfigs+ddir+'/'+name+'.'+form, 
                transparent=trans, facecolor=fig.get_facecolor(),
                format=form, bbox_inches='tight', pad_inches=0.0)
    if form=='pdf': 
        fig.savefig(os.getcwd()+addfigs+ddir+'/'+name+'.jpg', 
                    transparent=trans, facecolor=fig.get_facecolor(),
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
    print('saved figure in '+ addfigs+ddir+'/'+name+'.'+form)
    if close: plt.close()
    return

def save_fig(fig, filename, trans=False, dir='/', fc='white',
            figformat='pdf', close=False, addfigs=True):
    """
    Save a matplotlib figure to file with optional formats.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save.
    filename : str
        Filename without extension.
    trans : bool, optional
        If True, save with transparent background. Default is False.
    dir : str, optional
        Additional subdirectory path. Default is '/'.
    fc : str, optional
        Face color for non-transparent backgrounds. Default is 'white'.
    figformat : str, optional
        File format ('pdf', 'png', 'jpg', etc.). Default is 'pdf'.
    close : bool, optional
        If True, close the figure after saving. Default is False.
    addfigs : bool, optional
        If True, add '/figs/' to the rooth path. Default is True.
    
    Returns
    -------
    None
    
    Notes
    -----
    - Creates directory structure if it doesn't exist
    - If format is 'pdf', also saves a JPG version at 100 dpi
    - Saves with tight bounding box and no padding
    """
    fig_dir = os.getcwd()
    if addfigs: fig_dir+='/figs/'
    fig_dir+=dir
    
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    
    fig.savefig(fig_dir+'/'+filename+'.'+figformat, 
                transparent=trans, facecolor=fig.get_facecolor(),
                format=figformat, bbox_inches='tight', pad_inches=0.0)
    if figformat=='pdf': 
        fig.savefig(fig_dir+'/'+filename+'.png', 
                    transparent=trans, facecolor=fig.get_facecolor(),
                    format='png', dpi=300, bbox_inches='tight', pad_inches=0.0)
        
    print('saved figure in '+ fig_dir+'/'+filename+'.'+figformat)
    if close: plt.close()
    return

def cleanup(string):
    """
    Remove special characters and leading/trailing whitespace from a string.
    
    Parameters
    ----------
    string : str
        Input string to clean.
    
    Returns
    -------
    str
        Cleaned string with '#', tabs, newlines removed and whitespace stripped.
    
    Examples
    --------
    >>> cleanup('#  Temperature  \\n')
    'Temperature'
    >>> cleanup('\\t Data \\t')
    'Data'
    """
    string = string.replace('#', '')
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    while string.startswith(' '): string = string[1:]
    while string.endswith(' '): string = string[:-1]
    return string


def fns(path, end='.nc', start='', other_cond='', print_dir=True):
    """
    Find filenames in a directory matching specific criteria.
    
    Parameters
    ----------
    path : str
        Directory path to search.
    end : str, optional
        File extension to match. Default is '.nc'.
    start : str, optional
        String that filename must start with. Default is '' (any).
    other_cond : str, optional
        Additional substring that must be in filename. Default is '' (any).
    print_dir : bool, optional
        If True, print the list of found files. Default is True.
    
    Returns
    -------
    numpy.ndarray
        Sorted array of matching filenames.
    
    Examples
    --------
    >>> # Find all NetCDF files
    >>> files = fns('/data/climate/')
    >>> # Find CSV files starting with 'temp'
    >>> files = fns('/data/', end='.csv', start='temp')
    >>> # Find files containing 'annual' in name
    >>> files = fns('/data/', other_cond='annual')
    """
    fn = []
    for root, dirs, files in os.walk(path):
        for fil in files:
            if fil.endswith(end) and fil.startswith(start) and (other_cond in fil):
                fn.append(fil)
    if print_dir: print(fn)
    fn = np.sort(fn)
    return fn 

def conv_nan(value):
    """
    Convert NaN values to missing data indicator (-9999.99).
    
    Parameters
    ----------
    value : float or str
        Value to check and potentially convert.
    
    Returns
    -------
    float
        Original value if valid, or -9999.99 if NaN/missing.
    
    Notes
    -----
    Values already equal to -9999.99 are preserved to avoid double-conversion.
    """
    if value=='nan' or np.isnan(value):
        return -9999.99
    elif np.round(value, 2)==-9999.99:
        return value
    else:
        return value


def convert_to_nparray(data):
    """
    Convert data array to masked array with missing values marked.
    
    Converts NaN values and 'nan' strings to -9999.99, then creates a
    masked array with these values masked out.
    
    Parameters
    ----------
    data : array-like
        Input data that may contain NaN or 'nan' string values.
    
    Returns
    -------
    numpy.ma.MaskedArray
        Masked array with -9999.99 values masked.
    """
    data = np.array([conv_nan(vv) for vv in data])#, -9999.99)
    mask = np.round(data, 2)==-9999.99#np.array([False if np.round(vv, 2)!=-9999.99 else True for vv in data ])
    return np.ma.masked_array(data, mask)
    
def convert_to_float(txt):
    """
    Convert input to a float, returning a sentinel value on failure.

    Parameters
    ----------
    txt : any
        Input to be converted to a float. Typically a string, but any
        object accepted by ``float()`` is allowed.

    Returns
    -------
    float
        Parsed floating-point value if conversion succeeds.
        Returns ``-9999.99`` if conversion fails for any reason
        (e.g., invalid string, None, incompatible type).
    """
    try:
        return float(txt)
    except:
        return -9999.99


def parse_array_string(x):
    """
    Parse a string representation of a numeric array into a NumPy array.

    Parameters
    ----------
    x : str, list, or array-like
        Input representing an array. Supported forms include:
        - A string of comma-separated numbers, optionally enclosed
          in square brackets (e.g., ``"[1, 2, 3]"`` or ``"1,2,3"``).
        - A list containing a single such string.
        - A list or array-like object of numeric values.

    Returns
    -------
    numpy.ndarray
        One-dimensional NumPy array of type ``np.float32`` constructed
        from the parsed values.
    """
    if isinstance(x, str):
        # Remove brackets and split by comma
        x = x.strip('[]').replace('\n', '').replace(' ', '')
        if x:
            try:
                return np.fromstring(x, sep=',', dtype=np.float32)
            except:
                # If parsing fails, try eval (careful with this!)
                return np.array(eval('[' + x + ']'), dtype=np.float32)
    elif isinstance(x, list) and len(x) == 1 and isinstance(x[0], str):
        # Handle case where x is a list containing a string
        return parse_array_string(x[0])
    return np.array(x, dtype=np.float32)
