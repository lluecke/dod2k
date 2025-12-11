# -*- coding: utf-8 -*-
"""

Author: Lucie Luecke (includes functions by Feng Zhu and Kevin Fan)

Provides functions for filtering, homogenising, manipulating and analysing data(frames).

"""

import numpy as np
import matplotlib.pyplot as plt
import dod2k.functions as f
from functools import reduce
from matplotlib.gridspec import GridSpec as GS
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib



def filter_resolution(df, maxres):
    """
    Filter records in a DataFrame based on maximum resolution.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'resolution' column, where each entry is a list 
        of numeric resolution values.
    maxres : int or float
        Maximum allowed resolution. Records with all resolution values less than
        or equal to `minres` are kept.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame containing only records with resolution <= minres.
    """
    rmask = df.resolution.apply(lambda x: np.all(np.array(x)<=maxres))
    print('Keep %d records with resolution <=%d. Exclude %d records.'%(len(df[rmask]), maxres, len(df[~rmask])))
    
    return df[rmask]
    
def filter_record_length(df, nyears, mny, mxy):
    """
    Filter records based on the number of years with data in a given range.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'year' column, where each entry is an array-like
        of years available for the record.
    nyears : int
        Minimum number of years required within the specified range.
    mny : int
        Start year of the range.
    mxy : int
        End year of the range.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame containing only records with at least `nyears`
        of data between `mny` and `mxy`.
    """

    remove = []
    for ii in df.index:
        if np.sum((df.at[ii, 'year']>=mny)&(df.at[ii, 'year']<=mxy))<nyears:
            # print('No available data', ii)
            remove+=[ii]
    
    df=df.drop(labels=remove)
    # mask   = ~(df.length>=nyears)
    print('Keep %d records with nyears>=%d during %d-%d. Exclude %d records.'%(df.shape[0], nyears, mny, mxy, len(remove)))    
    return df



def filter_data_availability(df, mny, mxy):
    """
    Filter records based on data availability within a given year range.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'year' column, where each entry is an array-like
        of years available for the record.
    mny : int
        Start year of the range.
    mxy : int
        End year of the range.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame containing only records that have data available
        between `mny` and `mxy`.
    """
    remove = []
    for ii in df.index:
        if np.sum((df.at[ii, 'year']>=mny)&(df.at[ii, 'year']<=mxy))==0:
            # print('No available data', ii)
            remove+=[ii]
    df=df.drop(labels=remove)
    print('No available data: ', remove)
    print('Keep %d records with data available between %d-%d. Exclude %d records.'%(df.shape[0], mny, mxy, len(remove)))

    return df


def homogenise_time(df, mny, mxy, minres):
    """
    Homogenise the time coordinate of records in a DataFrame.

    This function creates a uniform time axis from `mny` to `mxy` with 
    steps of `minres` years and prints basic information about the 
    homogenised timeline. It also calls `find_shared_period` to report 
    the overlapping period across all records.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing climate or archive records. Must be compatible
        with `find_shared_period`.
    mny : int
        Start year of the homogenisation period.
    mxy : int
        End year of the homogenisation period.
    minres : int
        Minimum resolution (in years) for the homogenised timeline.

    Returns
    -------
    df : pandas.DataFrame
        The input DataFrame (unchanged in this function).
    years_hom : numpy.ndarray
        Array of homogenised years from `mny` to `mxy` with step `minres`.
    """
    
    years_hom     = np.arange(mny, mxy+minres, minres)                                       #
    
    print('Homogenised time coordinate: %d-%d CE'%(years_hom[0], years_hom[-1]))
    print('Resolution: %s years'%str(np.unique(np.diff(years_hom))))

    find_shared_period(df, minmax=(mny, mxy))
    
    return df, years_hom


def homogenise_data_dimensions(df, years_hom, title='', print_output=False, plot_output=True):
    """
    Homogenise the data arrays to a uniform time coordinate.

    This function assigns paleoData values and z-scores from each record in the
    DataFrame to a homogenised time axis (`years_hom`). Missing data are masked
    as zeros using numpy masked arrays. Optional plotting and printing of 
    intermediate checks is provided.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the records with columns:
        - 'year': array-like of years
        - 'paleoData_values': array-like of values
        - 'paleoData_zscores': array-like of z-scores
    years_hom : array-like
        Homogenised time coordinate to which all records are aligned.
    title : str, optional
        Title used for the plot if `plot_output` is True.
    print_output : bool, optional
        If True, prints debug information about array sizes and resolutions.
        Default is False.
    plot_output : bool, optional
        If True, generates diagnostic plots showing the homogenised and original
        paleoData values and z-scores. Default is True.

    Returns
    -------
    paleoData_values_hom : numpy.ma.MaskedArray
        Masked array of shape (n_records, n_years) containing homogenised
        paleoData values. Missing values are masked.
    paleoData_zscores_hom : numpy.ma.MaskedArray
        Masked array of shape (n_records, n_years) containing homogenised
        paleoData z-scores. Missing values are masked.
    year_hom_avbl : list of numpy.ma.MaskedArray
        List of length n_records containing the homogenised data arrays for
        paleoData_values.
    zsco_hom_avbl : list of numpy.ma.MaskedArray
        List of length n_records containing the homogenised data arrays for
        paleoData_zscores.
    """

    mny = years_hom[0]
    mxy = years_hom[-1]
    minres = np.unique(np.diff(years_hom))[0]

    n_recs = len(df) 
    
    # assign data values
    paleoData_values_hom  = np.ma.masked_array(np.zeros([n_recs, len(years_hom)]), np.zeros([n_recs, len(years_hom)]))
    paleoData_zscores_hom = np.ma.masked_array(np.zeros([n_recs, len(years_hom)]), np.zeros([n_recs, len(years_hom)]))

    year_hom_avbl = []
    zsco_hom_avbl = []
    
    for ijk, ii in enumerate(df.index):
        # create empty data arrays 

        time = df.at[ii, 'year']
        
        
        data_LR = np.zeros(len(years_hom))
        data_HR = df.at[ii, 'paleoData_values']
        
        zsco_HR = df.at[ii, 'paleoData_zscores']
        zsco_LR = np.zeros(len(years_hom))

        tt = []
        zz = []
        for jj, xi in enumerate(years_hom):
            window = (time>xi-minres)&(time<=xi)
            # print(xi, time[window])
            if len(time[window])==0:
                data_LR[jj] = 0#np.nan
                zsco_LR[jj] = 0#np.nan
            else:
                data_LR[jj] = np.average(data_HR[window])
                zsco_LR[jj] = np.average(zsco_HR[window])
                tt+=[xi]
                zz+=[np.average(zsco_HR[window])]
        

        yh_base = np.ma.masked_array(data_LR, mask=data_LR==0, fill_value=0)
        zh_base = np.ma.masked_array(zsco_LR, mask=zsco_LR==0, fill_value=0)
        
        # # check array is correct
        if print_output:
            # print(ii, ijk, 'years_hom size: ', years_hom[hmask].shape, 'new array size: ', 
            print(ii, ijk, 'years_hom size: ', years_hom.shape, 'new array size: ', 
                  yh_base[~yh_base.mask].shape, 'resolution: ', np.unique(np.diff(time)), 
                  # 'time coord: from %s-%s'%(yy[(yy>=mny)&(yy<=mxy)][0], yy[(yy>=mny)&(yy<=mxy)][-1])
                 )
            print(paleoData_values_hom[ijk,:].shape, yh_base.shape)  
            
        paleoData_values_hom[ijk,:]  = yh_base
        paleoData_zscores_hom[ijk,:] = zh_base
        year_hom_avbl.append(tt)
        zsco_hom_avbl.append(zz)

    print(paleoData_values_hom.shape)
    
    if plot_output:
        n_recs=min(len(df), 50)
        # plot paleoData_values_hom and paleoData_zscores_hom as they appear in df
        fig = plt.figure(figsize=(8,5))
        plt.suptitle(title)
        plt.subplot(221)
        plt.title('paleoData_values HOM')
        for ii in range(n_recs):
            shift = ii
            plt.plot(years_hom, paleoData_values_hom[ii,:]+shift, lw=1)
        plt.xlim(mny, mxy)
            
        plt.subplot(222)
        plt.title('paleoData_values')
        for ii in range(n_recs):
            shift = ii
            plt.plot(df.year.iloc[ii], df.paleoData_values.iloc[ii]+shift, lw=1)
        plt.xlim(mny, mxy)
        
        plt.subplot(223)
        plt.title('paleoData_zscores HOM')
        for ii in range(n_recs):
            shift = ii
            plt.plot(years_hom, paleoData_zscores_hom[ii,:]+shift , lw=1)
        plt.xlim(mny, mxy)
        
        plt.subplot(224)
        plt.title('paleoData_zscores')
        for ii in range(n_recs):
            shift = ii
            plt.plot(df.year.iloc[ii], df.paleoData_zscores.iloc[ii]+shift, lw=1)
        plt.xlim(mny, mxy)
        fig.tight_layout()
    return paleoData_values_hom, paleoData_zscores_hom, year_hom_avbl, zsco_hom_avbl  



def covert_subannual_to_annual(df):
    """
    Convert sub-annual data to annual averages.

    For each record in the DataFrame, this function computes yearly averages
    of the 'paleoData_values' and replaces the original 'year' array with 
    integer years.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'year' and 'paleoData_values' columns.

    Returns
    -------
    None
        The function modifies the DataFrame in place.
    """
    for ii in df.index:
        year = df.at[ii, 'year']
        sy = np.min(year)
        year_ar = np.unique(np.floor(year))
        # print(year_ar)
        data_ar = []
        for yy in year_ar:
            mask = np.floor(year)==yy
            # print(df.at[ii, 'year'][mask])
            data_ar+=[np.mean(df.at[ii, 'paleoData_values'][mask])]
        # print(data_ar)
        df.at[ii, 'paleoData_values'] = np.array(data_ar)
        df.at[ii, 'year'] = year_ar
    return 

def find_shared_period(df, minmax=False, time='year', data='paleoData_zscores'):
    """
    Determine the shared time period across all records.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the records with time and data columns.
    minmax : tuple or list, optional
        Year range to use for plotting if no shared period exists. Default is False.
    time : str, optional
        Name of the column containing the time axis. Default is 'year'.
    data : str, optional
        Name of the data column to plot if no shared period exists. Default is 'paleoData_zscores'.

    Returns
    -------
    miny : int or float
        Minimum year of the shared period, or np.nan if none.
    maxy : int or float
        Maximum year of the shared period, or np.nan if none.
    """
    try:
    
        miny = np.min(reduce(np.intersect1d, df[time]))
        maxy = np.max(reduce(np.intersect1d, df[time]))
        print('INTERSECT: %d-%d'%(miny, maxy))
    except ValueError:
        print('No shared period across all records.')
        miny = np.nan
        maxy = np.nan
        plt.figure()
        for jj, ii in enumerate(df.index):
            dd = df.at[ii, data]
            yy = df.at[ii, time]
            plt.plot(yy, dd-np.mean(dd)+jj)
            if minmax: plt.xlim(minmax[0], minmax[-1])
    return miny, maxy
 

def calc_z_score(x):
    """
    Calculate the z-score of paleoData_values for a single record.

    Parameters
    ----------
    x : pandas.Series
        Series representing a single record with a 'paleoData_values' array.

    Returns
    -------
    numpy.ndarray
        Z-scored values of the record.
    """
    # calculate z-score 
    z = x.paleoData_values-np.mean(x.paleoData_values)
    z /= np.std(x.paleoData_values)
    return z

def add_zscores_to_df(df, key, plot_output=True):
    """
    Add z-scores of paleoData_values to the DataFrame.

    This function calculates the z-score for each record and adds a new column
    'paleoData_zscores'. Optionally plots the original values and z-scores.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'year' and 'paleoData_values'.
    key : str
        Title for the plot.
    plot_output : bool, optional
        If True, generates a diagnostic plot. Default is True.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with an added 'paleoData_zscores' column.
    """
    
    df['paleoData_zscores'] = df.apply(lambda x: calc_z_score(x), axis=1)

    
    # plot paleoData_values and paleoData_zscores
    fig = plt.figure()
    plt.suptitle(key)
    plt.subplot(211)
    plt.ylabel('paleoData_values')#, y=0.85)
    for ii in df.index:
        plt.plot(df.at[ii, 'year'],
                 df.at[ii, 'paleoData_values'], lw=1)
    plt.xticks([])
    plt.subplot(212)
    plt.ylabel('paleoData_zscores')#, y=0.85)
    plt.xlabel('year CE')
    for ii in df.index:
        plt.plot(df.at[ii, 'year'],
                 df.at[ii, 'paleoData_zscores'], lw=1)
    fig.tight_layout()
    return df

def add_aux_variables(df, key, mincount=0, **kwargs):
    """
    Add auxiliary variables to the DataFrame and generate summary plots.

    Adds 'length', 'miny', and 'maxy' columns, then plots coverage, resolution,
    and length distributions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'year' and 'paleoData_values'.
    key : str
        Title for plots.
    mincount : int, optional
        Minimum count threshold for plotting resolution and length. Default is 0.
    **kwargs : dict
        Additional keyword arguments passed to `plot_coverage`.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with added auxiliary columns.
    """
    
    # add 'length, miny, maxy' to dataframe
    df['length'] = df.paleoData_values.apply(len)
    df['miny']   = df.year.apply(np.min)
    df['maxy']   = df.year.apply(np.max)
    
    years    = np.arange(min(df.miny), max(df.maxy)+1)
    plot_coverage(df, years, key, **kwargs)
    
    add_resolution_to_df(df, print_output=True)
    plot_resolution(df, key, mincount=mincount)
    plot_length(df, key, mincount=mincount)
    return df

def add_resolution_to_df(df, print_output=False, plot_output=False):
    """
    Compute the time resolution of each record and store in the DataFrame.

    Sorts the time and data values, then calculates the resolution as the 
    unique differences between consecutive years.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'year' and 'paleoData_values'.
    print_output : bool, optional
        If True, prints debug information. Default is False.
    plot_output : bool, optional
        Currently unused; reserved for future plotting. Default is False.

    Returns
    -------
    None
        The function updates the 'resolution' column of the DataFrame in place.
    """

    # sort year and data values and obtain resolution
    df['paleoData_values']= df.apply(lambda x: x.paleoData_values[np.argsort(x.year)], axis=1)
    df['year']= df.apply(lambda x: np.round(x.year[np.argsort(x.year)], 2), axis=1)
    df['resolution']= df.year.apply(np.diff).apply(np.unique)
    
    
    return 


def calc_covariance_matrix(df):
    """
    Compute the covariance matrix and overlapping years for all records.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing homogenised z-score arrays (`paleoData_zscores_hom_avbl`) 
        and their corresponding available years (`year_hom_avbl`).

    Returns
    -------
    covariance : numpy.ndarray
        Covariance matrix of shape (n_records, n_records) between all records.
    overlap : numpy.ndarray
        Matrix of shape (n_records, n_records) containing the number of overlapping 
        years between each pair of records.
    """
    n_recs = len(df)
    
    covariance = np.zeros([n_recs, n_recs])
    overlap    = np.zeros([n_recs, n_recs])
    for ii in range(n_recs):
        for jj in range(ii, n_recs):
            # print(ii, jj)
            
            time_1 = df.iloc[ii].year_hom_avbl
            time_2 = df.iloc[jj].year_hom_avbl
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) # saves intersect between the records
            
            overlap[ii, jj] = len(time_12)
            overlap[jj, ii] = len(time_12)
    
            data_1 = df.iloc[ii].paleoData_zscores_hom_avbl
            data_1 -= np.mean(data_1)
            # data_1 /= np.std(data_1)
            data_2 = df.iloc[jj].paleoData_zscores_hom_avbl
            data_2 -= np.mean(data_2)
            # data_2 /= np.std(data_2)
            covariance[ii, jj] = np.cov(data_1[int_1], data_2[int_2], bias=False)[0,1]      #Default normalization (False) is by (N - 1), where N is the number of observations given (unbiased estimate). If bias is True, then normalization is by N.
            
            covariance[jj, ii] =covariance[ii, jj]
            
    print('short records : ', overlap[overlap<40])
    return covariance, overlap


def PCA(covariance):
    """
    Performs principal component analysis using singular value decomposition (SVD).

    Parameters
    ----------
    covariance : numpy.ndarray
        Covariance matrix of shape (n_records, n_records).

    Returns
    -------
    eigenvalues : numpy.ndarray
        Eigenvalues from SVD (singular values).
    eigenvectors : numpy.ndarray
        Eigenvectors corresponding to the covariance matrix.
    """
    U, s, Vh = np.linalg.svd(covariance) # s eigenvalues, U, Vh rotation matrices

    eigenvalues  = s
    eigenvectors = Vh
    

    return eigenvalues, eigenvectors
    
def fraction_of_explained_var(covariance, eigenvalues, n_recs, title='', db_name=''):
    """
    Compute and plot the fraction of variance explained by principal components.

    Parameters
    ----------
    covariance : numpy.ndarray
        Covariance matrix of the records.
    eigenvalues : numpy.ndarray
        Eigenvalues from PCA.
    n_recs : int
        Number of records.
    title : str, optional
        Title used in the plot.
    db_name : str, optional
        Name suffix for saving the figure.

    Returns
    -------
    frac_explained_var : numpy.ndarray
        Fraction of variance explained by each principal component.
    """
    sorter = np.argsort(eigenvalues)[::-1] # sort eigenvalues in descending order
    
    explained_var  = eigenvalues[sorter]**2/ (n_recs - 1) 
    
    total_var = np.sum(explained_var)
    frac_explained_var = explained_var / total_var
    
    cum_frac_explained_var = np.cumsum(frac_explained_var)

    fig = plt.figure()
    plt.title(title)
    ax = plt.gca()
    plt.plot(np.arange(len(frac_explained_var))+1, frac_explained_var, label='fraction of explained variance')
    plt.xlim(-1, 10)
    plt.ylabel('fraction of explained variance')
    
    plt.xlabel('PC')
    
    ax1 = ax.twinx()
    ax1.plot(np.arange(len(frac_explained_var))+1, cum_frac_explained_var, ls=':', label='cumulative fraction of explained variance')
    plt.ylabel('cumulative fraction of explained variance') 
    
    f.figsave(fig, 'foev_%s'%title, add=db_name)

    return frac_explained_var


def smooth(data, time, res):
    """
    Apply simple moving average smoothing to time series data.

    Parameters
    ----------
    data : numpy.ndarray
        Time series data array (1D or 2D).
    time : numpy.ndarray
        Corresponding time axis array.
    res : int
        Window size for smoothing (number of points).

    Returns
    -------
    smooth_time : list
        Smoothed time axis.
    smooth_data : list
        Smoothed data values.
    """
    smooth_data = []
    smooth_time = []
    for ii in range(0, data.shape[0], 1):
        smooth_data += [np.mean(data[ii:ii+res])]
        smooth_time += [np.mean(time[ii:ii+res])]
    return smooth_time, smooth_data



def haversine(ll1, ll2):
    """
    Calculate the great circle distance between two points on Earth.
    
    Uses the haversine formula to compute the distance between two points
    on a sphere. This is an approximation suitable for short distances on Earth.
    
    Parameters
    ----------
    ll1 : array-like
        First location as [latitude, longitude] in decimal degrees.
    ll2 : array-like
        Second location as [latitude, longitude] in decimal degrees.
    
    Returns
    -------
    float
        Distance between the two points in kilometers.
    """
    r = 6371
    p = np.pi / 180
    a = 0.5 - np.cos((ll2[0]-ll1[0])*p)/2 + np.cos(ll1[0]*p) * np.cos(ll2[0]*p) * (1-np.cos((ll2[1]-ll1[1])*p))/2
    return 2 * r * np.arcsin(np.sqrt(a))
    
def gcd(lat1, lon1, lat2, lon2, radius=6378.137):
    """
    Calculate 2D great circle distance between points on Earth.
    
    Parameters
    ----------
    lat1 : float or array-like
        Latitude(s) of first point(s) in decimal degrees.
    lon1 : float or array-like
        Longitude(s) of first point(s) in decimal degrees.
    lat2 : float or array-like
        Latitude(s) of second point(s) in decimal degrees.
    lon2 : float or array-like
        Longitude(s) of second point(s) in decimal degrees.
    radius : float, optional
        Earth radius in kilometers. Default is 6378.137 km.
    
    Returns
    -------
    float or numpy.ndarray
        Great circle distance(s) in kilometers.
    
    Notes
    -----
    Uses the haversine formula for calculating distances on a sphere.
    Supports vectorized operations for arrays of coordinates.
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist = radius * c
    return dist

def find_nearest2d(da:xr.DataArray, lat, lon, lat_name='lat', lon_name='lon', new_dim='sites', r=1):
    """
    Find nearest valid grid points in 2D xarray DataArray to given coordinates.
    
    Selects the nearest grid point to specified lat/lon coordinates. If the
    nearest point is NaN, searches within a radius r for the closest valid point.
    
    Parameters
    ----------
    da : xr.DataArray
        Input DataArray with latitude and longitude dimensions.
    lat : float or array-like
        Target latitude(s) in decimal degrees.
    lon : float or array-like
        Target longitude(s) in decimal degrees.
    lat_name : str, optional
        Name of latitude dimension in da. Default is 'lat'.
    lon_name : str, optional
        Name of longitude dimension in da. Default is 'lon'.
    new_dim : str, optional
        Name for new dimension when concatenating results for multiple sites.
        Default is 'sites'.
    r : float, optional
        Search radius in degrees when nearest point is invalid. Default is 1.
    
    Returns
    -------
    xr.DataArray
        DataArray values at nearest valid grid points. If multiple lat/lon pairs
        provided, returns concatenated results along new_dim.
    
    Raises
    ------
    ValueError
        If no valid values found within search radius.
    
    Notes
    -----
    Author: Feng Zhu
    
    The function uses great circle distance to find the closest valid grid point
    when the simple nearest neighbor is NaN.
    
    Examples
    --------
    >>> # Extract data at single location
    >>> temp_site = find_nearest2d(temp_data, 52.52, 13.40)
    
    >>> # Extract data at multiple locations
    >>> lats = [40.7, 51.5, 48.8]
    >>> lons = [-74.0, -0.1, 2.3]
    >>> temps_sites = find_nearest2d(temp_data, lats, lons, r=2)
    """
    da_res = da.sel({lat_name: lat, lon_name:lon}, method='nearest')
    if da_res.isnull().any():
        if isinstance(lat, (int, float)): lat = [lat]
        if isinstance(lon, (int, float)): lon = [lon]
        da_res_list = []
        for la, lo in zip(lat, lon):
            mask_lat = (da.lat > la-r)&(da.lat < la+r)
            mask_lon = (da.lon > lo-r)&(da.lon < lo+r)
            # da_sub = da.sel({lat_name: slice(la-r, la+r), lon_name: slice(lo-r, lo+r)})
            da_sub = da.sel({'lat': mask_lat, 'lon': mask_lon})
            dist = gcd(da_sub[lat_name], da_sub[lon_name], la, lo)
            da_sub_valid = da_sub.where(~np.isnan(da_sub), drop=True)
            valid_mask = ~np.isnan(da_sub_valid)
            if valid_mask.sum() == 0:
                print('la:', la)
                print('lo:', lo)
                print('la+r:', la+r)
                print('la-r:', la-r)
                print('lo+r:', lo+r)
                print('lo-r:', lo-r)
                print(da_sub)
                raise ValueError('No valid values found. Please try larger `r` values.')

            dist_min = dist.where(dist == dist.where(~np.isnan(da_sub_valid)).min(), drop=True)
            nearest_lat = dist_min[lat_name].values.item()
            nearest_lon = dist_min[lon_name].values.item()
            da_res = da_sub_valid.sel({lat_name: nearest_lat, lon_name: nearest_lon}, method='nearest')
            da_res_list.append(da_res)
        da_res = xr.concat(da_res_list, dim=new_dim).squeeze()

    return da_res