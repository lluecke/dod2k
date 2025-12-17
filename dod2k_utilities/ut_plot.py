# -*- coding: utf-8 -*-
"""
@author: Lucie Luecke

Plotting functions for displaying data(frames).

"""
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib.gridspec import GridSpec as GS
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

from dod2k_utilities import ut_functions as utf

cols = [ '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#44AA99']

def shade_percentiles(x, y, color, ax, alpha=1, lu=False, zorder=None, lw=1,
                      ups=[60, 70, 80, 90, 95], label=None):
    """
    Shade percentile ranges of an ensemble on a matplotlib axis.
    
    Creates overlapping shaded regions showing different percentile ranges
    of an ensemble, useful for visualizing uncertainty in climate data.
    
    Parameters
    ----------
    x : array-like
        Time or x-axis values (1D array of length n).
    y : array-like
        Ensemble data as m×n array where m is ensemble dimension and n is
        time dimension.
    color : str or tuple
        Color for shading (matplotlib color specification).
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    alpha : float, optional
        Overall transparency multiplier (0-1). Default is 1.
    lu : bool, optional
        If True, plot dotted lines at 5th and 95th percentiles. Default is False.
    zorder : float, optional
        Drawing order for the shaded regions. Default is None.
    lw : float, optional
        Line width for percentile boundary lines if lu=True. Default is 1.
    ups : list of float, optional
        Upper percentiles to shade. Default is [60, 70, 80, 90, 95].
    label : str, optional
        Label for legend (applied to outermost shading). Default is None.
    
    Returns
    -------
    None
    
    Notes
    -----
    Shades symmetric percentile ranges with decreasing opacity:
    - Innermost: 40th-60th percentile (darkest)
    - Outermost: 5th-95th percentile (lightest)
    
    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> x = np.arange(100)
    >>> y = np.random.randn(50, 100)  # 50 ensemble members, 100 time steps
    >>> shade_percentiles(x, y, 'blue', ax, lu=True, label='Ensemble')
    """
    # shades the percentiles of an mxn array (ensemble dimension: m, time dimension: n)
    lows = [100-ii for ii in ups]
    alps = np.array([0.5,0.4, 0.35, 0.3, 0.25])*alpha
    ii=0
    for l, u, a in zip(lows, ups, alps):
        mina = np.nanpercentile(dc(y), l, axis=0)
        maxa = np.nanpercentile(dc(y), u, axis=0)
        X    = dc(np.array(x))
        ll=label if ii==0 else None
        ax.fill_between(X, y1=mina, y2=maxa, color=color, alpha=a, lw=0, 
                        zorder=zorder, label=ll)
        if (l, u) == (5, 95) and lu:
            ax.plot(X, mina, color=color, lw=lw, ls=':', zorder=zorder*2)
            ax.plot(X, maxa, color=color, lw=lw, ls=':', zorder=zorder*2)
        ii+=1
    return

def get_colours(data, colormap='brewer_RdBu_11', minval=False,
                maxval=False, return_mappable=False):
    """
    Generate colors from a colormap based on data values.
    
    Parameters
    ----------
    data : array-like
        Array or list of numerical values to map to colors.
    colormap : str, optional
        Matplotlib colormap name. Default is 'brewer_RdBu_11'.
    minval : float or False, optional
        Minimum value for color normalization. If False, uses min(data).
        Default is False.
    maxval : float or False, optional
        Maximum value for color normalization. If False, uses max(data).
        Default is False.
    return_mappable : bool, optional
        If True, also return ScalarMappable and Normalize objects for colorbar.
        Default is False.
    
    Returns
    -------
    list of tuple
        List of RGBA color tuples, one for each data value, in same order as data.
    
    Examples
    --------
    >>> temps = [15, 20, 25, 30, 35]
    >>> colors = get_colours(temps, colormap='coolwarm')
    >>> # Use colors for scatter plot
    >>> plt.scatter(x, y, c=colors)
    """
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    if not minval:
        minval = np.min(data)
    if not maxval:
        maxval = np.max(data)
    N = len(data)
    cmap         = cm.get_cmap(colormap)
    sm           = cm.ScalarMappable(cmap = colormap)
    sm.set_array(range(N))
    norm         = Normalize(vmin=minval, vmax=maxval)
    rgba         = cmap(norm(data))
    cols         = list(rgba)
    if return_mappable:
        return cols, sm, norm
    return cols


def get_colours2(data, colormap='brewer_RdBu_11', minval=False,
                maxval=False):
    """
    generates colours from a colormap based on the *data* values (array or list)
    returns *cols*: list of colours, in same order as data
    """
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    if not minval:
        minval = np.min(data)
    if not maxval:
        maxval = np.max(data)
    print(minval)
    print(maxval)
    N = len(data)
    cmap         = cm.get_cmap(colormap)
    sm           = cm.ScalarMappable(cmap = colormap)
    sm.set_array(range(N))
    norm         = Normalize(vmin=minval, vmax=maxval)
    rgba         = cmap(norm(data))
    cols         = list(rgba)
    return cols, sm, norm
def plot_resolution(df, title='', mincount=0, col='tab:blue'):
    """
    Plot a histogram of resolutions from a DataFrame.

    This function counts the occurrences of each "resolution" in the DataFrame,
    optionally merges bins with counts below `mincount`, and displays a bar plot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a column named 'resolution', where each entry is 
        a list of integers representing resolution values.
    title : str, optional
        Title of the plot.
    mincount : int, optional
        Minimum count threshold for individual resolution bins. Bins with fewer
        counts than `mincount` are merged into a coarser bin. Default is 0 
        (no merging).

    Returns
    -------
    None
        The function displays a matplotlib bar plot and does not return any value.

    """
    count_res = {}
    for dd in df['resolution'].values:
        if len(dd)>1:
            res = '%d - %d'%(min(dd), max(dd))
        else:
            res='%d'%min(dd)
        if res not in count_res: 
            count_res[res]=0
            
        count_res[res]+=1
    
    if mincount!=0:
        rmv = []
        for kk in list(count_res):
            if count_res[kk]<mincount:
                maxres = float(kk.split(' - ')[-1])
                if maxres<6:
                    maxres=5+np.round(maxres/10.)*10
                    newkey='   <%d'%maxres
                elif maxres<95:
                    maxres=5+np.round(maxres/10.)*10
                    newkey='  <%d'%maxres
                else:
                    maxres=50+np.round(maxres/100.)*100
                    newkey=' <%d'%maxres
                if newkey not in count_res:
                    count_res[newkey]=0
                    print(kk, newkey)
                count_res[newkey]+=1
                rmv+=[kk]
        for kk in rmv: del count_res[kk]
            
    
    plt.figure(dpi=100, figsize=(5,3))
    plt.title(title)
    ax=plt.gca()
    ii=0
    rr=[]
    for kk in np.sort(list(count_res)):
        plt.bar(ii, count_res[kk], color=col)
        ii+=1
        rr+=[kk]
    ax.set_xticks(range(ii))
    ax.set_xticklabels(rr, rotation=45, ha='right', fontsize=7)
    plt.xlabel('resolution')
    plt.ylabel('count')
    plt.show()
    return 
    


def plot_length(df, title='', mincount=0, col='tab:blue'):
    """
    Plot a histogram of lengths from a DataFrame.

    This function bins the 'length' values in the DataFrame into predefined ranges,
    optionally filters bins with counts below `mincount`, and displays a bar plot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a column named 'length' with numeric values.
    title : str, optional
        Title of the plot.
    mincount : int, optional
        Minimum count threshold for bins. Bins with fewer counts than `mincount`
        are excluded from the plot. Default is 0 (all bins shown).

    Returns
    -------
    None
        The function displays a matplotlib bar plot and does not return any value.
    """
    count_res = {'%s-%s'%(ii, ii+50): 0 for ii in range(0, 200, 50) }
    count_res.update({'%s-%s'%(ii, ii+100): 0 for ii in range(200, 800, 100) })
    count_res['>800'] = 0
    for dd in df['length'].values:
        if dd>800:
            count_res['>800']+=1
        for ii in range(0, 200, 50):
            if dd in range(ii, ii+50):
                count_res['%s-%s'%(ii, ii+50)]+=1
        for ii in range(200, 800, 100):
            if dd in range(ii, ii+100):
                count_res['%s-%s'%(ii, ii+100)]+=1
        
        
    
    plt.figure(dpi=100, figsize=(5,3))
    plt.title(title)
    ax=plt.gca()
    ii=0
    rr=[]
    for res, count in count_res.items():
        if count<mincount: continue
        plt.bar(ii, count, color=col)
        ii+=1
        rr+=[res]
    ax.set_xticks(range(ii))
    ax.set_xticklabels(rr, rotation=45, ha='right', fontsize=7)
    plt.xlabel('length')
    plt.ylabel('count')
    plt.show()
    return 

def get_archive_colours(archives_sorted, archive_count, cols= [ '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#44AA99', '#332288']):
    
    archive_colour = {'other': cols[-1]}
    other_archives = []
    major_archives = []
    
    for ii, at in enumerate(archives_sorted):
        print(ii, at, archive_count[at])
        if archive_count[at]>10:
            major_archives     +=[at]
            archive_colour[at] = cols[ii]
        else:
            other_archives     +=[at]
            archive_colour[at] = cols[-1]
    return archive_colour, major_archives, other_archives

def plot_count_proxy_by_archive_short(df, archive_proxy_count, archive_proxy_ticks, archive_colour) :

    fig = plt.figure(figsize=(8, 5), dpi=500)
    ax  = plt.gca()
    count_by_proxy_short   = [archive_proxy_count[tt] for tt in archive_proxy_ticks if archive_proxy_count[tt]>10 ]
    ticks_by_proxy_short   = [tt for tt in archive_proxy_ticks if archive_proxy_count[tt]>10 ]
    cols_by_proxy_short    = [archive_colour[tt.split(':')[0]] for tt in archive_proxy_ticks if archive_proxy_count[tt]>10 ]
    archive_by_proxy_short = [tt.split(':')[0] for tt in archive_proxy_ticks if archive_proxy_count[tt]>10 ]
    
    sort = np.argsort(count_by_proxy_short)[::-1]
    
    # create placeholder artists for legend and clean axis again
    plt.bar(range(len(set(archive_by_proxy_short))), range(len(set(archive_by_proxy_short))), 
            color=[archive_colour[aa] for aa in set(archive_by_proxy_short)],
            label=set(archive_by_proxy_short))
    h, l = ax.get_legend_handles_labels()
    plt.legend()
    ax.cla()
    
    plt.bar(np.arange(len(ticks_by_proxy_short)), np.array(count_by_proxy_short)[sort], 
            color=np.array(cols_by_proxy_short)[sort])
    
    plt.xlabel('proxy type')
    plt.ylabel('count')
    ax.set_xticks(np.arange(len(ticks_by_proxy_short)), 
                  [ticks_by_proxy_short[ii] for ii in sort], 
                  rotation=45, ha='right', fontsize=10)
    plt.legend(h[::-1], l[::-1])
    
    
    fig.tight_layout()
    return fig


def plot_count_proxy_by_archive_all(df, archive_proxy_count, archive_proxy_ticks, archive_colour) :
    fig = plt.figure(figsize=(10, 7), dpi=500)
    ax  = plt.gca()
    count_by_proxy_long   = [archive_proxy_count[tt] for tt in archive_proxy_ticks]
    ticks_by_proxy_long   = [tt for tt in archive_proxy_ticks]
    cols_by_proxy_long    = [archive_colour[tt.split(':')[0]] for tt in archive_proxy_ticks ]
    archive_by_proxy_long = [tt.split(':')[0] for tt in archive_proxy_ticks]
    
    sort = np.argsort(count_by_proxy_long)[::-1]
    
    # create placeholder artists for legend and clean axis again
    plt.bar(range(len(set(archive_by_proxy_long))), range(len(set(archive_by_proxy_long))), 
            color=[archive_colour[aa] for aa in set(archive_by_proxy_long)],
            label=set(archive_by_proxy_long))
    h, l = ax.get_legend_handles_labels()
    plt.legend()
    ax.cla()
    
    plt.bar(np.arange(len(ticks_by_proxy_long)), 
            np.array(count_by_proxy_long)[sort], 
            color=np.array(cols_by_proxy_long)[sort])
    
    plt.xlabel('proxy type')
    plt.ylabel('count')
    ax.set_xticks(np.arange(len(ticks_by_proxy_long)), 
                  [ticks_by_proxy_long[ii] for ii in sort], 
                  rotation=45, ha='right', fontsize=9)
    plt.legend(h[::-1], l[::-1], ncol=2)
    
    
    fig.tight_layout()
    return fig

def plot_geo_archive_proxy_short(df, archives_sorted, archive_proxy_count_short, archive_colour):
    
    proxy_lats = df['geo_meanLat'].values
    proxy_lons = df['geo_meanLon'].values
    
    # plots the map
    fig = plt.figure(figsize=(13, 8), dpi=350)
    grid = GS(1, 3)
    
    ax = plt.subplot(grid[:, :], projection=ccrs.Robinson()) # create axis with Robinson projection of globe
    # ax.stock_img()
    
    
    ax.add_feature(cfeature.LAND, alpha=0.6) # adds land features
    ax.add_feature(cfeature.OCEAN, alpha=0.6, facecolor='#C5DEEA') # adds ocean features
    ax.coastlines() # adds coastline features
    
    ax.set_global()
    
    
    mt = 'ov^s<>pP*XDdh'*10 # generates string of marker types
    
    ijk=0
    for at in archives_sorted:
        print(sorted(archive_proxy_count_short[at]))
        for ii, key in enumerate(sorted(archive_proxy_count_short[at])):
            marker = mt[ii]
            if 'other' not in key: 
                at, pt = key.split(': ')
                at_mask = df['archiveType']==at
                pt_mask = df['paleoData_proxy']==pt
                label = key+' (n=%d)'%archive_proxy_count_short[at][key]
            else:
                at= key.split('other ')[-1]
                exclude_types = [kk.split(': ')[-1] for kk in archive_proxy_count_short[at].keys() if at in kk if 'other' not in kk]
                at_mask = df['archiveType']==at
                pt_mask = ~np.isin(df['paleoData_proxy'], exclude_types)
                label = key+' (n=%d)'%df['paleoData_proxy'][pt_mask&at_mask].count()
                if exclude_types==[]:
                    marker=mt[ijk]
                    ijk+=1
                    label = label.replace('other ','')
            plt.scatter(proxy_lons[pt_mask&at_mask], proxy_lats[pt_mask&at_mask], 
                        transform=ccrs.PlateCarree(), zorder=999,
                        marker=marker, 
                        color=archive_colour[at], 
                        label=label,
                        lw=.3, ec='k', s=150)
        
    plt.legend(bbox_to_anchor=(0.03,-0.01), loc='upper left', ncol=3, fontsize=12, framealpha=0)
    grid.tight_layout(fig)
    return fig


def plot_geo_archive_proxy(df, archive_colour, highlight_archives=[], marker='default', size='default', figsize='default'):

    proxy_lats = df['geo_meanLat'].values
    proxy_lons = df['geo_meanLon'].values
    
    # plots the map
    figsize=(15, 12) if figsize=='default' else figsize
    fig = plt.figure(figsize=figsize, dpi=350)
    grid = GS(1, 3)
    
    ax = plt.subplot(grid[:, :], projection=ccrs.Robinson()) # create axis with Robinson projection of globe
    
    ax.add_feature(cfeature.LAND, alpha=0.5) # adds land features
    ax.add_feature(cfeature.OCEAN, alpha=0.6, facecolor='#C5DEEA') # adds ocean features
    ax.coastlines() # adds coastline features
    
    ax.set_global()
    
    # loop through the data to generate a scatter plot of each data record:
    # 1st loop: go through archive types individually (determines marker type)
    # 2nd loop: through paleo proxy types attributed to the specific archive, which is colour coded
    
    if marker=='default':
        mt = 'ov^s<>pP*XDdh'*10 # generates string of marker types
    else:
        mt = marker

    if size=='default':
        s = 200
    else:
        s = size
    archive_types = np.unique(df['archiveType'])
    
    ijk=0
    for jj, at in enumerate(archive_types):
        arch_mask = df['archiveType']==at
        arch_proxy_types = np.unique(df['paleoData_proxy'][arch_mask])
        for ii, pt in enumerate(arch_proxy_types):
            pt_mask = df['paleoData_proxy']==pt
            at_mask = df['archiveType']==at
            label = at+': '+pt+' ($n=%d$)'% df['paleoData_proxy'][(df['paleoData_proxy']==pt)&(df['archiveType']==at)].count()
            marker = mt[ii] if at in highlight_archives else mt[ijk]
            plt.scatter(proxy_lons[pt_mask&at_mask], proxy_lats[pt_mask&at_mask], 
                        transform=ccrs.PlateCarree(), zorder=999,
                        marker=marker, color=archive_colour[at], 
                        label=label,#.replace('marine sediment:', 'marine sediment:\n'), 
                        lw=.3, ec='k', s=s)
            if at not in highlight_archives: ijk+=1
                
    plt.legend(bbox_to_anchor=(-0.01,-0.01), loc='upper left', ncol=3, fontsize=13.5, framealpha=0)
    grid.tight_layout(fig)
    return fig

def plot_coverage(df, archives_sorted, major_archives, other_archives, archive_colour, all=False, ysc='linear', return_data=False):
    #%% compute the coverage of all records and coverage per archive 
    
    MinY     = np.array([min([float(sy) for sy in yy])  for yy in df['year']]) # find minimum year for each record
    MaxY     = np.array([max([float(sy) for sy in yy])  for yy in df['year']]) # find maximum year for each record
    years    = np.arange(min(MinY), max(MaxY)+1)
    
    # generate array of coverage (how many records are available each year, in total)
    coverage = np.zeros(years.shape[0])
    for ii in range(len(df['year'])):
        coverage[(years>=MinY[ii])&(years<=MaxY[ii])] += 1
    # generate array of coverage for each archive type
    coverage_by_archive = {arch: np.zeros(years.shape[0]) for arch in major_archives+['other'] }
    for arch in archives_sorted:
        arch_mask = df['archiveType']==arch 
        for ii in range(len(df[arch_mask]['year'])):
            if arch not in major_archives: arch='other'
            cc = coverage_by_archive[arch]
            coverage_by_archive[arch][(years>=MinY[arch_mask][ii])&(years<=MaxY[arch_mask][ii])] += 1
            
    fig = plt.figure(figsize=(8, 4), dpi=200)
    ax = plt.gca()
    if all:
        plt.step(years, coverage, color='k', label='all records', lw=3)
    plt.xlabel('year')
    plt.ylabel('total # of records')
    
    plt.xlim(-100, 2020)
    ax.grid(False)
    if np.sum(coverage_by_archive['other'])==0:
        archives = major_archives
    else: archives = major_archives+['other']
    for ii, arch in enumerate(archives):
        plt.step(years, coverage_by_archive[arch], color=archive_colour[arch],
                 label=arch, lw=1.8)
    
    h1, l1 = ax.get_legend_handles_labels()
    if ysc=='log':plt.legend(h1, [ll.replace(' ',' ') for ll in l1], 
                             ncol=4, framealpha=0, bbox_to_anchor=(0,1), loc='lower left' )
    else:plt.legend(h1, l1, ncol=3, framealpha=0)
    plt.ylabel('# of records per archive')
    fig.tight_layout()
    plt.yscale(ysc)
    if return_data:
        return fig, years, coverage, coverage_by_archive
    return fig

def plot_coverage2(df, years, title=''):
    """
    Plot the coverage of records over a range of years.

    This function counts how many records in the DataFrame overlap with each 
    year in the given range and produces a step plot showing total coverage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'year' data for each record. Each row should have
        'miny' and 'maxy' indicating the start and end year of the record.
    years : array-like
        Array of years over which to compute coverage.
    title : str, optional
        Title of the plot. Default is an empty string.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    coverage_filt = np.zeros(years.shape[0])

    miny, maxy = years[0], years[-1]
    for ii in range(len(df['year'])):
        # time_12, int_1, int_2 = np.intersect1d(years, df.iloc[ii].year, return_indices=True)
        coverage_filt[(years>=df.iloc[ii].miny)&(years<=df.iloc[ii].maxy)] += 1
        # coverage_filt[int_1]+=1
    
    fig = plt.figure(figsize=(6, 3), dpi=100)
    plt.title(title)
    ax = plt.gca()
    plt.step(years, coverage_filt, color='k', label='all records', lw=3)
    plt.xlabel('year')
    plt.ylabel('total # of records')
    
    h1, l1 = ax.get_legend_handles_labels()
    plt.legend(h1, l1, ncol=3, framealpha=0)
    plt.ylabel('# of records per archive')
    plt.show()
    return fig

def plot_coverage_analysis(df, years, key, col, title=''):
    """
    Plot the coverage of records over a range of years.

    This function counts how many records in the DataFrame overlap with each 
    year in the given range and produces a step plot showing total coverage.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing 'year' data for each record. Each row should have
        'miny' and 'maxy' indicating the start and end year of the record.
    years : array-like
        Array of years over which to compute coverage.
    title : str, optional
        Title of the plot. Default is an empty string.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    """
    coverage_filt = np.zeros(years.shape[0])

    miny, maxy = years[0], years[-1]
    for ii in range(len(df['year'])):
        # time_12, int_1, int_2 = np.intersect1d(years, df.iloc[ii].year, return_indices=True)
        coverage_filt[(years>=df.iloc[ii].miny)&(years<=df.iloc[ii].maxy)] += 1
        # coverage_filt[int_1]+=1
    
    fig = plt.figure(figsize=(6, 3), dpi=100)
    plt.title(title)
    ax = plt.gca()
    plt.step(years, coverage_filt, color=col, label=key, lw=3)
    plt.xlabel('year')
    plt.ylabel('total # of records')
    
    h1, l1 = ax.get_legend_handles_labels()
    plt.legend(h1, l1, ncol=3, framealpha=0)
    plt.ylabel('# of records per archive')
    plt.show()
    return fig

def plot_PCs(years_hom, eigenvectors, paleoData_zscores_hom, title='', name='', col='tab:blue'):
    """
    Plot principal components and reconstructed time series.

    Parameters
    ----------
    years_hom : numpy.ndarray
        Homogenised time axis.
    eigenvectors : numpy.ndarray
        Eigenvectors from PCA.
    paleoData_zscores_hom : numpy.ma.MaskedArray
        Homogenised z-score data array of shape (n_records, n_years).
    title : str, optional
        Title for plots.
    name : str, optional
        Name suffix for saving figures.

    Returns
    -------
    PCs : numpy.ndarray
        Principal component time series.
    eigenvectors : numpy.ndarray
        Eigenvectors (EOF loadings) corresponding to PCs.
    """
    PCs = np.dot(eigenvectors.T, paleoData_zscores_hom.data)

    Dz   = paleoData_zscores_hom.data
    Dzr  = np.ma.masked_array(np.dot(eigenvectors, PCs), mask=paleoData_zscores_hom.mask)
    

    fig = plt.figure()
    plt.suptitle(title)
    ax = plt.subplot(311)
    
    plt.plot(years_hom, np.ma.mean(paleoData_zscores_hom, axis=0), #color='k', 
             zorder=999, color=col)
    
    ax.axes.xaxis.set_ticklabels([])
    plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
    plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
    plt.xlim(years_hom[0]-20, years_hom[-1]+20)
    plt.ylabel('paleoData_zscores')
    for ii in range(2):
        ax = plt.subplot(311+ii+1)
        plt.plot(years_hom, PCs[ii], color=col)
        if ii==1: plt.xlabel('time (year CE)')
        plt.ylabel('PC %d'%(ii+1))
        plt.axhline(0, color='k', alpha=0.5, lw=0.5)
        plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
        plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
        plt.xlim(years_hom[0]-20, years_hom[-1]+20)
        if ii==0: ax.axes.xaxis.set_ticklabels([])

    utf.save_fig(fig, 'PCs_%s'%title, dir=name)

    plt.figure()
    for ii in range(paleoData_zscores_hom.shape[0]):
        plt.plot(paleoData_zscores_hom[ii,:], Dzr[ii,:],  alpha=0.4, lw=1, color=col)
    plt.xlabel('paleoData_zscores')
    plt.ylabel('paleoData_zscores_reconstructed')

    
    fig = plt.figure()
    plt.suptitle(title)
    ax = plt.subplot(211)
    for ii in range(paleoData_zscores_hom.shape[0]):
        plt.plot(years_hom, paleoData_zscores_hom[ii,:], color=col, alpha=0.4, lw=1)
    plt.plot(years_hom, np.ma.mean(paleoData_zscores_hom, axis=0), color='k', zorder=999)
    
    ax.axes.xaxis.set_ticklabels([])
    plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
    plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
    plt.xlim(years_hom[0]-20, years_hom[-1]+20)
    plt.ylabel('paleoData_zscores')
    
    ax = plt.subplot(212)
    for ii in range(Dzr.shape[0]):
        plt.plot(years_hom, Dzr[ii,:], color=col, alpha=0.4, lw=1)
    plt.plot(years_hom, np.ma.mean(Dzr, axis=0), color='k', zorder=999)
    
    ax.axes.xaxis.set_ticklabels([])
    plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
    plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
    plt.xlim(years_hom[0]-20, years_hom[-1]+20)
    plt.ylabel('paleoData_zscores \n (reconstructed)')
    

    n_recs = paleoData_zscores_hom.data.shape[0]
    fig = plt.figure()
    plt.suptitle(title)
    for ii in range(2):
        plt.subplot(211+ii)
        plt.plot(range(n_recs), eigenvectors[ii], color=col)
        if ii==1: plt.xlabel('rec')
        plt.ylabel('EOF %d load'%(ii+1))
        plt.axhline(0, color='k', alpha=0.5, lw=0.5)
        
    utf.save_fig(fig, 'EOFloading_%s'%title, dir=name)
    
    return PCs, eigenvectors

def geo_plot(df, fs=(9,4.5), dpi=350, return_col=False,  **kwargs):
    """
    Plot the spatial distribution of paleo-proxy records on a global map.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the records. Must include columns:
        'geo_meanLat', 'geo_meanLon', 'archiveType', 'paleoData_proxy', 'datasetId'.
    fs : tuple, optional
        Figure size (width, height) in inches. Default is (9, 4.5).
    dpi : int, optional
        Figure resolution in dots per inch. Default is 350.
    **kwargs : dict
        Optional keyword arguments. Supported keys:
        - 'mark_records': dict, to highlight specific datasets on the map.
        - 'mark_archives': list of archive keys to mark.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object containing the map.
    """
    archive_colour, archives_sorted, proxy_marker = df_colours_markers()
    
    #%% plot the spatial distribution of all records
    proxy_lats = df['geo_meanLat'].values
    proxy_lons = df['geo_meanLon'].values
    
    # plots the map
    fig = plt.figure(figsize=fs, dpi=dpi) #fs=(13,8), dpi=350
    grid = GS(1, 3)
    
    ax = plt.subplot(grid[:, :], projection=ccrs.Robinson()) # create axis with Robinson projection of globe


    ax.add_feature(cfeature.LAND, alpha=0.5) # adds land features
    ax.add_feature(cfeature.OCEAN, alpha=0.3, facecolor='#C5DEEA') # adds ocean features
    ax.coastlines() # adds coastline features
    
    ax.set_global()
    
    
    mt = 'ov^s<>pP*XDdh'*10 # generates string of marker types
    
    ijk=0
    h1, l1 = [], []
    h2, l2 = [], []
    for at in archives_sorted:
        at_mask = df['archiveType']==at
        for ii, pt in enumerate(set(df[at_mask]['paleoData_proxy'])):
            marker  = mt[ii]
            pt_mask = df['paleoData_proxy']==pt
            label   = at+': '+pt+' (n=%d)'%len(df[at_mask&pt_mask])
            plt.scatter(proxy_lons[pt_mask&at_mask], proxy_lats[pt_mask&at_mask], 
                        transform=ccrs.PlateCarree(), zorder=999,
                        marker=proxy_marker[at][pt], 
                        color=archive_colour[at], 
                        label=label,
                        lw=.3, ec='k', s=200)
            
            if kwargs and 'mark_records' in kwargs:
                hh, ll = ax.get_legend_handles_labels()
                key = '%s_%s'%(at, pt) if at!='lake sediment' else 'lake sediment_d18O+d2H'
                if key in kwargs['mark_archives']:
                    id_mask = np.isin(df['datasetId'], kwargs['mark_records'][key])
                    label='included in PCA'
                    # if label in ll:
                    #     label = None
                    plt.scatter(proxy_lons[pt_mask&at_mask&id_mask], proxy_lats[pt_mask&at_mask&id_mask], 
                                transform=ccrs.PlateCarree(), zorder=999,
                                marker=proxy_marker[at][pt], #label=label,
                                lw=2, ec='k', color=archive_colour[at], s=200)
                    
        
    hh, ll = ax.get_legend_handles_labels()
    plt.legend(hh, ll, bbox_to_anchor=(0.03,-0.01), loc='upper left', ncol=3, fontsize=12, framealpha=0)
    grid.tight_layout(fig)

    if return_col:
        return fig, archive_colour
    
    return fig


def df_colours_markers(db_name='dod2k_v2.0'):
    """
    Generate archive colours and proxy markers for plotting functions.

    Parameters
    ----------
    db_name : str, optional
        Name of the database CSV file to load. Default is 'dod2k_dupfree_dupfree'.

    Returns
    -------
    archive_colour : dict
        Dictionary mapping archive types to color codes.
    archives_sorted : numpy.ndarray
        Sorted list of archive types based on record count.
    proxy_marker : dict
        Dictionary mapping each archive type and proxy to a specific marker.
    """
    cols = [ '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#44AA99']

    df = utf.load_compact_dataframe_from_csv(db_name)

    
    # count archive types
    archive_count = {}
    for ii, at in enumerate(set(df['archiveType'])):
        archive_count[at] = df.loc[df['archiveType']==at, 'paleoData_proxy'].count()
        
    archive_colour = {'other': cols[-1]}
    proxy_marker   = {}
    other_archives = []
    major_archives = []

    
    mt = 'ov^s<>pP*XDdh'*10 # generates string of marker types

    ijk=0
    sort = np.argsort([cc for cc in archive_count.values()])
    archives_sorted = np.array([cc for cc in archive_count.keys()])[sort][::-1]
    for ii, at in enumerate(archives_sorted):
        print(ii, at, archive_count[at])
        if archive_count[at]>10:
            archive_colour[at] = cols[ii]
            major_archives+=[at]
        else:
            archive_colour[at] = cols[-1]
            other_archives+=[at]
        arch_mask = df['archiveType']==at
        arch_proxy_types = np.unique(df['paleoData_proxy'][arch_mask])
        proxy_marker[at]={}
        for jj, pt in enumerate(arch_proxy_types):
            marker = mt[jj] if at in major_archives else mt[ijk]
            proxy_marker[at][pt]=marker
        if at not in major_archives: ijk+=1
            
    return archive_colour, archives_sorted, proxy_marker 


def geo_EOF_plot(df, pca_rec, EOFs, keys, fs=(13,8), dpi=350, barlabel='EOF', which_EOF=0):
    """
    Plot geographic distribution of records colored by EOF loadings.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with paleo-proxy records. Must include columns
        'geo_meanLat', 'geo_meanLon', 'datasetId'.
    pca_rec : dict
        Dictionary mapping keys to lists of dataset IDs included in PCA.
    EOFs : dict
        Dictionary mapping keys to EOF arrays. 
    keys : list
        List of keys (record types) to plot.
    fs : tuple, optional
        Figure size in inches. Default is (13, 8).
    dpi : int, optional
        Figure resolution in dots per inch. Default is 350.
    barlabel : str, optional
        Label for the colorbar. Default is 'EOF'.
    which_EOF : int, optional
        Index of the EOF to plot. Default is 0 (first EOF).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure object containing the EOF-colored map.
    """
    #%% plot the spatial distribution of all records
    proxy_lats = df['geo_meanLat'].values
    proxy_lons = df['geo_meanLon'].values
    
    # plots the map
    fig = plt.figure(figsize=fs, dpi=dpi) #fs=(13,8), dpi=350
    grid = GS(1, 3)
    
    ax = plt.subplot(grid[:, :], projection=ccrs.Robinson()) # create axis with Robinson projection of globe
    
    # ax.stock_img(clip_on=False)
    
    ax.add_feature(cfeature.LAND, alpha=0.6) # adds land features
    ax.add_feature(cfeature.OCEAN, alpha=0.6, facecolor='#C5DEEA') # adds ocean features
    ax.coastlines() # adds coastline features
    
    ax.set_global()
    
    mt = 'v^soD<''osD>pP*Xdh' # generates string of marker types

    # some of the following lines are hard-coded to plot EOF1, 
    # but asking for EOFs[key][0] here and also in f.get_colours will give the plot of EOF2
    # also need to modify the colorscale label cax.set_ylabel('EOF 2')
    
    # if we are multipling the PCs x -1, multiply the EOF loadings by -1 as well
    a= {}
    label={}
    for key in keys:
        if key in ['tree_d18O', 'coral_d18O']:# multiply EOF sign by -1
            a[key] = -1
            label[key] = key+' ($\\ast(-1)$)'
        else:
            a[key] = 1
            label[key]=key
    print(a)
    
    all_EOFs = [a[key]*EOFs[key][which_EOF][ii]  for key in keys for ii in range(len(EOFs[key][which_EOF]))]
   
    colors, sm, norm = get_colours2(all_EOFs, 
                                colormap='RdBu_r',minval=-0.6,maxval=0.6)
    
    ijk=0

    

    for key in keys:
        
        marker  = mt[ijk]

        colors = get_colours(a[key]*EOFs[key][which_EOF], colormap='RdBu_r',minval=-0.6,maxval=0.6)
        id_mask = np.isin(df['datasetId'], pca_rec[key]) 
        for jj in range(len(pca_rec[key])):
            
            scat_label   = label[key]+' (n=%d)'%len(pca_rec[key]) if jj==0 else None
            
            plt.scatter(proxy_lons[id_mask][jj], proxy_lats[id_mask][jj], 
                        transform=ccrs.PlateCarree(), zorder=999,
                        marker=marker, 
                        color=colors[jj], 
                        label=None,
                        lw=.3, ec='k', s=200)
            plt.scatter(proxy_lons[id_mask][jj], proxy_lats[id_mask][jj], 
                        transform=ccrs.PlateCarree(), zorder=999,
                        marker=marker, 
                        color='none', 
                        label=scat_label, 
                        lw=1, ec='k', s=200)
        ijk+=1
    
    cax=ax.inset_axes([1.02, 0.1, 0.035, 0.8])
    sm.set_array([])
    
    matplotlib.colorbar.ColorbarBase(cax, cmap='RdBu_r', norm=norm)
    cax.set_ylabel(barlabel, fontsize=13.5)
    
    plt.legend(bbox_to_anchor=(0.03,-0.01), loc='upper left', ncol=3, fontsize=13.5, framealpha=0)
    grid.tight_layout(fig)

    return fig


