"""
This script includes functions which search for duplicates. 
Update 22/10/24 updated duplicate_decisions: 
   - created backup decision file which is intermediately saved
   - outputs URL which can be copied and pasted into browser
   - implemented a composite option in the decision process, to create a composite of two records
   - removed date (YY-MM-DD) of decision output filename
Update 8/10/24 changed colours of figure in dup_plot (dropped highlighting differing metadata). 
                replaced db_name with df.name


Update 2/10/24 Implemented a commenting option in duplicate_decisions to comment on decision process and on individual decisions.

Update 27/9/24 Updated directory names and changed the correlation and distances output in find_duplicates to only output data from potential duplicates (replaced saving all pairs)

Update 9/9/24 Introduced timestamps and contact details into duplicate decision output csv and changed the file= and dirnames for streamlining purposes.
Update 23/8/24 Replaced the function cleanup_database and split into two: 
plot_duplicates: plots the candidate pairs, saves figures and a summary csv sheet.
cleanup_database_2: goes through the candidate pairs and makes decisions based on the options: a) raw input b) keep all records c) automatically keep only updated records and eliminate the other candidate. Decisions and metadata are saved in csv file.

Update 22/8/24: Fixed a bug in find_duplicates logical algorithm- wrong bracket closure. Also changed location_crit to account for nans in elevation. Calculation of z-scores only divided by std if std!=0 to avoid nans.

Update 15/8/24: updated numerical checks for duplicate detection: implemented check for correlation and rmse of records plus correlation and rmse of first difference. 

Update 13/8/24: updated keys to account for updated dataframe terminology. Revised loading/saving of data in find_duplicates
   find_duplicates:
        updated the logic for duplicate detection:
            overlap_crit now accounts for short records (allows short records to pass through without overlap check)
            corr_crit only one numerical criterion needs to be satisfied for the data (either correlation or rmse or 1st difference)
            location_crit now includes elevation too

   cleanup_database:
       updated plot to include URL, Database and streamlined table
Script written by Lucie Luecke, 6/7/23
"""

#==============================================================================

import numpy as np
import datetime
import matplotlib.pyplot as plt
import geopy.distance
import pandas as pd
import os
from matplotlib.gridspec import GridSpec as GS
import csv
import sys
from scipy.stats import gaussian_kde

if str(os.getcwd())+'/dod2k_utilities' not in sys.path:
    sys.path.insert(0, str(os.getcwd())+'/dod2k_utilities')
    
from ut_functions import write_csv, read_csv, save_fig


def find_duplicates_optimized(df, dist_tolerance_km=8, n_points_thresh=10, 
                    corr_thresh=0.9, rmse_thresh=0.1, corr_diff_thresh=0.9,
                    rmse_diff_thresh=0.1, elev_tolerance=0, ignore_same_database=False, 
                    save=True, print_output=False, return_data=False
                   ):
    """
    Identify potential duplicate records in a dataset based on metadata and time series similarity.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing proxy records with metadata. Expected columns include:
        ['geo_meanLat', 'geo_meanLon', 'geo_meanElev', 'year', 'paleoData_values', 
         'archiveType', 'paleoData_proxy', 'geo_siteName', 'originalDatabase', 
         'originalDataURL', 'datasetId'].
    dist_tolerance_km : float, optional
        Maximum allowed geographical distance between duplicates (km). Default is 8.
    n_points_thresh : int, optional
        Minimum number of overlapping time points required. Default is 10.
    corr_thresh : float, optional
        Minimum correlation threshold for duplicate detection. Default is 0.9.
    rmse_thresh : float, optional
        Maximum RMSE allowed for duplicate detection. Default is 0.1.
    corr_diff_thresh : float, optional
        Minimum correlation of first differences threshold. Default is 0.9.
    rmse_diff_thresh : float, optional
        Maximum RMSE of first differences allowed. Default is 0.1.
    elev_tolerance : float, optional
        Maximum allowed elevation difference. Default is 0.
    ignore_same_database : bool, optional
        If True, ignores potential duplicates within the same database. Default is False.
    save : bool, optional
        If True, saves results as a CSV file. Default is True.
    print_output : bool, optional
        If True, prints progress and matching information. Default is False.

    Returns
    -------
    pot_dup_inds : list of list of int
        Indices of detected potential duplicates.
    pot_dup_IDs : list of list of str
        Dataset IDs of detected potential duplicates.
    distances_km : numpy.ndarray
        Matrix of pairwise distances between records.
    correlations : numpy.ndarray
        Matrix of pairwise correlations between records.
    """
    n_proxies = len(df)
    print(df.name)
    
    #database_name = '_'.join(df.originalDatabase.unique())
    
    # Loop through the proxies
    distances_km    = np.zeros((n_proxies,n_proxies))
    correlations    = np.zeros((n_proxies,n_proxies))
    rmse            = np.zeros((n_proxies,n_proxies))
    corr_diff       = np.zeros((n_proxies,n_proxies))
    rmse_diff       = np.zeros((n_proxies,n_proxies))
    
    n_pot_dups     = 0 # counts number of detected duplicates
    pot_dup_inds   = [] # stores potential duplicate indices 
    pot_dup_IDs    = [] # stores potential duplicate IDs
    pot_dup_corrs  = [] # stores potential duplicate correlations 
    pot_dup_dists  = [] # stores potential duplicate distances
    pot_dup_meta   = [['index 1', 'index 2', 'ID 1', 'ID 2', 'correlation', 'distance (km)']]
    
    print('Start duplicate search:')
    print('='*33)
    print('checking parameters:')
    print('%-16s'%'proxy archive                ', ' : %-16s'%' must match')
    print('%-16s'%'proxy type                   ', ' : %-16s'%' must match')
    print('%-16s'%'distance (km)                ', ' < %-16s'%str(dist_tolerance_km)) 
    print('%-16s'%'elevation                    ', ' : %-16s'%' must match')
    print('%-16s'%'time overlap                 ', ' > %-16s'%str(n_points_thresh))
    print('%-16s'%'correlation                  ', ' > %-16s'%str(corr_thresh))
    print('%-16s'%'RMSE                         ', ' < %-16s'%str(rmse_thresh))
    print('%-16s'%'1st difference rmse          ', ' < %-16s'%str(rmse_diff_thresh))
    print('%-16s'%'correlation of 1st difference', ' > %-16s'%str(corr_diff_thresh))
    print('='*33)

    ddir = 'data/%s/dup_detection/'%(df.name)
    fn   = 'dup_detection_candidates_'+df.name
    

    print('Start duplicate search')
    # otherwise re-generate from scratch
    for ii in range(n_proxies):
        if ii in range(0, 10000, 10):
            print('Progress: '+str(ii)+'/'+str(n_proxies))
    
        # Location of proxy 1
        lat_1   = df['geo_meanLat'].iloc[ii]
        lon_1   = df['geo_meanLon'].iloc[ii]
        # time and data of proxy 1
        time_1_ma = df['year'].iloc[ii]
        data_1_ma = np.array(df['paleoData_values'].iloc[ii])
        if type(time_1_ma)==np.ma.core.MaskedArray:
            time_1  = time_1_ma.data[~data_1_ma.mask]
            data_1  = data_1_ma.data[~data_1_ma.mask]
        else:
            time_1  = time_1_ma
            data_1  = data_1_ma
        # archive and proxy type of proxy 1
        arch_1  = df['archiveType'].iloc[ii].lower()
        type_1  = df['paleoData_proxy'].iloc[ii].lower()
        site_1  = df['geo_siteName'].iloc[ii].lower()
        db_1    = df['originalDatabase'].iloc[ii].lower()
        url_1   = df['originalDataURL'].iloc[ii].lower()
        id_1    = df['datasetId'].iloc[ii].lower()
        #
        for jj in range(ii+1, n_proxies):
            db_2 = df['originalDatabase'].iloc[jj].lower()
            id_2    = df['datasetId'].iloc[jj].lower()
            
            if (db_1==db_2) & ignore_same_database: continue
            # Location of proxy 2
            lat_2 = df['geo_meanLat'].iloc[jj]
            lon_2 = df['geo_meanLon'].iloc[jj]
            # time and data of proxy 2
            time_2_ma = df['year'].iloc[jj]
            data_2_ma = np.array(df['paleoData_values'].iloc[jj])
            
            
            if type(time_2_ma)==np.ma.core.MaskedArray:
                time_2  = time_2_ma.data[~data_2_ma.mask]
                data_2  = data_2_ma.data[~data_2_ma.mask]
            else:
                time_2  = time_2_ma
                data_2  = data_2_ma
                
            # archive and proxy type of proxy 2
            arch_2  = df['archiveType'].iloc[jj].lower()
            type_2  = df['paleoData_proxy'].iloc[jj].lower()
            site_2  = df['geo_siteName'].iloc[jj].lower()
            db_2    = df['originalDatabase'].iloc[jj].lower()
            url_2   = df['originalDataURL'].iloc[jj].lower()

            ### Criterion on matching metadata
            meta_crit         = (arch_1 == arch_2) & (type_1 == type_2) # archive types and proxy types must agree
            if not meta_crit: continue

            ### Criterion on elevation
            elevation_not_nan = ((~np.isnan(df['geo_meanElev'].iloc[ii]))& ~np.isnan(df['geo_meanElev'].iloc[jj]))
            elevation_dist    = np.abs(df['geo_meanElev'].iloc[ii]-df['geo_meanElev'].iloc[jj])
            if not elevation_not_nan and elevation_dist>elev_tolerance: continue

            ### Criterion on time overlap
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) # possibly need to round these time series?
            if time_12.shape[0]<=n_points_thresh: continue
            
            ### Criterion on distance
            # Calculate distance between proxy 1 and proxy 2 in km
            distances_km[ii, jj] = geopy.distance.great_circle((lat_1,lon_1), (lat_2,lon_2)).km
            if distances_km[ii,jj] > dist_tolerance_km: continue
            
            # calculate correlation, rms, sum of differences between shared i and j data
            z_1  = (data_1[int_1][:-1]-data_1[int_1][1:])
            z_1 -= np.mean(z_1)        
            if np.std(z_1)!=0: 
                z_1 /= np.std(z_1)
            # else:
            #     plt.figure()
            #     plt.scatter(time_1, data_1, label='data_1 (%s)'%id_1)
            #     plt.scatter(time_12, data_1[int_1], label='data_1 SHARED TIME (%s)'%id_1)
            #     plt.scatter(time_2, data_2, label='data_2 (%s)'%id_2)
            #     plt.scatter(time_12, data_2[int_2], label='data_2 SHARED TIME (%s)'%id_2)
                
            #     plt.legend()
            #     plt.show()
            
            
            z_2  = (data_2[int_2][:-1]-data_2[int_2][1:])
            z_2 -= np.mean(z_2)
            if np.std(z_2)!=0: 
                z_2 /= np.std(z_2)
            # else:
                
            #     plt.figure()
                
            #     plt.scatter(time_12, data_1[int_1]-np.mean(data_1[int_1]), label='data_1 SHARED TIME (%s)'%id_1)
                
            #     plt.scatter(time_12, data_2[int_2]-np.mean(data_2[int_2]), label='data_2 SHARED TIME (%s)'%id_2)
                
            #     plt.legend()
            #     plt.show()
                
            
            correlations[ii, jj] = np.corrcoef(np.vstack([data_1[int_1], data_2[int_2]]))[0,1]
            
            rmse[ii, jj]         = np.sqrt(np.sum((data_1[int_1]-data_2[int_2])**2)/len(time_12))

            
            rmse_diff[ii, jj]    = np.sqrt(np.sum((z_1-z_2)**2)/len(time_12))
            corr_diff[ii, jj]    = np.corrcoef(np.vstack([z_1, z_2]))[0,1]

            
            
            if ((np.isnan(correlations[ii,jj]) & np.isnan(rmse[ii,jj]))|(np.isnan(corr_diff[ii,jj]) & np.isnan(rmse_diff[ii,jj]))):
                print('!!! %d|%d nan detected in both correlation and rmse of record data (or its 1st difference). Danger of missing duplicate!!'%(ii, jj))
                for idd, dd in enumerate([correlations[ii,jj], rmse[ii,jj], rmse_diff[ii,jj],  corr_diff[ii,jj]]):
                    if np.isnan(dd):
                        print('nan detected in %s'%(['correlation','RMSE','RMSE of 1st difference', 'correlation of 1st difference'][idd]))
            
            # DEFINE CRITERIA:

            
    # i,j are located within dist_tolerance_km of each others and have the same elevation (provided they're not nans)
            overlap_crit  = ((len(time_12) > n_points_thresh) |
                             (len(time_1) <= n_points_thresh) |
                             (len(time_2)  <= n_points_thresh) )  # we have a sufficient time overlap between the datasets 
            site_crit     = np.any([s1 in s2 for s2 in site_2.split(' ') for s1 in site_1.split(' ')])| np.any([s1 in s2 for s2 in site_2.split('-') for s1 in site_1.split('-')])| np.any([s1 in s2 for s2 in site_2.split('_') for s1 in site_1.split('_')])|np.any([site_1 in site_2, site_2 in site_1]) # there is at least some overlap in the site name 
            corr_crit     = (((correlations[ii, jj] > corr_thresh) | 
                             (corr_diff[ii, jj] > corr_diff_thresh)) # correlation between shared datapoints exceeds correlation threshol dor correlation of first difference above threshold
                             & 
                             ((rmse[ii, jj] < rmse_thresh) | 
                             (rmse_diff[ii, jj] < rmse_diff_thresh))) #  RMSE of records or of first difference below threshold
            url_crit       = (url_1==url_2 if db_1==db_2 else True)
            if print_output:
                print('--------')
                print(id_1, id_2)
                print('archive and proxy match: %s'%meta_crit)
                print('site match: %s'%site_crit)
                # print('location match: %s'%location_crit)
                print('correlation high: %s'%corr_crit)
                print('overlap %s'%overlap_crit)
            if (meta_crit & site_crit #& url_crit
                # & location_crit
                & corr_crit
                & overlap_crit)|((correlations[ii, jj] > 0.98)& overlap_crit):
                # if these criteria are satisfied, we found a possible duplicate!
                n_pot_dups    += 1
                pot_dup_inds  += [[ii, jj]]
                pot_dup_IDs   += [df['datasetId'].iloc[[ii, jj]].values]
                pot_dup_corrs += [[correlations[ii, jj]]]
                pot_dup_dists += [[distances_km[ii,jj]]]
                pot_dup_meta  += [[ii, jj, df['datasetId'].iloc[ii], df['datasetId'].iloc[jj], correlations[ii, jj], distances_km[ii,jj]]]
                print('--> Found potential duplicate: %d: %s&%d: %s (n_potential_duplicates=%d)'%(ii, id_1, jj, id_2, n_pot_dups))
                
    if save: 
        os.makedirs(ddir, exist_ok=True)
        write_csv(pot_dup_meta, ddir+fn)
        
        print('='*60)
        print('Saved indices, IDs, distances, correlations in %s'%ddir)
    print('='*60)
    print('Detected %d possible duplicates in %s.'%(n_pot_dups, df.name))
    print('='*60)
    if len(pot_dup_inds)<=10:
        print('='*60)
        print('Indices: %s'%(', ').join([str(pdi) for pdi in pot_dup_inds]))
        print('IDs: %s'%(', ').join([pdi[0]+' + '+pdi[1] for pdi in pot_dup_IDs]))
        print('='*60)
    if return_data:
        return pot_dup_IDs
    return #pot_dup_inds, pot_dup_IDs, distances_km, correlations

def find_duplicates(df, dist_tolerance_km=8, n_points_thresh=10, 
                    corr_thresh=0.9, rmse_thresh=0.1, corr_diff_thresh=0.9,
                    rmse_diff_thresh=0.1, elev_tolerance=0, ignore_same_database=False, 
                    save=True, print_output=False
                   ):
    """
    Identify potential duplicate records in a dataset using metadata and time series similarity.

    This is a simpler version of `find_duplicates_optimized`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing proxy records with metadata. Expected columns include:
        ['geo_meanLat', 'geo_meanLon', 'geo_meanElev', 'year', 'paleoData_values', 
         'archiveType', 'paleoData_proxy', 'geo_siteName', 'originalDatabase', 
         'originalDataURL', 'datasetId'].
    dist_tolerance_km : float, optional
        Maximum allowed geographical distance between duplicates (km). Default is 8.
    n_points_thresh : int, optional
        Minimum number of overlapping time points required. Default is 10.
    corr_thresh : float, optional
        Minimum correlation threshold for duplicate detection. Default is 0.9.
    rmse_thresh : float, optional
        Maximum RMSE allowed for duplicate detection. Default is 0.1.
    corr_diff_thresh : float, optional
        Minimum correlation of first differences threshold. Default is 0.9.
    rmse_diff_thresh : float, optional
        Maximum RMSE of first differences allowed. Default is 0.1.
    elev_tolerance : float, optional
        Maximum allowed elevation difference. Default is 0.
    ignore_same_database : bool, optional
        If True, ignores potential duplicates within the same database. Default is False.
    save : bool, optional
        If True, saves results as a CSV file. Default is True.
    print_output : bool, optional
        If True, prints progress and matching information. Default is False.

    Returns
    -------
    pot_dup_inds : list of list of int
        Indices of detected potential duplicates.
    pot_dup_IDs : list of list of str
        Dataset IDs of detected potential duplicates.
    distances_km : numpy.ndarray
        Matrix of pairwise distances between records.
    correlations : numpy.ndarray
        Matrix of pairwise correlations between records.
    """
    n_proxies = len(df)
    print(df.name)
    
    #database_name = '_'.join(df.originalDatabase.unique())
    
    # Loop through the proxies
    distances_km    = np.zeros((n_proxies,n_proxies))
    correlations    = np.zeros((n_proxies,n_proxies))
    rmse            = np.zeros((n_proxies,n_proxies))
    corr_diff       = np.zeros((n_proxies,n_proxies))
    rmse_diff       = np.zeros((n_proxies,n_proxies))
    
    n_pot_dups     = 0 # counts number of detected duplicates
    pot_dup_inds   = [] # stores potential duplicate indices 
    pot_dup_IDs    = [] # stores potential duplicate IDs
    pot_dup_corrs  = [] # stores potential duplicate correlations 
    pot_dup_dists  = [] # stores potential duplicate distances
    pot_dup_meta   = [['index 1', 'index 2', 'ID 1', 'ID 2', 'correlation', 'distance (km)']]
    
    print('Start duplicate search:')
    print('='*33)
    print('checking parameters:')
    print('%-16s'%'proxy archive                ', ' : %-16s'%' must match')
    print('%-16s'%'proxy type                   ', ' : %-16s'%' must match')
    print('%-16s'%'distance (km)                ', ' < %-16s'%str(dist_tolerance_km)) 
    print('%-16s'%'elevation                    ', ' : %-16s'%' must match')
    print('%-16s'%'time overlap                 ', ' > %-16s'%str(n_points_thresh))
    print('%-16s'%'correlation                  ', ' > %-16s'%str(corr_thresh))
    print('%-16s'%'RMSE                         ', ' < %-16s'%str(rmse_thresh))
    print('%-16s'%'1st difference rmse          ', ' < %-16s'%str(rmse_diff_thresh))
    print('%-16s'%'correlation of 1st difference', ' > %-16s'%str(corr_diff_thresh))
    print('='*33)

    ddir = 'data/%s/dup_detection/'%(df.name)
    fn   = 'dup_detection_candidates_'+df.name
    # fn   = 'pot_dup_%s_'+df.name

    print('Start duplicate search')
    # otherwise re-generate from scratch
    for ii in range(n_proxies):
        if ii in range(0, 10000, 10):
            print('Progress: '+str(ii)+'/'+str(n_proxies))
    
        # Location of proxy 1
        lat_1   = df['geo_meanLat'].iloc[ii]
        lon_1   = df['geo_meanLon'].iloc[ii]
        # time and data of proxy 1
        time_1_ma = df['year'].iloc[ii]
        data_1_ma = np.array(df['paleoData_values'].iloc[ii])
        if type(time_1_ma)==np.ma.core.MaskedArray:
            time_1  = time_1_ma.data[~data_1_ma.mask]
            data_1  = data_1_ma.data[~data_1_ma.mask]
        else:
            time_1  = time_1_ma
            data_1  = data_1_ma
        # archive and proxy type of proxy 1
        arch_1  = df['archiveType'].iloc[ii].lower()
        type_1  = df['paleoData_proxy'].iloc[ii].lower()
        site_1  = df['geo_siteName'].iloc[ii].lower()
        db_1    = df['originalDatabase'].iloc[ii].lower()
        url_1   = df['originalDataURL'].iloc[ii].lower()
        id_1    = df['datasetId'].iloc[ii].lower()
        #
        for jj in range(ii+1, n_proxies):
            db_2 = df['originalDatabase'].iloc[jj].lower()
            id_2    = df['datasetId'].iloc[jj].lower()
            
            if (db_1==db_2) & ignore_same_database: continue
            # Location of proxy 2
            lat_2 = df['geo_meanLat'].iloc[jj]
            lon_2 = df['geo_meanLon'].iloc[jj]
            # time and data of proxy 2
            time_2_ma = df['year'].iloc[jj]
            data_2_ma = np.array(df['paleoData_values'].iloc[jj])

            # print(jj, db_2, type(time_2_ma), type(data_2_ma))
            
            if type(time_2_ma)==np.ma.core.MaskedArray:
                time_2  = time_2_ma.data[~data_2_ma.mask]
                data_2  = data_2_ma.data[~data_2_ma.mask]
            else:
                time_2  = time_2_ma
                data_2  = data_2_ma
                
            # archive and proxy type of proxy 2
            arch_2  = df['archiveType'].iloc[jj].lower()
            type_2  = df['paleoData_proxy'].iloc[jj].lower()
            site_2  = df['geo_siteName'].iloc[jj].lower()
            db_2    = df['originalDatabase'].iloc[jj].lower()
            url_2   = df['originalDataURL'].iloc[jj].lower()
            
            # Calculate distance between proxy 1 and proxy 2 in km
            distances_km[ii, jj] = geopy.distance.great_circle((lat_1,lon_1),
                                                               (lat_2,lon_2)).km
            # find shared time values 
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) # possibly need to round these time series?
            
            if time_12.shape[0]<=n_points_thresh: 
                #print('%d|%d no shared time'%(ii,jj))
                continue
            
            # calculate correlation, rms, sum of differences between shared i and j data
            # z_1 is the first difference between each datapoints for record 1
            z_1  = (data_1[int_1][:-1]-data_1[int_1][1:])
            z_1 -= np.mean(z_1)        
            if np.std(z_1)!=0: 
                z_1 /= np.std(z_1)
                
            
            # z_2 is the first difference between each datapoints for record 2
            z_2  = (data_2[int_2][:-1]-data_2[int_2][1:])
            z_2 -= np.mean(z_2)
            if np.std(z_2)!=0: 
                z_2 /= np.std(z_2)
                
            
            correlations[ii, jj] = np.corrcoef(np.vstack([data_1[int_1], data_2[int_2]]))[0,1]
            
            rmse[ii, jj]         = np.sqrt(np.sum((data_1[int_1]-data_2[int_2])**2)/len(time_12))

            
            rmse_diff[ii, jj]    = np.sqrt(np.sum((z_1-z_2)**2)/len(time_12))
            corr_diff[ii, jj]    = np.corrcoef(np.vstack([z_1, z_2]))[0,1]

            
            
            if ((np.isnan(correlations[ii,jj]) & np.isnan(rmse[ii,jj]))|(np.isnan(corr_diff[ii,jj]) & np.isnan(rmse_diff[ii,jj]))):
                print('!!! %d|%d nan detected in both correlation and rmse of record data (or its 1st difference). Danger of missing duplicate!!'%(ii, jj))
                for idd, dd in enumerate([correlations[ii,jj], rmse[ii,jj], rmse_diff[ii,jj],  corr_diff[ii,jj]]):
                    if np.isnan(dd):
                        print('nan detected in %s'%(['correlation','RMSE','RMSE of 1st difference', 'correlation of 1st difference'][idd]))
            
            # DEFINE CRITERIA:
            
            meta_crit         = (arch_1 == arch_2) & (type_1 == type_2) # archive types and proxy types must agree
            elevation_not_nan = ((~np.isnan(df['geo_meanElev'].iloc[ii]))& ~np.isnan(df['geo_meanElev'].iloc[jj]))
            elevation_dist    = np.abs(df['geo_meanElev'].iloc[ii]-df['geo_meanElev'].iloc[jj])

            
            location_crit = ((distances_km[ii,jj] <= dist_tolerance_km) & 
                             (elevation_dist<=elev_tolerance if elevation_not_nan else True))
            #print('loc', location_crit)
    # i,j are located within dist_tolerance_km of each others and have the same elevation (provided they're not nans)
            overlap_crit  = ((len(time_12) > n_points_thresh) |
                             (len(time_1) <= n_points_thresh) |
                             (len(time_2)  <= n_points_thresh) )  # we have a sufficient time overlap between the datasets 
            site_crit     = (np.any([s1 in s2 for s2 in site_2.split(' ') for s1 in site_1.split(' ')])| 
                             np.any([s1 in s2 for s2 in site_2.split('-') for s1 in site_1.split('-')])| 
                             np.any([s1 in s2 for s2 in site_2.split('_') for s1 in site_1.split('_')])|
                             np.any([site_1 in site_2, site_2 in site_1])) # there is at least some overlap in the site name 
            corr_crit     = (((correlations[ii, jj] > corr_thresh) | 
                             (corr_diff[ii, jj] > corr_diff_thresh)) # correlation between shared datapoints exceeds correlation threshol dor correlation of first difference above threshold
                             & 
                             ((rmse[ii, jj] < rmse_thresh) | 
                             (rmse_diff[ii, jj] < rmse_diff_thresh))) #  RMSE of records or of first difference below threshold
            url_crit       = (url_1==url_2 if db_1==db_2 else True)
            if print_output:
                print('--------')
                print(id_1, id_2)
                print('archive and proxy match: %s'%meta_crit)
                print('site match: %s'%site_crit)
                print('location match: %s'%location_crit)
                print('correlation high: %s'%corr_crit)
                print('overlap %s'%overlap_crit)
            if (meta_crit & site_crit #& url_crit
                & location_crit
                & corr_crit
                & overlap_crit)|((correlations[ii, jj] > 0.98)& overlap_crit):
                # if these criteria are satisfied, we found a possible duplicate!
                n_pot_dups    += 1
                pot_dup_inds  += [[ii, jj]]
                pot_dup_IDs   += [df['datasetId'].iloc[[ii, jj]].values]
                pot_dup_corrs += [[correlations[ii, jj]]]
                pot_dup_dists += [[distances_km[ii,jj]]]
                pot_dup_meta  += [[ii, jj, df['datasetId'].iloc[ii], df['datasetId'].iloc[jj], correlations[ii, jj], distances_km[ii,jj]]]
                print('--> Found potential duplicate: %d: %s&%d: %s (n_potential_duplicates=%d)'%(ii, id_1, jj, id_2, n_pot_dups))
                
    if save: 
        os.makedirs(ddir, exist_ok=True)
        write_csv(pot_dup_meta, ddir+fn)
        
    print('='*60)
    print('Saved indices, IDs, distances, correlations in %s'%ddir)
    print('='*60)
    print('Detected %d possible duplicates in %s.'%(n_pot_dups, df.name))
    print('='*60)
    if len(pot_dup_inds)<=10:
        print('='*60)
        print('Indices: %s'%(', ').join([str(pdi) for pdi in pot_dup_inds]))
        print('IDs: %s'%(', ').join([pdi[0]+' + '+pdi[1] for pdi in pot_dup_IDs]))
        print('='*60)
    return #pot_dup_inds, pot_dup_IDs, distances_km, correlations

#==============================================================================
def dup_plot(df, ii, jj, id_1, id_2, time_1, time_2, time_12, data_1, data_2, int_1, int_2, 
             pot_dup_corr, keys_to_print=['originalDatabase', 'originalDataURL', 'datasetId', 'archiveType', 
                                          'proxy | variableName', #'archive|proxy', 
                                          'geo_siteName', 'lat | lon | elev', 'mean | std | units', 'year' ], 
             dup_mdata_row=[], plot_text=True, fig_scale=1):
    """
    Plots the duplicate candidates. Plots the record data as a timeseries of anomalies in a common panel (w.r.t. shared time period) and prints out the most relevant metadata. Highlights identical metadata in orange and different metadata in green.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing proxy records and metadata.
    ii : int
        Index of the first record.
    jj : int
        Index of the second record.
    id_1 : str
        Dataset ID of the first record.
    id_2 : str
        Dataset ID of the second record.
    time_1 : array-like
        Time vector for the first record.
    time_2 : array-like
        Time vector for the second record.
    time_12 : array-like
        Shared time points between both records.
    data_1 : array-like
        Data values of the first record.
    data_2 : array-like
        Data values of the second record.
    int_1 : array-like
        Indices of shared times in the first record.
    int_2 : array-like
        Indices of shared times in the second record.
    pot_dup_corr : float
        Correlation between the two records.
    keys_to_print : list of str, optional
        List of metadata keys to display. Default is a standard set of keys.
    dup_mdata_row : list, optional
        Stores metadata differences for output.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object of the duplicate plot.
    dup_mdata_row : list
        Metadata rows generated for display.
    """
    from matplotlib import font_manager
    fs        = 10
    scale     = 0.16
    init_offs = -.6
    str_limit = 50
    
    fig  = plt.figure(figsize=(10, 8.5), dpi=300*fig_scale)
    grid = GS(3,4, wspace=0.6)
    ax   = plt.subplot(grid[0,:3])
    #
    label1 = f'#{ii}: {id_1}' 
    # print('int_1', int_1)
    ax.scatter(time_1, data_1-np.mean(data_1[int_1]), facecolor='None', 
               edgecolor='tab:blue', marker='o',#marker='o',
               s=15, label=label1, alpha=0.9)
    
    label2 = f'#{jj}: {id_2}'# f'ID2: {id_2} (#{jj}))' 
    ax.scatter(time_2, data_2-np.mean(data_2[int_2]), color='tab:red', marker='x',#marker='D',
               s=15, label=label2, alpha=0.9)
    
    
    ax.legend(loc='upper right')
    ax.set_ylabel(f'anomalies w.r.t. {int(time_12[0])}-{int(time_12[-1])} \n '+
                  f'#{ii}: '+str(df['paleoData_units'].iloc[ii])+f'#{jj}: '+str(df['paleoData_units'].iloc[jj]))
    
    ax.set_xlabel(df['yearUnits'].iloc[ii])
    ax.set_title('Possible duplicates. Correlation=%3.2f'%pot_dup_corr
                 +'. Time overlap=%d (%d'%(len(time_12), len(time_12)/len(time_1)*100)+
                 '%'+', %d'%(len(time_12)/len(time_2)*100)+'%)')

    txt = ('Metadata (differences are highlighted in blue; only first '+
           str(str_limit)+' characters are displayed)')

    # column positions
    cols = [.1, -.1+0, -.1+.3, -.1+.9+0.15]
    
    rows = [-0.3]+[-0.45]*3
    
    ax.text(cols[0], rows[0], txt, transform=ax.transAxes, ha='left', fontsize=fs-1)
    
    ax.text(cols[1], rows[1],'Key', transform=ax.transAxes, fontsize=fs+2, ha='left')
    ax.text(cols[2], rows[2],'Record 1 (blue circles)', transform=ax.transAxes, ha='left', fontsize=fs+1)
    ax.text(cols[3], rows[3],'Record 2 (red crosses)',  transform=ax.transAxes, ha='left', fontsize=fs+1)
    # plot the metadata 
    for ik, key in enumerate(keys_to_print):
        if key=='lat | lon | elev':
            metadata_field1 = '%3.1f | %3.1f | %3.1f'%(df['geo_meanLat'].iloc[ii],
                                                   df['geo_meanLon'].iloc[ii], 
                                                   df['geo_meanElev'].iloc[ii])
            metadata_field2 = '%3.1f | %3.1f | %3.1f'%(df['geo_meanLat'].iloc[jj],
                                                   df['geo_meanLon'].iloc[jj], 
                                                   df['geo_meanElev'].iloc[jj])
        elif key=='year':
            metadata_field1 = '%3.1f - %3.1f %s'%(np.min(df['year'].iloc[ii]), 
                                                  np.max(df['year'].iloc[ii]), 
                                                  df['yearUnits'].iloc[ii])
            metadata_field2 = '%3.1f - %3.1f %s'%(np.min(df['year'].iloc[jj]), 
                                                  np.max(df['year'].iloc[jj]), 
                                                  df['yearUnits'].iloc[jj])
        elif key=='mean | std | units':
            metadata_field1 = '%4.2f | %4.2f | %s'%(np.mean(df['paleoData_values'].iloc[ii]), 
                                                  np.std(df['paleoData_values'].iloc[ii]), 
                                                  df['paleoData_units'].iloc[ii])
            metadata_field2 = '%4.2f | %4.2f | %s'%(np.mean(df['paleoData_values'].iloc[jj]), 
                                                  np.std(df['paleoData_values'].iloc[jj]), 
                                                  df['paleoData_units'].iloc[jj])
        elif key=='archive | proxy': #'archiveType','paleoData_proxy',
            metadata_field1 = '%s | %s'%(df['archiveType'].iloc[ii],
                                       df['paleoData_proxy'].iloc[ii])
            metadata_field2 = '%s | %s'%(df['archiveType'].iloc[jj],
                                       df['paleoData_proxy'].iloc[jj])
        elif key=='proxy | variableName': #'archiveType','paleoData_proxy',
            metadata_field1 = '%s | %s'%(df['paleoData_proxy'].iloc[ii],
                                       df['paleoData_variableName'].iloc[ii])
            metadata_field2 = '%s | %s'%(df['paleoData_proxy'].iloc[jj],
                                       df['paleoData_variableName'].iloc[jj])
        else:
            metadata_field1, metadata_field2 = df[key].iloc[[ii, jj]]
            
        if 'URL' in key:
            metadata_field1 = metadata_field1.replace('https://','')
            metadata_field2 = metadata_field2.replace('https://','')
            
        row = init_offs-scale*ik
        col = [-.1, -.1+.2, -.1+.75]

        
        
        plt.text(cols[1], row, ' '*180,transform=ax.transAxes, fontsize=fs+1, ha='left', 
                 bbox={'facecolor': 'tab:grey',   'alpha':.05, 'pad':2.5, 'edgecolor': 'None'
                      })
        # if metadata_field1 != metadata_field2:
        #     c = 'white'
        #     # plt.text(cols[1], row, ' '*250,transform=ax.transAxes, fontsize=fs+1, ha='left', 
        #     #          bbox={'facecolor': 'tab:blue', 
        #     #                'alpha':.2, 'pad':2.5, 'edgecolor': 'None'
        #     #               })
        # else:
        #     c = 'tab:orange'
        #     # plt.text(cols[1], row, ' '*250,transform=ax.transAxes, fontsize=fs+1, ha='left',
        #     #          bbox={'facecolor':'tab:orange','alpha':.2, 'pad':2.5, 'edgecolor': 'None'})
        
        # ax.text(cols[1], row, key, transform=ax.transAxes,fontsize=fs, ha='left', 
        #         bbox={'facecolor':c,'alpha':.2, 'pad':2.5, 'edgecolor': 'None'})
        # ax.text(cols[2], row, str(metadata_field1)[0:str_limit],ha='left', 
        #         transform=ax.transAxes,fontsize=fs, 
        #         bbox={'facecolor':c,'alpha':.2, 'pad':2.5, 'edgecolor': 'None'})
        # ax.text(cols[3], row, str(metadata_field2)[0:str_limit],ha='left', 
        #         transform=ax.transAxes,fontsize=fs, 
        #         bbox={'facecolor':c,'alpha':.2, 'pad':2.5, 'edgecolor': 'None'})


        # In the metadata loop, replace the current text plotting with:

        

        # if metadata_field1 != metadata_field2:
            # Plot background for the whole row
        plt.text(cols[1], row, ' '*180, transform=ax.transAxes, fontsize=fs+1, ha='left', 
                 bbox={'facecolor': 'tab:grey', 'alpha':.05, 'pad':2.5, 'edgecolor': 'None'})
        
        # For compound fields, split and highlight individually
       
        if '|' in str(metadata_field1):# Plot background for the whole row
            plt.text(cols[1], row, ' '*180, transform=ax.transAxes, fontsize=fs+1, ha='left', 
                     bbox={'facecolor': 'tab:grey', 'alpha':.05, 'pad':2.5, 'edgecolor': 'None'})
    
                
            parts1 = str(metadata_field1).split('|')
            parts2 = str(metadata_field2).split('|')
            
            # Plot the key normally
            ax.text(cols[1], row, key, transform=ax.transAxes, fontsize=fs, #family='monospace',
                    ha='left')
            
            # Calculate spacing for each part
            cumulative_offset = 0
            char_width = 0.015  # Approximate character width in axes coordinates (adjust as needed)
            
            for i, (p1, p2) in enumerate(zip(parts1, parts2)):
                p1, p2 = p1.strip(), p2.strip()
                color = 'tab:orange' if p1 == p2 else 'white'
                
                # For record 1 (cols[2])
                ax.text(cols[2] + cumulative_offset, row, p1, ha='left',#family='monospace',
                       transform=ax.transAxes, fontsize=fs,
                       bbox={'facecolor': color, 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
                
                # For record 2 (cols[3])
                ax.text(cols[3] + cumulative_offset, row, p2, ha='left',#family='monospace',
                       transform=ax.transAxes, fontsize=fs,
                       bbox={'facecolor': color, 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
                
                # Add separator if not last element
                if i < len(parts1) - 1:
                    sep_offset = cumulative_offset + len(p1) * char_width
                    ax.text(cols[2] + sep_offset, row, ' | ', ha='left',#family='monospace',
                           transform=ax.transAxes, fontsize=fs)
                    ax.text(cols[3] + sep_offset, row, ' | ', ha='left',#family='monospace',
                           transform=ax.transAxes, fontsize=fs)
                    cumulative_offset = sep_offset + 3 * char_width
                else:
                    cumulative_offset += len(p1) * char_width
                
        else:
            if metadata_field1 != metadata_field2:
                # Non-compound field - use original logic
                c = 'white'
                ax.text(cols[1], row, key, transform=ax.transAxes, fontsize=fs, ha='left', #family='monospace',
                        bbox={'facecolor': c, 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
                ax.text(cols[2], row, str(metadata_field1)[0:str_limit], ha='left', #family='monospace',
                        transform=ax.transAxes, fontsize=fs,
                        bbox={'facecolor': c, 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
                ax.text(cols[3], row, str(metadata_field2)[0:str_limit], ha='left',#family='monospace',
                        transform=ax.transAxes, fontsize=fs,
                        bbox={'facecolor': c, 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
            else:
                # Fields match - keep original orange highlighting
                c = 'tab:orange'
                plt.text(cols[1], row, ' '*180, transform=ax.transAxes, fontsize=fs+1, ha='left', #family='monospace',
                         bbox={'facecolor': 'tab:grey', 'alpha':.05, 'pad':2.5, 'edgecolor': 'None'})
                ax.text(cols[1], row, key, transform=ax.transAxes, fontsize=fs, ha='left',#family='monospace',
                        bbox={'facecolor': 'None', 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
                ax.text(cols[2], row, str(metadata_field1)[0:str_limit], ha='left',#family='monospace',
                        transform=ax.transAxes, fontsize=fs,
                        bbox={'facecolor': c, 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
                ax.text(cols[3], row, str(metadata_field2)[0:str_limit], ha='left',#family='monospace',
                        transform=ax.transAxes, fontsize=fs,
                        bbox={'facecolor': c, 'alpha': .2, 'pad': 2.5, 'edgecolor': 'None'})
        
        dup_mdata_row += [[metadata_field1, metadata_field2][m_ind].split('|')[s_ind]   
                          for s_ind in range(len(metadata_field1.split('|'))) 
                          for m_ind in [0,1]]
    ax2   = plt.subplot(grid[0,3])

    x  = data_1[int_1]-np.mean(data_1[int_1])
    y  = data_2[int_2]-np.mean(data_2[int_2])
    
    # density_scatter(x , y, ax = ax2, fig=fig)
    
    # xy = np.vstack([x,y])
    # z  = gaussian_kde(xy)(xy)

    # idx = z.argsort()
    # x, y, z = x[idx], y[idx], z[idx]
    

    density_scatter(x , y, ax2, s=5)
    # ax2.scatter(x, y, color='k', marker='+', alpha=0.5)#,
    #             #s=15, linestyle='None')
    plt.xlabel(label1)
    plt.ylabel(label2)
    
    # Add 1:1 line spanning the data range
    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax2.plot([min_val, max_val], [min_val, max_val],color='tab:grey',  
             alpha=0.75, zorder=0, label='1:1 line')
        
    
    return fig, dup_mdata_row


def density_scatter( x , y, ax = None, fig=None, sort = True, bins = 20, **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    
    from matplotlib import cm
    from matplotlib.colors import Normalize 
    from scipy.interpolate import interpn
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )

    norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = plt.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    cbar.ax.set_ylabel('data density')

    return ax
    
#==============================================================================
def plot_duplicates(df, save_figures=True, write_output=True, display=False):
    """
    Generate plots of potential duplicate records in a proxy database and optionally save their metadata.

    This function identifies potential duplicate entries in a paleoclimate proxy dataset,
    visualizes them using the `dup_plot` function, and saves the metadata of duplicates as a CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing proxy data and metadata. Expected columns include:
            - 'geo_meanLat', 'geo_meanLon' : Latitude and longitude of the site.
            - 'year', 'paleoData_values' : Time vector and proxy values.
            - 'archiveType', 'paleoData_proxy' : Archive type and proxy type.
            - 'datasetId' : Unique identifier or dataset name.
            - 'geo_siteName' : Site name.
            - 'originalDatabase' : Name of the original database.
    save_figures : bool, optional
        If True, save the generated figures to disk. Default is True.
    write_output : bool, optional
        If True, save the duplicate metadata to a CSV file. Default is True.

    Returns
    -------
    None
        The function primarily generates plots and writes metadata; it does not return any objects.

    Notes
    -----
    - The function assumes that potential duplicates have already been identified by a separate
      process (e.g., `find_duplicates`) and that a corresponding CSV exists in `df.name/dup_detection/`.
    - Metadata printed and saved includes site names, coordinates, dataset IDs, original database,
      and summary statistics (mean, std, units, etc.).
    - Figures are saved in PDF format by default using `save_fig`.
    """

    
    keys_to_print = ['originalDatabase', 'originalDataURL', 'datasetId', 'archiveType', 'proxy | variableName', #'archive|proxy',
                     'geo_siteName', 'lat | lon | elev', 'mean | std | units', 'year' ]

    ddir = 'data/%s/dup_detection/'%(df.name)+'dup_detection_candidates_'+df.name
    # load the potential duplicate data as found in find_duplicates function
    pot_dup_meta, head = read_csv(ddir, header=True)

    pot_dup_inds   = np.array(np.array(pot_dup_meta)[:, :2], dtype=int)
    pot_dup_corrs  = np.array(np.array(pot_dup_meta)[:, 4], dtype=float)
    pot_dup_dists  = np.array(np.array(pot_dup_meta)[:, 5], dtype=float)

    n_pot_dups   = pot_dup_inds.shape[0]
    
    dup_mdata  = []
    dup_mdata += [['ind 1', 'ind 2']+[ki+' %d'%ii_kk for kk in keys_to_print for ki in kk.split('|')  for ii_kk in [1, 2]]]
    
    # loop through the potential duplicates.
    for i_pot_dups, (ii, jj) in enumerate(pot_dup_inds):
        dup_mdata_row    = [ii,jj]
        # data and metadata associated with the two potential duplicates
        id_1, id_2       = df['datasetId'].iloc[[ii, jj]]
        time_1, time_2   = df['year'].iloc[[ii, jj]]
        time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) # possibly need to round these time series?
        data_1           = np.array(df['paleoData_values'].iloc[ii])
        data_2           = np.array(df['paleoData_values'].iloc[jj])
        
        lat_1, lat_2     = df['geo_meanLat'].iloc[[ii, jj]]
        lon_1, lon_2     = df['geo_meanLon'].iloc[[ii, jj]]
        site_1, site_2   = df['geo_siteName'].iloc[[ii, jj]]
        or_db_1, or_db_2 = df['originalDatabase'].iloc[[ii, jj]]
        
        print('> %d/%d'%(i_pot_dups, n_pot_dups), #set_txt, 
               id_1, id_2, #var_1,var_2, season_1,season_2,
               pot_dup_dists[i_pot_dups], pot_dup_corrs[i_pot_dups],
               sep=',')
    
        # Plot observations
        #-----------------------------------------------------
        
        # Metadata display parameters
        fig, dup_mdata_row = dup_plot(df, ii, jj, id_1, id_2, time_1, time_2, time_12, data_1, 
                                      data_2, int_1, int_2, pot_dup_corrs[i_pot_dups],
                                      keys_to_print, dup_mdata_row)

        
        dup_mdata += [dup_mdata_row]
        plt.show(block=False)
        if save_figures:
            save_fig(fig, f'%03d_{id_1}_{id_2}'%i_pot_dups+'_'+f'_{ii}_{jj}', dir='%s/dup_detection'%df.name, figformat='pdf')
        if not display:
            plt.close(fig)

        
        print('=== POTENTIAL DUPLICATE %d/%d'%(i_pot_dups, n_pot_dups)+': %s+%s ==='%(df['datasetId'].iloc[ii], df['datasetId'].iloc[jj]))
    return 

#==============================================================================


def duplicate_decisions(df, operator_details=False, choose_recollection=True, #keep_all=False, 
                        plot=True, remove_identicals=True, dist_tolerance_km=8):
    """
    Review potential duplicate pairs in a proxy database and decide which records to keep, remove, or combine.

    This function walks through each potential duplicate pair identified in a dataset,
    displays metadata and optionally plots the data, and allows the operator to make decisions.
    Decisions are saved to a CSV file, and a duplicate-free dataframe can be generated.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing proxy data and metadata. Expected columns include:
            - 'geo_meanLat', 'geo_meanLon' : Latitude and longitude of the site.
            - 'year', 'paleoData_values' : Time vector and proxy values.
            - 'archiveType', 'paleoData_proxy' : Archive type and proxy type.
            - 'datasetId' : Unique identifier or dataset name.
            - 'geo_siteName' : Site name.
            - 'originalDatabase' : Name of the original database.
            - 'geo_meanElev' : Mean elevation of the site (optional for duplicate checks).
            - 'Hierarchy' : Numeric value representing dataset priority (used for auto-decisions).
            - 'originalDataURL' : URL of the original dataset.

    operator_details : tuple or bool, optional
        Tuple containing (initials, fullname, email) of the operator. If False, user input is requested.
        Default is False.
    choose_recollection : bool, optional
        If True, automatically selects the record that is a recollection or update when applicable.
        Default is True.
    plot : bool, optional
        If True, generate plots for manual inspection of duplicate pairs. Default is True.
    remove_identicals : bool, optional
        If True, automatically remove records that are identical in data and metadata. Default is True.
    dist_tolerance_km : float, optional
        Maximum distance (in km) for considering records as recollection updates. Default is 8 km.

    Returns
    -------
    None
        Decisions are saved as CSV backup and final CSV in the `df.name/dup_detection/` directory.

    Notes
    -----
    - Automatic decisions are made based on data identity, metadata identity, perfect correlation,
      and recollection indicators in site names.
    - Manual decisions are prompted via command-line input if automatic rules do not apply.
    - Decision types include:
        - 'AUTO: UPDATE' : Automatically select record that is a recollection/update.
        - 'AUTO: IDENTICAL' : Automatically select record based on hierarchy if records are identical.
        - 'MANUAL' : Decision requires operator input.
    - Figures are saved with a standardized naming convention and linked in the CSV output.
    """

    import datetime
    # Select the metadata keys to show on the output figures - keys depend on the 
    # dataframe and need to be determined according to the dataframe input. 
    # if using this code on fused_database.pkl, make sure that all entries are available 
    # in the dataframe1 Use fuse_datasets.py to fuse and homogenise metadata  of different databases.

    if not operator_details:
        initials = input('Please enter your initials here:')
        fullname = input('Please enter your full name here:')
        email    = input('Please enter your email address here:')
    else:
        initials, fullname, email = operator_details
    date_time= str(datetime.datetime.utcnow())+' (UTC)'
    header   = [['# Decisions for duplicate candidate pairs. ']]
    header  += [['# Operated by %s (%s)'%(fullname, initials)]]
    header  += [['# E-Mail: %s'%email]]
    header  += [['# Created on: %s'%date_time]]

    
    dirname = 'data/%s/dup_detection/'%(df.name)
    filename = f'dup_decisions_{df.name}' # name of csv file which saves duplicate candidate pairs
    
    filename+=f'_{initials}'
    
    detection_file = dirname+'dup_detection_candidates_'+df.name
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # try to load data from backup!
    try:
        data, hh = read_csv(dirname+filename+'_BACKUP', header=True, last_header_row=4)
        backup_file=input(f'Found backup file ({dirname+filename}_BACKUP.csv). Do you want to start decision process from the backup file? [y/n]')
        if backup_file=='n':
            raise FileNotFoundError
        data, hh = read_csv(dirname+filename+'_BACKUP', header=True, last_header_row=4)
        print('header', hh)
        print('data', list(data))
        data = list(data)
        
        load_saved_data      = True
        last_index = len(data)
        print('start with index: ', last_index)
    except FileNotFoundError:
        print('No back up.')
        load_saved_data      = False
        data = []
        pass

    # load the potential duplicate data as found in find_duplicates function
    pot_dup_meta, head = read_csv(detection_file, header=True)

    pot_dup_inds   = np.array(np.array(pot_dup_meta)[:, :2], dtype=int)  # indices for each pairs
    # pot_dup_IDs    = np.array(np.array(pot_dup_meta)[:, 2:4], dtype=int) # IDs for each pairs
    pot_dup_corrs  = np.array(np.array(pot_dup_meta)[:, 4], dtype=float)
    pot_dup_dists  = np.array(np.array(pot_dup_meta)[:, 5], dtype=float)


    n_pot_dups   = pot_dup_inds.shape[0]
    
    # loop through the potential duplicates.
    decisions = ['KEEP']*df.shape[0] #set False if index should be discarded from dataset.
    cols    = [['index 1', 'index 2', 'figure path',
                'datasetId 1', 'datasetId 2', 
                'originalDatabase 1', 'originalDatabase 2',
                'geo_siteName 1', 'geo_siteName 2', 
                'geo_meanLat 1', 'geo_meanLat 2', 
                'geo_meanLon 1', 'geo_meanLon 2', 
                'geo_meanElevation 1', 'geo_meanElevation 2', 
                'archiveType 1', 'archiveType 2',
                'paleoData_proxy 1', 'paleoData_proxy 2',
                'originalDataURL 1', 'originalDataURL 2',
                'year 1', 'year 2',
                'Decision 1', 'Decision 2',
                'Decision type', 'Decision comment' ]]
    dup_dec = []

    
        # Write header and existing data ONCE before the loop (only if starting fresh)
    if not load_saved_data:
        with open(dirname+filename+'_BACKUP.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(header)
            writer.writerows(cols)
    else:
        # If loading from backup, populate dup_dec with existing data
        for data_row in list(data):
            dup_dec += [list(data_row)]
            
    # Open backup file in APPEND mode for writing new decisions
    with open(dirname+filename+'_BACKUP.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        
        for i_pot_dups, (ii, jj) in enumerate(pot_dup_inds):
            if load_saved_data:
                if i_pot_dups<last_index: continue
                else:
                    print(ii, jj, last_index, i_pot_dups)
            # data and metadata associated with the two potential duplicates
            id_1, id_2       = df['datasetId'].iloc[[ii, jj]]
            time_1, time_2   = df['year'].iloc[[ii, jj]]
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) 
            data_1           = np.array(df['paleoData_values'].iloc[ii])
            data_2           = np.array(df['paleoData_values'].iloc[jj])
            
            lat_1, lat_2     = df['geo_meanLat'].iloc[[ii, jj]]
            lon_1, lon_2     = df['geo_meanLon'].iloc[[ii, jj]]
            site_1, site_2   = df['geo_siteName'].iloc[[ii, jj]]
            or_db_1, or_db_2 = df['originalDatabase'].iloc[[ii, jj]]

            auto_choice = '1' if df['Hierarchy'].iloc[ii]>=df['Hierarchy'].iloc[jj] else '2'
            
            print('> %d/%d'%(i_pot_dups+1, n_pot_dups), #set_txt, 
                   id_1, id_2, #var_1, var_2, season_1,season_2,
                   pot_dup_dists[i_pot_dups], pot_dup_corrs[i_pot_dups],
                   sep=',')
        
            # Print the interpretation values 
            print('====================================================================')
            print('=== POTENTIAL DUPLICATE %d/%d'%(i_pot_dups, n_pot_dups)+': %s+%s ==='%(df['datasetId'].iloc[ii], df['datasetId'].iloc[jj]))
            print('=== URL 1: %s   ==='%(df['originalDataURL'].iloc[ii]))
            print('=== URL 2: %s   ==='%(df['originalDataURL'].iloc[jj]))
            
            keep = ''
            dec_comment = ''

            
            elevation_nan = (np.isnan(df['geo_meanElev'].iloc[ii])|
                             np.isnan(df['geo_meanElev'].iloc[jj]))
            
            metadata_identical = ((np.abs(lat_1-lat_2)<=0.1) & 
                                  (np.abs(lon_1-lon_2)<=0.1)  & 
                                  ((np.abs(df['geo_meanElev'].iloc[ii]-df['geo_meanElev'].iloc[jj])<=1) 
                                    | elevation_nan) & 
                                  (df['archiveType'].iloc[ii]==df['archiveType'].iloc[jj]) & 
                                  (df['paleoData_proxy'].iloc[ii]==df['paleoData_proxy'].iloc[jj]) 
                                 )
            sites_identical     = site_1==site_2
            URL_identical       = (df['originalDataURL'].iloc[ii]==df['originalDataURL'].iloc[jj]) 
            # print(data_1, data_2)
            data_identical      = (list(data_1)==list(data_2)) & (list(time_1)==list(time_2))
            correlation_perfect = (True if pot_dup_corrs[i_pot_dups]>=0.98 else False) & (len(time_1)==len(time_2))
            print('True if pot_dup_corrs[i_pot_dups]>=0.98 else False', True if pot_dup_corrs[i_pot_dups]>=0.98 else False)
            print('(len(time_1)==len(time_2))', (len(time_1)==len(time_2)))
            
            print('metadata_identical: ', metadata_identical)
            print('lat', (np.abs(lat_1-lat_2)<=0.1), 'lon' , (np.abs(lon_1-lon_2)<=0.1)   ,'elevation',
                                  ( (np.abs(df['geo_meanElev'].iloc[ii]-df['geo_meanElev'].iloc[jj])<=1) 
                                    | elevation_nan) , 'archivetype',
                                  (df['archiveType'].iloc[ii]==df['archiveType'].iloc[jj]) ,'paleodata_proxy',
                                  (df['paleoData_proxy'].iloc[ii]==df['paleoData_proxy'].iloc[jj]) 
                                 )
            print('sites_identical: ', sites_identical)
            print('URL_identical: ', URL_identical)
            print('data_identical: ', data_identical)
            print('correlation_perfect: ', correlation_perfect)
            figpath = 'no figure'
            while keep not in ['1', '2', 'b', 'n', 'c']:
                # go through the data and metadata and check if decision is =automatic or manual
                if (choose_recollection 
                    # if record 1 is update of record 2, choose 1
                      & np.any([(ss in site_1.lower()) for ss in ['recollect', 'update', 're-collect']]) 
                      & (pot_dup_dists[i_pot_dups]<dist_tolerance_km)
                      & ~np.any([ss in site_2.lower() for ss in ['recollect', 'update', 're-collect']])
                     ):
                    keep = '1'
                    dec_comment = 'Record 1 (%s) is UPDATE of record 2(%s) . Automatically choose 1.'%(id_1, id_2)
                    print(dec_comment)
                    dec_type='AUTO: UPDATE'
                elif (choose_recollection 
                      # if record 2 is update of record 1, choose 2
                      & np.any([ss in site_2.lower() for ss in ['recollect', 'update', 're-collect']]) 
                      & (pot_dup_dists[i_pot_dups]<dist_tolerance_km)
                      & ~np.any([ss in site_1.lower() for ss in ['recollect', 'update', 're-collect']])
                     ):
                    keep = '2'
                    dec_comment = 'Record 2 (%s) is UPDATE of record 1 (%s). Automatically choose 2.'%(id_2, id_1)
                    print(dec_comment)
                    dec_type='AUTO: UPDATE'
                elif (remove_identicals & metadata_identical & sites_identical & URL_identical & (data_identical|correlation_perfect)):
                    # if all metadata and data matches, choose record according to hierarchy of databases
                    if data_identical:
                        dec_comment = 'RECORDS IDENTICAL (identical data). Automatically choose #%s.'%auto_choice
                    else:
                        dec_comment = 'RECORDS IDENTICAL (perfect correlation). Automatically choose #%s.'%auto_choice
                    
                    print(dec_comment)
                    keep = auto_choice#'1'
                    dec_type='AUTO: IDENTICAL'
                elif (remove_identicals & (data_identical|correlation_perfect)):
                    # if most metadata and data matches except URL and/or site name, choose record according to hierarchy of databases
                    if data_identical:
                        dec_comment = 'RECORDS IDENTICAL (identical data) except for metadata. Automatically choose #%s.'%auto_choice
                    else:
                        dec_comment = 'RECORDS IDENTICAL (perfect correlation) except for metadata. Automatically choose #%s.'%auto_choice
                        
                    print(dec_comment)
                    keep = auto_choice#'1'
                    dec_type='AUTO: IDENTICAL except for URLs and/or geo_siteName.'
                    
                else:
                    
                    dec_type = 'MANUAL'
                    fig, dup_mdata_row = dup_plot(df, ii, jj, id_1, id_2, time_1, time_2, time_12, data_1, data_2, int_1, int_2, pot_dup_corrs[i_pot_dups])
                    plt.show(block=False)
                    print('**Decision required for this duplicate pair (see figure above).**')
                    
                    print('Before inputting your decision. Would you like to leave a comment on your decision process?')
                    dec_comment = input(f'{i_pot_dups+1}/{n_pot_dups}: **COMMENT** Please type your comment here and/or press enter.')
                    keep = input(f'{i_pot_dups+1}/{n_pot_dups}: **DECISION** Keep record 1 (%s, blue circles) [1], record 2 (%s, red crosses) [2], keep both [b], keep none [n] or create a composite of both records [c]?  [Type 1/2/b/n/c]:'%(id_1, id_2))

                
                
                    save_fig(fig, '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+'_%d_%d'%(ii,jj), dir=f'/dup_detection/{df.name}', figformat='jpg')
        
                    
                    figpath    = 'https://nzero.umd.edu:444/hub/user-redirect/lab/tree/dod2k_v2.0/figs/dup_detection/%s'%df.name
                    figpath  += '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+'_%d_%d,jpg'%(ii,jj)
                    
                # now write down the decision
                if keep=='1':
                    print('KEEP BLUE CIRCLES: keep %s, remove %s.'%(id_1, id_2))
                    decisions[ii]='KEEP'
                    decisions[jj]='REMOVE'
                elif keep=='2':
                    print('KEEP RED CROSSES: remove %s, keep %s.'%(id_1, id_2))
                    decisions[jj]='KEEP'
                    decisions[ii]='REMOVE'
                elif keep=='n':
                    print('REMOVE BOTH: remove %s, remove %s.'%(id_2, id_1))
                    decisions[ii]='REMOVE'
                    decisions[jj]='REMOVE'
                elif keep=='b':
                    print('KEEP BOTH: keep %s, keep %s.'%(id_1, id_2))
                    decisions[ii]='KEEP'
                    decisions[jj]='KEEP'
                elif keep=='c':
                    print('CREATE A COMPOSITE OF BOTH RECORDS: %s, %s.'%(id_1, id_2))
                    decisions[ii]='COMPOSITE'
                    decisions[jj]='COMPOSITE'


            
            cand_pair =[[ii, jj, figpath,
                        id_1, id_2, or_db_1, or_db_2, site_1, site_2,
                        lat_1, lat_2, lon_1, lon_2, 
                        df['geo_meanElev'].iloc[ii], df['geo_meanElev'].iloc[jj],
                        df['archiveType'].iloc[ii], df['archiveType'].iloc[jj],
                        df['paleoData_proxy'].iloc[ii], df['paleoData_proxy'].iloc[jj],
                        df['originalDataURL'].iloc[ii], df['originalDataURL'].iloc[jj],
                        '%3.1f-%3.1f'%(np.min(df['year'].iloc[ii]), np.max(df['year'].iloc[ii])), 
                        '%3.1f-%3.1f'%(np.min(df['year'].iloc[jj]), np.max(df['year'].iloc[jj])), 
                        decisions[ii], decisions[jj],
                        dec_type, dec_comment]]
            dup_dec  += cand_pair
            writer.writerows(cand_pair)
            print('write decision to backup file')
            # raise Exception
        
    print('=====================================================================')
    print('END OF DUPLICATE DECISION PROCESS.')
    print('=====================================================================')

    comment = '# '+input('Type your comment on your decision process here and/or press enter:')
    header  += [[comment]]
    # header  += cols
    
    print(np.array(dup_dec).shape)
    filename +=f'_{date_time[2:10]}'
    write_csv(np.array(dup_dec), dirname+filename, header=header, cols=cols)
    print('Saved the decisions under %s.csv'%(dirname+filename))
    
    print('Summary of all decisions made:')
    for ii_d in range(len(dup_dec)):
        keep_1=dup_dec[ii_d][-4]
        keep_2=dup_dec[ii_d][-3]
        print('#%d: %s record %s. %s record %s.'%(ii_d, keep_1, dup_dec[ii_d][3], 
                                                  keep_2, dup_dec[ii_d][4]))
    return 


def duplicate_decisions_multiple(df, operator_details=False, choose_recollection=True, #keep_all=False, 
                                 plot=True, remove_identicals=True, dist_tolerance_km=8, backup=True,
                                 comment=True, automate_db_choice=False):
    """
    Review potential duplicate pairs in a proxy database and decide which records to keep, remove, or combine.

    This function walks through each potential duplicate pair identified in a dataset,
    displays metadata and optionally plots the data, and allows the operator to make decisions.
    Decisions are saved to a CSV file, and a duplicate-free dataframe can be generated.

    ALLOWS FOR MULTIPLE DUPLICATES!

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing proxy data and metadata. Expected columns include:
            - 'geo_meanLat', 'geo_meanLon' : Latitude and longitude of the site.
            - 'year', 'paleoData_values' : Time vector and proxy values.
            - 'archiveType', 'paleoData_proxy' : Archive type and proxy type.
            - 'datasetId' : Unique identifier or dataset name.
            - 'geo_siteName' : Site name.
            - 'originalDatabase' : Name of the original database.
            - 'geo_meanElev' : Mean elevation of the site (optional for duplicate checks).
            - 'Hierarchy' : Numeric value representing dataset priority (used for auto-decisions).
            - 'originalDataURL' : URL of the original dataset.

    operator_details : tuple or bool, optional
        Tuple containing (initials, fullname, email) of the operator. If False, user input is requested.
        Default is False.
    choose_recollection : bool, optional
        If True, automatically selects the record that is a recollection or update when applicable.
        Default is True.
    plot : bool, optional
        If True, generate plots for manual inspection of duplicate pairs. Default is True.
    remove_identicals : bool, optional
        If True, automatically remove records that are identical in data and metadata. Default is True.
    dist_tolerance_km : float, optional
        Maximum distance (in km) for considering records as recollection updates. Default is 8 km.

    Returns
    -------
    None
        Decisions are saved as CSV backup and final CSV in the `df.name/dup_detection/` directory.

    Notes
    -----
    - Automatic decisions are made based on data identity, metadata identity, perfect correlation,
      and recollection indicators in site names.
    - Manual decisions are prompted via command-line input if automatic rules do not apply.
    - Decision types include:
        - 'AUTO: UPDATE' : Automatically select record that is a recollection/update.
        - 'AUTO: IDENTICAL' : Automatically select record based on hierarchy if records are identical.
        - 'MANUAL' : Decision requires operator input.
    - Figures are saved with a standardized naming convention and linked in the CSV output.
    """
    
    def create_ID_dup_dict(pot_dup_IDs):
        dup_dict         = {}
        reverse_dup_dict = {}
        for id1, id2 in pot_dup_IDs:
            # check if id1 appears in reverse_dup_dict: 
            if id1 in reverse_dup_dict:
                dup_id = reverse_dup_dict[id1] # if YES, it means it is already associated with a duplicate record, use this for mapping
            else:
                dup_id = id1 # if NO this is the first time id1 appears as a potential dup, then use this for mapping
            # dup_id is the unique mapper to the duplicate category.
            if dup_id not in dup_dict: 
                dup_dict[dup_id] = [] # create an empty dictionary key in case this is the first time that dup_id appears
            # if dup_id!=id1: # if id1 is not identical to the unique mapper
            #     if id1 not in dup_dict[dup_id]:
            #         dup_dict[dup_id]+=[id1]
            if id1 not in dup_dict[dup_id]:
                if dup_id!=id1:
                    dup_dict[dup_id]+=[id1] # add id1 to the dup_dict if not in there yet (keep entries unique) and id1 is not unique mapper
            if id2 not in dup_dict[dup_id]:
                dup_dict[dup_id]+=[id2]  # add id2 to the dup_dict if not in there yet (keep entries unique)
                
            reverse_dup_dict[id1] = dup_id # make sure id1 gets mapped back to its unique mapper
            reverse_dup_dict[id2] = dup_id # make sure id2 gets mapped back to its unique mapper
                
                    
        # # drop all PAIRS to get multiples only
        dup_dict_multiples = {}
        for kk, vv in dup_dict.items():
            if len(vv)>1:
                dup_dict_multiples[kk]=vv
        multiple_list = []
        for kk, vv in dup_dict_multiples.items():
            # print(kk, len(vv), vv)
            multiple_list.append(kk)
            for vvv in vv:
                multiple_list.append(vvv)
        print('------'*10)
        if len(dup_dict_multiples)>0:
            print('Detected MULTIPLE duplicates, including:')
            for kk, vv in dup_dict_multiples.items():
                print(kk, len(vv), vv)
            print('PLEASE PAY ATTENTION WHEN MAKING DECISIONS FOR THESE DUPLICATES!')
            print('The decision process will go through the duplicates on a PAIR-BY-PAIR basis, which is not optimised for multiple duplicates.')
            print('The multiples will be highlighted throughout the decision process.')
            print('Should the operator want to go back and revise a previous decision based on the presentation of a new candidate pair, they can manually modify the backup file to alter any previous decisions.')
        print('------'*10)
        return dup_dict_multiples, reverse_dup_dict, multiple_list 
        
    import datetime
    # Select the metadata keys to show on the output figures - keys depend on the 
    # dataframe and need to be determined according to the dataframe input. 
    # if using this code on fused_database.pkl, make sure that all entries are available 
    # in the dataframe1 Use fuse_datasets.py to fuse and homogenise metadata  of different databases.

    if not operator_details:
        initials = input('Please enter your initials here:')
        fullname = input('Please enter your full name here:')
        email    = input('Please enter your email address here:')
    else:
        initials, fullname, email = operator_details
    date_time= str(datetime.datetime.utcnow())+' (UTC)'
    header   = [['# Decisions for duplicate candidate pairs. ']]
    header  += [['# Operated by %s (%s)'%(fullname, initials)]]
    header  += [['# E-Mail: %s'%email]]
    header  += [['# Created on: %s'%date_time]]

    
    dirname = 'data/%s/dup_detection/'%(df.name)
    filename = f'dup_decisions_{df.name}' # name of csv file which saves duplicate candidate pairs
    
    filename+=f'_{initials}'
    
    detection_file = dirname+'dup_detection_candidates_'+df.name


    if automate_db_choice is not False:
        auto_db_choice = True
        auto_db_pref = automate_db_choice['preferred_db']
        auto_db_rej  = automate_db_choice['rejected_db']
    else:
        auto_db_choice = False
        auto_db_pref = ''
        auto_db_rej = ''
        
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # try to load data from backup!
    try:
        data, hh = read_csv(dirname+filename+'_BACKUP', header=True, last_header_row=4)
        if backup:
            backup_file=input(f'Found backup file ({dirname+filename}_BACKUP.csv). Do you want to start decision process from the backup file? [y/n]')
        else:
            backup_file='n'
        if backup_file=='n':
            raise FileNotFoundError
        data, hh = read_csv(dirname+filename+'_BACKUP', header=True, last_header_row=4)
        print('header', hh)
        print('data', list(data))
        data = list(data)
        
        load_saved_data      = True
        last_index = len(data)
        print('start with index: ', last_index)
    except FileNotFoundError:
        print('No back up.')
        load_saved_data      = False
        data = []
        pass

    # load the potential duplicate data as found in find_duplicates function
    pot_dup_meta, head = read_csv(detection_file, header=True)

    pot_dup_inds   = np.array(np.array(pot_dup_meta)[:, :2], dtype=int)  # indices for each pairs
    pot_dup_IDs    = np.array(np.array(pot_dup_meta)[:, 2:4], dtype=str) # IDs for each pairs
    pot_dup_corrs  = np.array(np.array(pot_dup_meta)[:, 4], dtype=float)
    pot_dup_dists  = np.array(np.array(pot_dup_meta)[:, 5], dtype=float)


    dup_dict_multiples, reverse_dict, multiple_list   = create_ID_dup_dict(pot_dup_IDs)
    # raise Exception

    n_pot_dups   = pot_dup_inds.shape[0]
    
    # loop through the potential duplicates.
    decisions = ['N/A']*df.shape[0] #set False if index should be discarded from dataset.
    cols    = [['index 1', 'index 2', 'figure path',
                'datasetId 1', 'datasetId 2', 
                'originalDatabase 1', 'originalDatabase 2',
                'geo_siteName 1', 'geo_siteName 2', 
                'geo_meanLat 1', 'geo_meanLat 2', 
                'geo_meanLon 1', 'geo_meanLon 2', 
                'geo_meanElevation 1', 'geo_meanElevation 2', 
                'archiveType 1', 'archiveType 2',
                'paleoData_proxy 1', 'paleoData_proxy 2',
                'originalDataURL 1', 'originalDataURL 2',
                'year 1', 'year 2',
                'Decision 1', 'Decision 2',
                'Decision type', 'Decision comment' ]]
    dup_dec = []

    
    # Write header and existing data ONCE before the loop (only if starting fresh)
    if not load_saved_data:
        # start a new backup file. 
        with open(dirname+filename+'_BACKUP.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(header)
            writer.writerows(cols)
            
    else:
        # If loading from backup, populate dup_dec with existing data
        for data_row in list(data):
            dup_dec += [list(data_row)]
            
    # Open backup file in APPEND mode for writing new decisions
    with open(dirname+filename+'_BACKUP.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        
        for i_pot_dups, (ii, jj) in enumerate(pot_dup_inds):
            if load_saved_data:
                if i_pot_dups<last_index: continue
                else:
                    print(ii, jj, last_index, i_pot_dups)
            # data and metadata associated with the two potential duplicates
            id_1, id_2       = df['datasetId'].loc[[ii, jj]]
            time_1, time_2   = df['year'].loc[[ii, jj]]
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) 
            data_1           = np.array(df['paleoData_values'].loc[ii])
            data_2           = np.array(df['paleoData_values'].loc[jj])
            
            lat_1, lat_2     = df['geo_meanLat'].loc[[ii, jj]]
            lon_1, lon_2     = df['geo_meanLon'].loc[[ii, jj]]
            site_1, site_2   = df['geo_siteName'].loc[[ii, jj]]
            or_db_1, or_db_2 = df['originalDatabase'].loc[[ii, jj]]

            auto_choice = '1' if df['Hierarchy'].loc[ii]>=df['Hierarchy'].loc[jj] else '2'
            
            print('> %d/%d'%(i_pot_dups+1, n_pot_dups), #set_txt, 
                   id_1, id_2, #var_1, var_2, season_1,season_2,
                   pot_dup_dists[i_pot_dups], pot_dup_corrs[i_pot_dups],
                   sep=',')
        
            # Print the interpretation values 
            print('====================================================================')
            print('=== POTENTIAL DUPLICATE %d/%d'%(i_pot_dups, n_pot_dups)+': %s+%s ==='%(df['datasetId'].loc[ii], df['datasetId'].loc[jj]))
            print('=== URL 1: %s   ==='%(df['originalDataURL'].loc[ii]))
            print('=== URL 2: %s   ==='%(df['originalDataURL'].loc[jj]))
            
            keep = ''
            dec_comment = ''

            
            elevation_nan = (np.isnan(df['geo_meanElev'].loc[ii])|
                             np.isnan(df['geo_meanElev'].loc[jj]))
            
            metadata_identical = ((np.abs(lat_1-lat_2)<=0.1) & 
                                  (np.abs(lon_1-lon_2)<=0.1)  & 
                                  ((np.abs(df['geo_meanElev'].loc[ii]-df['geo_meanElev'].loc[jj])<=1) 
                                    | elevation_nan) & 
                                  (df['archiveType'].loc[ii]==df['archiveType'].loc[jj]) & 
                                  (df['paleoData_proxy'].loc[ii]==df['paleoData_proxy'].loc[jj]) 
                                 )
            sites_identical     = site_1==site_2
            URL_identical       = (df['originalDataURL'].loc[ii]==df['originalDataURL'].loc[jj]) 
            # print(data_1, data_2)
            data_identical      = (list(data_1)==list(data_2)) & (list(time_1)==list(time_2))
            correlation_perfect = (True if pot_dup_corrs[i_pot_dups]>=0.98 else False) & (len(time_1)==len(time_2))

            # if one db is the preferred and the other the rejected db, only if automate_db_choice is not False
            auto_db_choice_cond = auto_db_choice & (((or_db_1==auto_db_pref) & (or_db_2==auto_db_rej))|((or_db_2==auto_db_pref) & (or_db_1==auto_db_rej)))
            
            print('True if pot_dup_corrs[i_pot_dups]>=0.98 else False', True if pot_dup_corrs[i_pot_dups]>=0.98 else False)
            print('(len(time_1)==len(time_2))', (len(time_1)==len(time_2)))
            
            print('metadata_identical: ', metadata_identical)
            print('lat', (np.abs(lat_1-lat_2)<=0.1), 'lon' , (np.abs(lon_1-lon_2)<=0.1)   ,'elevation',
                                  ( (np.abs(df['geo_meanElev'].loc[ii]-df['geo_meanElev'].loc[jj])<=1) 
                                    | elevation_nan) , 'archivetype',
                                  (df['archiveType'].loc[ii]==df['archiveType'].loc[jj]) ,'paleodata_proxy',
                                  (df['paleoData_proxy'].loc[ii]==df['paleoData_proxy'].loc[jj]) 
                                 )
            print('sites_identical: ', sites_identical)
            print('URL_identical: ', URL_identical)
            print('data_identical: ', data_identical)
            print('correlation_perfect: ', correlation_perfect)
            figpath = 'no figure'
            while keep not in ['1', '2', 'b', 'n', 'c']:
                # go through the data and metadata and check if decision is =automatic or manual
                if (choose_recollection 
                    # if record 1 is update of record 2, choose 1
                      & np.any([(ss in site_1.lower()) for ss in ['recollect', 'update', 're-collect']]) 
                      & (pot_dup_dists[i_pot_dups]<dist_tolerance_km)
                      & ~np.any([ss in site_2.lower() for ss in ['recollect', 'update', 're-collect']])
                     ):
                    keep = '1'
                    dec_comment = 'Record 1 (%s) is UPDATE of record 2(%s) . Automatically choose 1.'%(id_1, id_2)
                    print(dec_comment)
                    dec_type='AUTO: UPDATE'
                elif (choose_recollection 
                      # if record 2 is update of record 1, choose 2
                      & np.any([ss in site_2.lower() for ss in ['recollect', 'update', 're-collect']]) 
                      & (pot_dup_dists[i_pot_dups]<dist_tolerance_km)
                      & ~np.any([ss in site_1.lower() for ss in ['recollect', 'update', 're-collect']])
                     ):
                    keep = '2'
                    dec_comment = 'Record 2 (%s) is UPDATE of record 1 (%s). Automatically choose 2.'%(id_2, id_1)
                    print(dec_comment)
                    dec_type='AUTO: UPDATE'
                elif (remove_identicals & metadata_identical & sites_identical & URL_identical & (data_identical|correlation_perfect)):
                    # if all metadata and data matches, choose record according to hierarchy of databases
                    if data_identical:
                        dec_comment = 'RECORDS IDENTICAL (identical data). Automatically choose #%s.'%auto_choice
                    else:
                        dec_comment = 'RECORDS IDENTICAL (perfect correlation). Automatically choose #%s.'%auto_choice
                    
                    print(dec_comment)
                    keep = auto_choice#'1'
                    dec_type='AUTO: IDENTICAL'
                elif (remove_identicals & (data_identical|correlation_perfect)):
                    # if most metadata and data matches except URL and/or site name, choose record according to hierarchy of databases
                    if data_identical:
                        dec_comment = 'RECORDS IDENTICAL (identical data) except for metadata. Automatically choose #%s.'%auto_choice
                    else:
                        dec_comment = 'RECORDS IDENTICAL (perfect correlation) except for metadata. Automatically choose #%s.'%auto_choice
                        
                    print(dec_comment)
                    keep = auto_choice 
                    dec_type='AUTO: IDENTICAL except for URLs and/or geo_siteName.'
                elif auto_db_choice_cond & metadata_identical:
                    reason = automate_db_choice['reason']
                    dec_comment = f'Automated choice. Metadata identical, automatically choose {auto_db_pref} over {auto_db_rej}. {reason}'
                        
                    print(dec_comment)
                    if or_db_1==auto_db_pref:
                        keep     =  '1' 
                    elif or_db_2==auto_db_pref:
                        keep     =  '2' 
                    else:
                        raise Exception('Auto condition error.')
                    dec_type = 'AUTO: preferred db and metadata identical.'
                    
                else:
                    
                    dec_type = 'MANUAL'
                    fig, dup_mdata_row = dup_plot(df, ii, jj, id_1, id_2, time_1, time_2, time_12, data_1, data_2, int_1, int_2, pot_dup_corrs[i_pot_dups])
                    plt.show(block=False)
                    print('**Decision required for this duplicate pair (see figure above).**')

                    if (id_1 in multiple_list) | (id_2 in multiple_list):
                        print('-------'*15)
                        print('***ATTENTION*** THIS RECORD IS ASSOCIATED WITH MULTIPLE DUPLICATES! PLEASE PAY SPECIAL ATTENTION WHEN MAKING DECISIONS FOR THIS RECORD!')
                        print('The potential duplicates also associated with this record are:')
                        if id_1!=reverse_dict[id_1]:
                            print('- ', reverse_dict[id_1])
                        for m_id in dup_dict_multiples[reverse_dict[id_1]]:
                            if m_id==id_1: continue
                            if m_id==id_2: continue
                            print('......'*10)


                            
                            r  = df[df['datasetId']==m_id].iloc[0]  # Get first row as Series
                            mm = df[df['datasetId']==m_id].index.values[0]



                            print(f"     - {'Dataset ID':20s}: {r['datasetId']}")
                            print(f"     - {'URL':20s}: {r['originalDataURL']}")

                            
                            # Prepare the text info
                            info_text = (f"{r['datasetId']} \n {r['geo_siteName']} ({r['geo_meanLat']:.1f}, {r['geo_meanLon']:.1f}, {r['geo_meanElev']:.1f}) | "
                                         f"{r['archiveType']} | {r['paleoData_proxy']} | {r['paleoData_variableName']}")
                            
                            # Check for previous duplicates
                            prev_dup_text = ""
                            if len(dup_dec) > 1:
                                if str(mm) in np.array(dup_dec)[:, 0]:
                                    mm_loc = np.array(dup_dec)[:, 0] == str(mm)
                                    prev_dup_text = f" \n DECISION: {np.array(dup_dec)[mm_loc, -4][0]} (detected as pot dup of: {np.array(dup_dec)[mm_loc, 4][0]})"
                                elif str(mm) in np.array(dup_dec)[:, 1]:
                                    mm_loc = np.array(dup_dec)[:, 1] == str(mm)
                                    prev_dup_text = f" \n DECISION: {np.array(dup_dec)[mm_loc, -3][0]}  (detected as pot dup of: {np.array(dup_dec)[mm_loc, 3][0]})"


                            def line_11(x,y):
                                if (len(x)>0) & (len(y)>0):
                                    min_val = min(x.min(), y.min())
                                    max_val = max(x.max(), y.max())
                                    plt.plot([min_val, max_val], [min_val, max_val],color='tab:grey', alpha=0.75, zorder=0, label='1:1 line')
                                return
                            
                            # Create figure with info as title
                            mt, mi1, mi2 = np.intersect1d(df['year'].loc[ii], df['year'].loc[mm], return_indices=True) 
                            fig = plt.figure(figsize=(4*2, 2), dpi=120)
                            fig.suptitle(info_text + prev_dup_text, fontsize=8, y=0.98)



                            
                            # First subplot: 
                            plt.subplot(141)
                            x = np.array(df['paleoData_values'].loc[mm])[mi2]
                            y = np.array(df['paleoData_values'].loc[ii])[mi1]
                            plt.scatter(x, y, s=5)
                            plt.ylabel(f'{m_id[:20]}\n{m_id[20:40]}\n{m_id[40:]}'.replace('\n\n',''), fontsize=8)
                            plt.xlabel(f'{id_1[:20]}\n{id_1[20:40]}\n{id_1[40:]}'.replace('\n\n',''), fontsize=8)
                            plt.tick_params(labelsize=7)
                            line_11(x,y)
                            
                            # Second subplot
                            mt, mi1, mi2 = np.intersect1d(df['year'].loc[jj], df['year'].loc[mm], return_indices=True) 
                            plt.subplot(142)
                            x = np.array(df['paleoData_values'].loc[mm])[mi2]
                            y = np.array(df['paleoData_values'].loc[jj])[mi1]
                            plt.scatter(x, y, s=5)
                            plt.ylabel(f'{m_id[:20]}\n{m_id[20:40]}\n{m_id[40:]}'.replace('\n\n',''), fontsize=8)
                            plt.xlabel(f'{id_2[:20]}\n{id_2[20:40]}\n{id_2[40:]}'.replace('\n\n',''), fontsize=8)
                            plt.tick_params(labelsize=7)
                            line_11(x,y)


                            # Third subplot
                            plt.subplot(143)
                            plt.scatter(df['year'].loc[ii], df['paleoData_values'].loc[ii], label=df['datasetId'].loc[ii],
                                       edgecolor='tab:blue', marker='o', s=6, facecolor='None', lw=0.7)
                            plt.scatter(df['year'].loc[mm], df['paleoData_values'].loc[mm], label=df['datasetId'].loc[mm], 
                                        color='k', marker='o', zorder=0, alpha=0.9, s=3)
                            plt.ylabel('year', fontsize=8)
                            plt.xlabel('paleoData_values', fontsize=8)
                            plt.tick_params(labelsize=7)
                            plt.legend(fontsize=8)
                            
                            # Fourth subplot
                            plt.subplot(144)
                            plt.scatter(df['year'].loc[jj], df['paleoData_values'].loc[jj], label=df['datasetId'].loc[jj],
                                       color='tab:red', marker='x', s=6,lw=.7)
                            plt.scatter(df['year'].loc[mm], df['paleoData_values'].loc[mm], label=df['datasetId'].loc[mm], 
                                       color='k', marker='o', zorder=0, alpha=0.9, s=3)
                            plt.ylabel('year', fontsize=8)
                            plt.xlabel('paleoData_values', fontsize=8)
                            plt.tick_params(labelsize=7)
                            plt.legend(fontsize=8)


                            plt.tight_layout()  # Leave room for suptitle
                            plt.show(block=True)



                        print('-------'*15)
                        
                    if comment:
                        print('Before inputting your decision. Would you like to leave a comment on your decision process?')
                        dec_comment = input(f'{i_pot_dups+1}/{n_pot_dups}: **COMMENT** Please type your comment here and/or press enter.')
                    else:
                        dec_comment = ''
                    keep = input(f'{i_pot_dups+1}/{n_pot_dups}: **DECISION** Keep record 1 (%s, blue circles) [1], record 2 (%s, red crosses) [2], keep both [b], keep none [n] or create a composite of both records [c]? Note: only overlapping timesteps are being composited. [Type 1/2/b/n/c]:'%(id_1, id_2))

                
                
                    save_fig(fig, '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+'_%d_%d'%(ii,jj), dir=f'/dup_detection/{df.name}', figformat='jpg')
        
                    
                    figpath    = 'https://nzero.umd.edu:444/hub/user-redirect/lab/tree/dod2k_v2.0/figs/dup_detection/%s'%df.name
                    figpath  += '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+'_%d_%d,jpg'%(ii,jj)
                    
                # now write down the decision
                if keep=='1':
                    print('KEEP BLUE CIRCLES: keep %s, remove %s.'%(id_1, id_2))
                    decisions[ii]='KEEP'
                    decisions[jj]='REMOVE'
                elif keep=='2':
                    print('KEEP RED CROSSES: remove %s, keep %s.'%(id_1, id_2))
                    decisions[jj]='KEEP'
                    decisions[ii]='REMOVE'
                elif keep=='n':
                    print('REMOVE BOTH: remove %s, remove %s.'%(id_2, id_1))
                    decisions[ii]='REMOVE'
                    decisions[jj]='REMOVE'
                elif keep=='b':
                    print('KEEP BOTH: keep %s, keep %s.'%(id_1, id_2))
                    decisions[ii]='KEEP'
                    decisions[jj]='KEEP'
                elif keep=='c':
                    print('CREATE A COMPOSITE OF BOTH RECORDS: %s, %s.'%(id_1, id_2))
                    decisions[ii]='COMPOSITE'
                    decisions[jj]='COMPOSITE'


            
            cand_pair =[[ii, jj, figpath,
                        id_1, id_2, or_db_1, or_db_2, site_1, site_2,
                        lat_1, lat_2, lon_1, lon_2, 
                        df['geo_meanElev'].loc[ii], df['geo_meanElev'].loc[jj],
                        df['archiveType'].loc[ii], df['archiveType'].loc[jj],
                        df['paleoData_proxy'].loc[ii], df['paleoData_proxy'].loc[jj],
                        df['originalDataURL'].loc[ii], df['originalDataURL'].loc[jj],
                        '%3.1f-%3.1f'%(np.min(df['year'].loc[ii]), np.max(df['year'].loc[ii])), 
                        '%3.1f-%3.1f'%(np.min(df['year'].loc[jj]), np.max(df['year'].loc[jj])), 
                        decisions[ii], decisions[jj],
                        dec_type, dec_comment]]

            dup_dec  += cand_pair
            writer.writerows(cand_pair)
            print('write decision to backup file')
            # raise Exception
        
    print('=====================================================================')
    print('END OF DUPLICATE DECISION PROCESS.')
    print('=====================================================================')

    comment = '# '+input('Type your comment on your decision process here and/or press enter:')
    header  += [[comment]]
    # header  += cols
    
    print(np.array(dup_dec).shape)
    filename +=f'_{date_time[2:10]}'
    write_csv(np.array(dup_dec), dirname+filename, header=header, cols=cols)
    print('Saved the decisions under %s.csv'%(dirname+filename))
    
    print('Summary of all decisions made:')
    for ii_d in range(len(dup_dec)):
        keep_1=dup_dec[ii_d][-4]
        keep_2=dup_dec[ii_d][-3]
        print('#%d: %s record %s. %s record %s.'%(ii_d, keep_1, dup_dec[ii_d][3], 
                                                  keep_2, dup_dec[ii_d][4]))
    return 


def define_hierarchy(df, hierarchy='default'):
    #define hierarchy, 
    df['Hierarchy'] = 0 
    if hierarchy=='default':
        df.loc[df['originalDatabase']=='PAGES2k v2.2.0', 'Hierarchy'] = 5
        df.loc[df['originalDatabase']=='FE23 (Breitenmoser et al. (2014))', 'Hierarchy'] = 4
        df.loc[df['originalDatabase']=='CoralHydro2k v1.0.1', 'Hierarchy'] = 2
        df.loc[df['originalDatabase']=='Iso2k v1.1.2', 'Hierarchy'] = 3
        df.loc[df['originalDatabase']=='SISAL v3', 'Hierarchy'] = 1
    else:
        df.loc[df['originalDatabase']=='PAGES2k v2.2.0', 'Hierarchy'] = hierarchy['pages2k']
        df.loc[df['originalDatabase']=='FE23 (Breitenmoser et al. (2014))', 'Hierarchy'] = hierarchy['fe23']
        df.loc[df['originalDatabase']=='CoralHydro2k v1.0.1', 'Hierarchy'] = hierarchy['ch2k']
        df.loc[df['originalDatabase']=='Iso2k v1.1.2', 'Hierarchy'] = hierarchy['iso2k']
        df.loc[df['originalDatabase']=='SISAL v3', 'Hierarchy'] = hierarchy['sisal']
        

    return df

# def duplicate_decisions_old(df, operator_details=False, choose_recollection=True, #keep_all=False, 
#                         plot=True, remove_identicals=True, dist_tolerance_km=8):
#     """
#     Review potential duplicate pairs in a proxy database and decide which records to keep, remove, or combine.

#     This function walks through each potential duplicate pair identified in a dataset,
#     displays metadata and optionally plots the data, and allows the operator to make decisions.
#     Decisions are saved to a CSV file, and a duplicate-free dataframe can be generated.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         The dataframe containing proxy data and metadata. Expected columns include:
#             - 'geo_meanLat', 'geo_meanLon' : Latitude and longitude of the site.
#             - 'year', 'paleoData_values' : Time vector and proxy values.
#             - 'archiveType', 'paleoData_proxy' : Archive type and proxy type.
#             - 'datasetId' : Unique identifier or dataset name.
#             - 'geo_siteName' : Site name.
#             - 'originalDatabase' : Name of the original database.
#             - 'geo_meanElev' : Mean elevation of the site (optional for duplicate checks).
#             - 'Hierarchy' : Numeric value representing dataset priority (used for auto-decisions).
#             - 'originalDataURL' : URL of the original dataset.

#     operator_details : tuple or bool, optional
#         Tuple containing (initials, fullname, email) of the operator. If False, user input is requested.
#         Default is False.
#     choose_recollection : bool, optional
#         If True, automatically selects the record that is a recollection or update when applicable.
#         Default is True.
#     plot : bool, optional
#         If True, generate plots for manual inspection of duplicate pairs. Default is True.
#     remove_identicals : bool, optional
#         If True, automatically remove records that are identical in data and metadata. Default is True.
#     dist_tolerance_km : float, optional
#         Maximum distance (in km) for considering records as recollection updates. Default is 8 km.

#     Returns
#     -------
#     None
#         Decisions are saved as CSV backup and final CSV in the `df.name/dup_detection/` directory.

#     Notes
#     -----
#     - Automatic decisions are made based on data identity, metadata identity, perfect correlation,
#       and recollection indicators in site names.
#     - Manual decisions are prompted via command-line input if automatic rules do not apply.
#     - Decision types include:
#         - 'AUTO: UPDATE' : Automatically select record that is a recollection/update.
#         - 'AUTO: IDENTICAL' : Automatically select record based on hierarchy if records are identical.
#         - 'MANUAL' : Decision requires operator input.
#     - Figures are saved with a standardized naming convention and linked in the CSV output.
#     """

#     import datetime
#     # Select the metadata keys to show on the output figures - keys depend on the 
#     # dataframe and need to be determined according to the dataframe input. 
#     # if using this code on fused_database.pkl, make sure that all entries are available 
#     # in the dataframe1 Use fuse_datasets.py to fuse and homogenise metadata  of different databases.

#     if not operator_details:
#         initials = input('Please enter your initials here:')
#         fullname = input('Please enter your full name here:')
#         email    = input('Please enter your email address here:')
#     else:
#         initials, fullname, email = operator_details
#     date_time= str(datetime.datetime.utcnow())+' (UTC)'
#     header   = [['# Decisions for duplicate candidate pairs. ']]
#     header  += [['# Operated by %s (%s)'%(fullname, initials)]]
#     header  += [['# E-Mail: %s'%email]]
#     header  += [['# Created on: %s'%date_time]]

    
#     dirname = 'data/%s/dup_detection/'%(df.name)
#     filename = f'dup_decisions_{df.name}' # name of csv file which saves duplicate candidate pairs
    
#     filename+=f'_{initials}'
    
#     detection_file = dirname+'dup_detection_candidates_'+df.name
    
#     if not os.path.exists(dirname):
#         os.makedirs(dirname)

#     # try to load data from backup!
#     try:
#         data, hh = read_csv(dirname+filename+'_BACKUP', header=True, last_header_row=4)
#         backup_file=input(f'Found backup file ({dirname+filename}_BACKUP.csv). Do you want to start decision process from the backup file? [y/n]')
#         if backup_file=='n':
#             raise FileNotFoundError
#         data, hh = read_csv(dirname+filename+'_BACKUP', header=True, last_header_row=4)
#         print('header', hh)
#         print('data', list(data))
#         data = list(data)
        
#         load_saved_data      = True
#         last_index = len(data)
#         print('start with index: ', last_index)
#     except FileNotFoundError:
#         print('No back up.')
#         load_saved_data      = False
#         data = []
#         pass

#     # load the potential duplicate data as found in find_duplicates function
#     pot_dup_meta, head = read_csv(detection_file, header=True)

#     pot_dup_inds   = np.array(np.array(pot_dup_meta)[:, 2], dtype=int)
#     pot_dup_corrs  = np.array(np.array(pot_dup_meta)[:, 4], dtype=float)
#     pot_dup_dists  = np.array(np.array(pot_dup_meta)[:, 5], dtype=float)

#     n_pot_dups   = pot_dup_inds.shape[0]
    
#     # loop through the potential duplicates.
#     decisions = ['KEEP']*df.shape[0] #set False if index should be discarded from dataset.
#     cols    = [['index 1', 'index 2', 'figure path',
#                 'datasetId 1', 'datasetId 2', 
#                 'originalDatabase 1', 'originalDatabase 2',
#                 'geo_siteName 1', 'geo_siteName 2', 
#                 'geo_meanLat 1', 'geo_meanLat 2', 
#                 'geo_meanLon 1', 'geo_meanLon 2', 
#                 'geo_meanElevation 1', 'geo_meanElevation 2', 
#                 'archiveType 1', 'archiveType 2',
#                 'paleoData_proxy 1', 'paleoData_proxy 2',
#                 'originalDataURL 1', 'originalDataURL 2',
#                 'year 1', 'year 2',
#                 'Decision 1', 'Decision 2',
#                 'Decision type', 'Decision comment' ]]
#     dup_dec = []
    
#     with open(dirname+filename+'_BACKUP.csv', 'w', newline='') as f:
        
#         writer = csv.writer(f)
#         writer.writerows(header)
#         print('header for backup', header)
#         writer.writerows(cols)
#         print('cols for backup', cols)
#         for data_row in list(data):
#             writer.writerow(list(data_row))
#             print('list(data_row)', list(data_row))
#             dup_dec+=[list(data_row)]
#         print('data for backup', data)
    
#         for i_pot_dups, (ii, jj) in enumerate(pot_dup_inds):
#             if load_saved_data:
#                 if i_pot_dups<last_index: continue
#                 else:
#                     print(ii, jj, last_index, i_pot_dups)
#             # data and metadata associated with the two potential duplicates
#             id_1, id_2       = df['datasetId'].iloc[[ii, jj]]
#             time_1, time_2   = df['year'].iloc[[ii, jj]]
#             time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) 
#             data_1           = np.array(df['paleoData_values'].iloc[ii])
#             data_2           = np.array(df['paleoData_values'].iloc[jj])
            
#             lat_1, lat_2     = df['geo_meanLat'].iloc[[ii, jj]]
#             lon_1, lon_2     = df['geo_meanLon'].iloc[[ii, jj]]
#             site_1, site_2   = df['geo_siteName'].iloc[[ii, jj]]
#             or_db_1, or_db_2 = df['originalDatabase'].iloc[[ii, jj]]

#             auto_choice = '1' if df['Hierarchy'].iloc[ii]>=df['Hierarchy'].iloc[jj] else '2'
            
#             print('> %d/%d'%(i_pot_dups+1, n_pot_dups), #set_txt, 
#                    id_1, id_2, #var_1, var_2, season_1,season_2,
#                    pot_dup_dists[i_pot_dups], pot_dup_corrs[i_pot_dups],
#                    sep=',')
        
#             # Print the interpretation values 
#             print('====================================================================')
#             print('=== POTENTIAL DUPLICATE %d/%d'%(i_pot_dups, n_pot_dups)+': %s+%s ==='%(df['datasetId'].iloc[ii], df['datasetId'].iloc[jj]))
#             print('=== URL 1: %s   ==='%(df['originalDataURL'].iloc[ii]))
#             print('=== URL 2: %s   ==='%(df['originalDataURL'].iloc[jj]))
            
#             keep = ''
#             dec_comment = ''

            
#             elevation_nan = (np.isnan(df['geo_meanElev'].iloc[ii])|
#                              np.isnan(df['geo_meanElev'].iloc[jj]))
#             # (np.round(lat_1, 1)==np.round(lat_2, 1)) & 
#             #                       (np.round(lon_1, 1)==np.round(lon_2, 1)) 
#             metadata_identical = ((np.abs(lat_1-lat_2)<=0.1) & 
#                                   (np.abs(lon_1-lon_2)<=0.1)  & 
#                                   ((np.abs(df['geo_meanElev'].iloc[ii]-df['geo_meanElev'].iloc[jj])<=1) 
#                                     | elevation_nan) & 
#                                   (df['archiveType'].iloc[ii]==df['archiveType'].iloc[jj]) & 
#                                   (df['paleoData_proxy'].iloc[ii]==df['paleoData_proxy'].iloc[jj]) 
#                                  )
#             sites_identical     = site_1==site_2
#             URL_identical       = (df['originalDataURL'].iloc[ii]==df['originalDataURL'].iloc[jj]) 
#             # print(data_1, data_2)
#             data_identical      = (list(data_1)==list(data_2)) & (list(time_1)==list(time_2))
#             correlation_perfect = (True if pot_dup_corrs[i_pot_dups]>=0.99 else False) & (len(time_1)==len(time_2))
            
#             print('metadata_identical: ', metadata_identical)
#             print('lat', (np.abs(lat_1-lat_2)<=0.1), 'lon' , (np.abs(lon_1-lon_2)<=0.1)   ,'elevation',
#                                   ( (np.abs(df['geo_meanElev'].iloc[ii]-df['geo_meanElev'].iloc[jj])<=1) 
#                                     | elevation_nan) , 'archivetype',
#                                   (df['archiveType'].iloc[ii]==df['archiveType'].iloc[jj]) ,'paleodata_proxy',
#                                   (df['paleoData_proxy'].iloc[ii]==df['paleoData_proxy'].iloc[jj]) 
#                                  )
#             print('sites_identical: ', sites_identical)
#             print('URL_identical: ', URL_identical)
#             print('data_identical: ', data_identical)
#             print('correlation_perfect: ', correlation_perfect)
            
#             while keep not in ['1', '2', 'b', 'n', 'c']:
#                 # go through the data and metadata and check if decision is =automatic or manual
#                 if (choose_recollection 
#                     # if record 1 is update of record 2, choose 1
#                       & np.any([(ss in site_1.lower()) for ss in ['recollect', 'update', 're-collect']]) 
#                       & (pot_dup_dists[i_pot_dups]<dist_tolerance_km)
#                       & ~np.any([ss in site_2.lower() for ss in ['recollect', 'update', 're-collect']])
#                      ):
#                     keep = '1'
#                     dec_comment = 'Record 1 (%s) is UPDATE of record 2(%s) . Automatically choose 1.'%(id_1, id_2)
#                     print(dec_comment)
#                     dec_type='AUTO: UPDATE'
#                 elif (choose_recollection 
#                       # if record 2 is update of record 1, choose 2
#                       & np.any([ss in site_2.lower() for ss in ['recollect', 'update', 're-collect']]) 
#                       & (pot_dup_dists[i_pot_dups]<dist_tolerance_km)
#                       & ~np.any([ss in site_1.lower() for ss in ['recollect', 'update', 're-collect']])
#                      ):
#                     keep = '2'
#                     dec_comment = 'Record 2 (%s) is UPDATE of record 1 (%s). Automatically choose 2.'%(id_2, id_1)
#                     print(dec_comment)
#                     dec_type='AUTO: UPDATE'
#                 elif (remove_identicals & metadata_identical & sites_identical & URL_identical & (data_identical|correlation_perfect)):
#                     # if all metadata and data matches, choose record according to hierarchy of databases
#                     if data_identical:
#                         dec_comment = 'RECORDS IDENTICAL (identical data). Automatically choose #%s.'%auto_choice
#                     else:
#                         dec_comment = 'RECORDS IDENTICAL (perfect correlation). Automatically choose #%s.'%auto_choice
                    
#                     print(dec_comment)
#                     keep = auto_choice#'1'
#                     dec_type='AUTO: IDENTICAL'
#                 elif (remove_identicals & metadata_identical & (data_identical|correlation_perfect)):
#                     # if most metadata and data matches except URL and/or site name, choose record according to hierarchy of databases
#                     if data_identical:
#                         dec_comment = 'RECORDS IDENTICAL (identical data) except for URLs and/or site Name. Automatically choose #%s.'%auto_choice
#                     else:
#                         dec_comment = 'RECORDS IDENTICAL (perfect correlation) except for URLs and/or site Name. Automatically choose #%s.'%auto_choice
                        
#                     print(dec_comment)
#                     keep = auto_choice#'1'
#                     dec_type='AUTO: IDENTICAL except for URLs and/or geo_siteName.'
                    
#                 else:
#                     # if keep_all:
#                     #     keep='b'
#                     #     dec_type='AUTO: KEEP ALL'
#                     # else:
#                     dec_type = 'MANUAL'
#                     fig, dup_mdata_row = dup_plot(df, ii, jj, id_1, id_2, time_1, time_2, time_12, data_1, data_2, int_1, int_2, pot_dup_corrs[i_pot_dups])
#                     plt.show(block=False)
#                     print('**Decision required for this duplicate pair (see figure above).**')
#                     print('Before inputting your decision. Would you like to leave a comment on your decision process?')
#                     dec_comment = input(' Please type your comment here and/or press enter.')
#                     keep = input('Keep record 1 (%s, blue circles) [1], record 2 (%s, red crosses) [2], keep both [b], keep none [n] or create a composite of both records [c]?  [Type 1/2/b/n/c]:'%(id_1, id_2))
                        
#                 # now write down the decision
#                 if keep=='1':
#                     print('KEEP BLUE: keep %s, remove %s.'%(id_1, id_2))
#                     decisions[jj]='REMOVE'
#                 elif keep=='2':
#                     print('KEEP BLACK: remove %s, keep %s.'%(id_1, id_2))
#                     decisions[ii]='REMOVE'
#                 elif keep=='n':
#                     print('REMOVE BOTH: remove %s, remove %s.'%(id_2, id_1))
#                     decisions[ii]='REMOVE'
#                     decisions[jj]='REMOVE'
#                 elif keep=='b':
#                     print('KEEP BOTH: keep %s, keep %s.'%(id_1, id_2))
#                 elif keep=='c':
#                     print('CREATE A COMPOSITE OF BOTH RECORDS: %s, %s.'%(id_1, id_2))
#                     decisions[ii]='COMPOSITE'
#                     decisions[jj]='COMPOSITE'
                    
#             save_fig(fig, '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+'_%d_%d.jpg'%(ii,jj), dir=f'/dup_detection/{df.name}')

            
#             figpath    = 'https://nzero.umd.edu:444/hub/user-redirect/lab/tree/dod2k_v2.0/figs/dup_detection/%s'%df.name
#             figpath  += '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+'_%d_%d.jpg'%(ii,jj)
#             cand_pair =[[ii, jj, figpath,
#                         id_1, id_2, or_db_1, or_db_2, site_1, site_2,
#                         lat_1, lat_2, lon_1, lon_2, 
#                         df['geo_meanElev'].iloc[ii], df['geo_meanElev'].iloc[jj],
#                         df['archiveType'].iloc[ii], df['archiveType'].iloc[jj],
#                         df['paleoData_proxy'].iloc[ii], df['paleoData_proxy'].iloc[jj],
#                         df['originalDataURL'].iloc[ii], df['originalDataURL'].iloc[jj],
#                         '%3.1f-%3.1f'%(np.min(df['year'].iloc[ii]), np.max(df['year'].iloc[ii])), 
#                         '%3.1f-%3.1f'%(np.min(df['year'].iloc[jj]), np.max(df['year'].iloc[jj])), 
#                         decisions[ii], decisions[jj],
#                         dec_type, dec_comment]]
#             dup_dec  += cand_pair
#             writer.writerows(cand_pair)
#             print('write decision to backup file')
#             # raise Exception
        
#     print('=====================================================================')
#     print('END OF DUPLICATE DECISION PROCESS.')
#     print('=====================================================================')

#     comment = '# '+input('Type your comment on your decision process here and/or press enter:')
#     header  += [[comment]]
#     # header  += cols
    
#     print(np.array(dup_dec).shape)
#     filename +=f'_{date_time[2:10]}'
#     write_csv(np.array(dup_dec), dirname+filename, header=header, cols=cols)
#     print('Saved the decisions under %s.csv'%(dirname+filename))
    
#     print('Summary of all decisions made:')
#     for ii_d in range(len(dup_dec)):
#         keep_1=dup_dec[ii_d][-4]
#         keep_2=dup_dec[ii_d][-3]
#         print('#%d: %s record %s. %s record %s.'%(ii_d, keep_1, dup_dec[ii_d][3], 
#                                                   keep_2, dup_dec[ii_d][4]))
#     return 

def join_composites_metadata(df, comp_ID_pairs, df_decisions, header):
    """
    Create composite records from overlapping duplicate proxy datasets and generate metadata.

    This function combines pairs of proxy records that were identified as duplicates
    and decided to be composited. It standardizes the data as z-scores, averages overlapping 
    periods, merges metadata, and generates a composite DataFrame. A scatter plot of the 
    original and composite data is created and saved for visual inspection.

    Parameters
    ----------
    df : pd.DataFrame
        Original DataFrame containing proxy data and metadata. Must include columns:
            - 'paleoData_values' : proxy values
            - 'year' : time vector
            - 'geo_meanLat', 'geo_meanLon', 'geo_meanElev' : site coordinates
            - 'archiveType', 'geo_siteName', 'paleoData_proxy' : metadata
            - 'climateInterpretation_variable', 'dataSetName', 'originalDatabase', 'originalDataURL', 'paleoData_notes', 'duplicateDetails'
            - 'datasetId' : unique record identifier

    comp_ID_pairs : pd.DataFrame
        DataFrame listing pairs of record IDs to be composited. Must include columns:
            - 'datasetId 1', 'datasetId 2'
            - 'originalDatabase 1', 'originalDatabase 2'
            - 'Decision type', 'Decision comment'

    df_decisions : pd.DataFrame
        DataFrame containing the decisions made during duplicate evaluation.
        Used to annotate the composite metadata with decision type and comments.

    header : list
        Metadata header from the duplicate decision process. Used for documenting
        operator details in the composite notes.

    Returns
    -------
    df_comp : pd.DataFrame
        A new DataFrame containing the composited proxy records, including:
            - Combined 'paleoData_values' as z-scores
            - Merged 'year' vector
            - Updated metadata fields, including a composite 'datasetId'
            - Detailed 'duplicateDetails' recording the composition process

    Notes
    -----
    - For numerical metadata that differs between records, the mean is taken.
    - For categorical metadata, entries are concatenated into a composite string.
    - Overlapping periods are averaged, and non-overlapping periods are appended.
    - A scatter plot is generated for each composite showing the original records
      and the resulting composite, and it is saved using `save_fig`.
    - The function maintains provenance of original datasets, including notes and URLs,
      in the 'duplicateDetails' field.
    """
    
    # create composite dataframe
    df_comp = pd.DataFrame()
    
    for ii in comp_ID_pairs.index:

        # if ii not in df.datasetId.values: 
        #     print(ii)
        #     print(df.datasetId.values)
        #     print(f'Skip {ii} - allowed for testing only!!')
        #     continue
        
        row = {}
        ID_1, ID_2   = comp_ID_pairs.loc[ii, ['datasetId 1', 'datasetId 2']]
        db_1, db_2   = comp_ID_pairs.loc[ii, ['originalDatabase 1', 'originalDatabase 2']]
        dec_1, dec_2 = comp_ID_pairs.loc[ii, ['Decision 1', 'Decision 2']]
        print(ID_1, ID_2)
    
        # metadata should match exactly for the following columns (check), if not choose metadata values:
        add_dup_note = ''
        for key in ['archiveType', 'geo_meanElev', 'geo_meanLat', 'geo_meanLon',
                    'geo_siteName', 'paleoData_proxy', 'yearUnits', 'interpretation_variable', 'interpretation_direction', 'interpretation_seasonality']:
            
            md1, md2 = df.at[ID_1, key], df.at[ID_2, key]

            if isinstance(md1, (float, np.floating, np.float32, np.float64)):
                md1 = np.round(md1, 3)
                
            if isinstance(md2, (float, np.floating, np.float32, np.float64)):
                md2 = np.round(md2, 3)
                
            if md1==md2:
                row[key]=md1
                add_dup_note += f'{key}: Metadata identical'
            else:
                print('--------------------------------------------------------------------------------')
                add_dup_note = 'Metadata differs for %s in original records: %s (%s) and %s (%s). '%(key, ID_1, str(df.at[ID_1, key]), 
                                                                                 ID_2, str(df.at[ID_2, key]))
                print('Metadata different for >>>%s<<< in: %s (%s) and %s (%s). '%(key, ID_1, str(df.at[ID_1, key]), 
                                                                                 ID_2, str(df.at[ID_2, key])))
                try:
                    entry='COMPOSITE: '+df.at[ID_1, key]+' + '+df.at[ID_2, key]
                    add_dup_note += f'{key}: Metadata composited to: '+entry
                    if key in ['interpretation_variable', 'interpretation_direction', 'interpretation_seasonality']:
                        if df.at[ID_1, key]=='N/A': entry=df.at[ID_2, key]
                        if df.at[ID_2, key]=='N/A': entry=df.at[ID_1, key]
                        if (df.at[ID_1, key] in df.at[ID_2, key]):  entry=df.at[ID_2, key]
                        if (df.at[ID_2, key] in df.at[ID_1, key]):  entry=df.at[ID_1, key]
                    loop=False
                except TypeError:
                    # print('Can not create composites for numerical metadata! Create average instead.')
                    # av = input('Type [y] if you want to average the metadata. Otherwise type [n].')
                    # if av.lower() in ['y', 'yes']:
                    entry = np.mean([df.at[ID_1, key], df.at[ID_2, key]])                        
                    add_dup_note += f'{key}: Metadata averaged to: '+str(entry)
                    loop=False
                row[key]= entry
            # print('Add the following note to duplicateDetails:', add_dup_note)
        # create new entries for the following columns:
        # create z-scores of data
        data_1, data_2 = np.array(df.at[ID_1, 'paleoData_values']), np.array(df.at[ID_2, 'paleoData_values'])
    
        # year
        time_1, time_2 = np.array(df.at[ID_1, 'year']), np.array(df.at[ID_2, 'year'])
        time_12, ii_1, ii_2 = np.intersect1d(time_1, time_2, return_indices=True)
        ii_1x   = [ii for ii in range(len(time_1)) if ii not in ii_1]
        ii_2x   = [ii for ii in range(len(time_2)) if ii not in ii_2]
    
        data_1 /= np.std(data_1[ii_1])
        data_1 -= np.mean(data_1[ii_1])
        data_2 /= np.std(data_2[ii_2])
        data_2 -= np.mean(data_2[ii_2])
    
        data_12 = (data_1[ii_1]+data_2[ii_2])/2.
    
        data = list(data_1[ii_1x])+list(data_12)+list(data_2[ii_2x])
        time = list(time_1[ii_1x])+list(time_12)+list(time_2[ii_2x])
        
        row['paleoData_values'] = data
        row['year'] = time
    
        fig = plt.figure(figsize=(6, 2), dpi=100)
        
        plt.scatter(time_1, data_1, s=20, color='tab:blue', label=ID_1)
        plt.scatter(time_2, data_2, s=20, color='tab:orange', label=ID_2)
        plt.scatter(time, data, s=10, color='k', label='composite')
        plt.legend()
        plt.show()
        save_fig(fig, 'composite_%s_%s'%(ID_1, ID_2), dir='/%s/dup_detection/'%df.name)
        
        
        # new metadata for identification etc.
        row['dataSetName']      = df.at[ID_1, 'dataSetName']+', '+df.at[ID_2, 'dataSetName']
        row['originalDatabase'] = 'dod2k_composite_z'
        row['originalDataURL']  = '%s: %s, %s: %s'%(ID_1, df.at[ID_1, 'originalDataURL'], ID_2, df.at[ID_2, 'originalDataURL'])
        row['paleoData_notes']  = '%s: %s, %s: %s'%(ID_1, df.at[ID_1, 'paleoData_notes'], ID_2, df.at[ID_2, 'paleoData_notes'])
        row['interpretation_variableDetail']  = '%s: %s, %s: %s'%(ID_1, df.at[ID_1, 'interpretation_variableDetail'], ID_2, df.at[ID_2, 'interpretation_variableDetail'])
        row['datasetId']        = 'dod2k_composite_z_'+ID_1.replace('dod2k_composite_z_','')+'_'+ID_2.replace('dod2k_composite_z_','')   
        row['paleoData_units']  = 'z-scores'


        # save details on duplicates in duplicateDetails:
        row['duplicateDetails']={'duplicate ID': f'{ID_1} and {ID_2}',
                                 'duplicate database': f'{db_1} and {db_2}',
                                 'duplicate decision': 'COMPOSITE',
                                 'decision type': df_decisions.loc[ii, 'Decision type']}
        
    
        # save details on composition process in duplicateDetails:
        row['duplicateDetails']['composite details'] = add_dup_note
        
    
        # save operator details
        if comp_ID_pairs.at[ii, 'Decision type']=='MANUAL': 
            operator_details = ' '.join(header[1:]).replace(' Modified ','')[:-2].replace(':','').replace('  E-Mail', '')
            row['duplicateDetails']['operator'] = operator_details
            row['duplicateDetails']['note'] = comp_ID_pairs.at[ii, 'Decision comment']

            

        # migrate original duplicate details
        
        row['duplicateDetails']['original duplicate details'] = {ID_1: df.at[ID_1, 'duplicateDetails'], ID_2: df.at[ID_2, 'duplicateDetails']}
            
        
        # create dataframe for composites
        df_comp = pd.concat([df_comp, pd.DataFrame({kk: [vv] for kk, vv in row.items()})], ignore_index = True, axis=0)
        
        # print(df_comp)
    return df_comp


def provide_dup_details(df_decisions, header):
    dup_details = {}
    dup_counts  = {}
    for ind in df_decisions.index:
        dec_1, dec_2 = df_decisions.loc[ind, ['Decision 1', 'Decision 2']]
        if dec_1=='KEEP' and dec_2=='KEEP': continue # in this case no true duplicates!
        
        id_1, id_2 = df_decisions.loc[ind, ['datasetId 1', 'datasetId 2']]
        db_1, db_2 = df_decisions.loc[ind, ['originalDatabase 1', 'originalDatabase 2']]
        for id in [id_1, id_2]:
            if id not in dup_details:
                dup_counts[id] = 0
                dup_details[id] = {}
            else:
                dup_counts[id]+=1
        dup_details[id_1][dup_counts[id_1]] = {'duplicate ID': id_2, 'duplicate database': db_2, 'duplicate decision': dec_2, 'decision type': df_decisions.loc[ind, 'Decision type']}
        dup_details[id_2][dup_counts[id_2]] = {'duplicate ID': id_1, 'duplicate database': db_1, 'duplicate decision': dec_1, 'decision type': df_decisions.loc[ind, 'Decision type']}
        
        if df_decisions.loc[ind, 'Decision type']=='MANUAL': 
            operator_details_str = ','.join(header[1:]).replace(' Modified ','')[:-2].replace(':','').replace('  E-Mail', '')
            for id in [id_1, id_2]:
                dup_details[id][dup_counts[id]]['operator'] = operator_details_str
                dup_details[id][dup_counts[id]]['note'] = df_decisions.loc[ind, 'Decision comment']
        else:
            for id in [id_1, id_2]:
                dup_details[id][dup_counts[id]]['operator'] = 'N/A'
                dup_details[id][dup_counts[id]]['note'] = 'N/A'       
    return dup_details
