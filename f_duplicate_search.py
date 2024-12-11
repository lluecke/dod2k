
#==============================================================================
# This script includes functions which search for duplicates. 
# Update 22/10/24 updated duplicate_decisions: 
#    - created backup decision file which is intermediately saved
#    - outputs URL which can be copied and pasted into browser
#    - implemented a composite option in the decision process, to create a composite of two records
#    - removed date (YY-MM-DD) of decision output filename
# Update 8/10/24 changed colours of figure in dup_plot (dropped highlighting differing metadata). 
#                 replaced db_name with df.name


# Update 2/10/24 Implemented a commenting option in duplicate_decisions to comment on decision process and on individual decisions.

# Update 27/9/24 Updated directory names and changed the correlation and distances output in find_duplicates to only output data from potential duplicates (replaced saving all pairs)

# Update 9/9/24 Introduced timestamps and contact details into duplicate decision output csv and changed the file= and dirnames for streamlining purposes.
# Update 23/8/24 Replaced the function cleanup_database and split into two: 
# plot_duplicates: plots the candidate pairs, saves figures and a summary csv sheet.
# cleanup_database_2: goes through the candidate pairs and makes decisions based on the options: a) raw input b) keep all records c) automatically keep only updated records and eliminate the other candidate. Decisions and metadata are saved in csv file.

# Update 22/8/24: Fixed a bug in find_duplicates logical algorithm- wrong bracket closure. Also changed location_crit to account for nans in elevation. Calculation of z-scores only divided by std if std!=0 to avoid nans.

# Update 15/8/24: updated numerical checks for duplicate detection: implemented check for correlation and rmse of records plus correlation and rmse of first difference. 

# Update 13/8/24: updated keys to account for updated dataframe terminology. Revised loading/saving of data in find_duplicates
#    find_duplicates:
#         updated the logic for duplicate detection:
#             overlap_crit now accounts for short records (allows short records to pass through without overlap check)
#             corr_crit only one numerical criterion needs to be satisfied for the data (either correlation or rmse or 1st difference)
#             location_crit now includes elevation too

#    cleanup_database:
#        updated plot to include URL, Database and streamlined table
# Script written by Lucie Luecke, 6/7/23

#==============================================================================

import numpy as np
import datetime
import matplotlib.pyplot as plt
import geopy.distance
import pandas as pd
import os
from matplotlib.gridspec import GridSpec as GS
import csv

def write_csv(data, ddir, filename, header=False):
    # writes data to csv
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    
    with open(ddir+filename+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        if header: writer.writerows(header)
        writer.writerows(data)
    return

def read_csv(filename, dtype=str, header=False, last_header_row=0):
    # reads csv file
    if header:
        header = []
        data   = []
        with open(filename+'.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for irow, row in enumerate(reader):
                if row[0].startswith('#') or irow<=last_header_row:
                    header.append(row[0].replace('#',''))
                else:
                    data.append(row)
        return np.array(data, dtype=dtype), header
    else:
        data = []
        with open(filename+'.csv', 'r', newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
        return np.array(data, dtype=dtype)
        


def figsave(fig, name, trans=False, add='/', fc='white',
            form='jpg'):
    """
    Saves figure *fig* as *name*, can also add directory via *add*

    """
    # from datetime import date
    # day = date.today().isoformat()
    # ddir = day+'/'+add if save_date else add
    ddir = add
    if not os.path.exists(os.getcwd()+'/'+ddir):
        os.makedirs(os.getcwd()+'/'+ddir)
    
    fig.savefig(os.getcwd()+'/'+ddir+'/'+name+'.'+form, 
                transparent=trans, facecolor=fig.get_facecolor(),
                format=form, bbox_inches='tight', pad_inches=0.0)
    if form=='pdf': 
        fig.savefig(os.getcwd()+'/'+ddir+'/'+name+'.jpg', 
                    transparent=trans, facecolor=fig.get_facecolor(),
                    format='jpg', dpi=100, bbox_inches='tight', pad_inches=0.0)
    print('saved figure in '+ ddir+'/'+name+'.'+form)
    plt.close()
    return

def find_duplicates(df, dist_tolerance_km=8, n_points_thresh=10, 
                    corr_thresh=0.9, rmse_thresh=0.1, corr_diff_thresh=0.9,
                    rmse_diff_thresh=0.1, elev_tolerance=0, ignore_same_database=False, 
                    save=True, print_output=False
                   ):
    """
    Searches for duplicates within DataFrame and saves potential duplicate 
    candidates (indices, IDs, correlations and distances)
    
    Parameters:
        df : pd.DataFrame object, holds the data and metadata.
             Define relevant keys below. Default is Pages2k keys.
             
        
        #define the threshold parameters:
        dist_tolerance_km  = 8 # geographical distance threshold (in km)
        n_points_thresh    = 10 # how many points should be shared, at least
        corr_thresh        = 0.9 # correlation threshold between timeseries
        rmse_thresh        = 0.1 # threshold for rmse
        rmse_diff_thresh   = 0.1 # threshold for rmse of first differemce
        corr_diff_thresh   = 0.9 # threshold for correlation of first differemce
        
        
        ignore_same_database = True or False. Set True if you want to 
                             ignore duplicates from the same database 
                             (useful for joint databases, after screening through 
                              the individual datasets and deciding that these
                              entries are not true duplicates)
    Returns: 
        - potential duplicate candidate's:
        - indices
        - IDs
        - correlations
        - distances
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

    ddir = '%s/dup_detection/'%(df.name)
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
            # time_2 = time_2_ma.data[~data_2_ma.mask]
            # data_2 = data_2_ma.data[~data_2_ma.mask]
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
            
            if time_12.shape[0]<=4: 
                #print('%d|%d no shared time'%(ii,jj))
                continue
            
            # calculate correlation, rms, sum of differences between shared i and j data
            z_1  = (data_1[int_1][:-1]-data_1[int_1][1:])
            z_1 -= np.mean(z_1)        
            if np.std(z_1)!=0: 
                z_1 /= np.std(z_1)
            else:
                # print('1. std=%3.1f'%np.std(z_1), ii, db_1, id_1)
                plt.figure()
                plt.scatter(time_1, data_1, label='data_1 (%s)'%id_1)
                plt.scatter(time_12, data_1[int_1], label='data_1 SHARED TIME (%s)'%id_1)
                plt.scatter(time_2, data_2, label='data_2 (%s)'%id_2)
                plt.scatter(time_12, data_2[int_2], label='data_2 SHARED TIME (%s)'%id_2)
                # print(np.round(data_2, 2))
                # print(np.round(data_2, 2)==-9999.99)
                plt.legend()
                plt.show()
            
            
            z_2  = (data_2[int_2][:-1]-data_2[int_2][1:])
            z_2 -= np.mean(z_2)
            if np.std(z_2)!=0: 
                z_2 /= np.std(z_2)
            else:
                # print('2.  std=%3.1f'%np.std(z_2), jj, db_2, id_2)
                plt.figure()
                # plt.scatter(time_1, data_1, label='data_1 (%s)'%id_1)
                plt.scatter(time_12, data_1[int_1]-np.mean(data_1[int_1]), label='data_1 SHARED TIME (%s)'%id_1)
                # plt.scatter(time_2, data_2, label='data_2 (%s)'%id_2)
                plt.scatter(time_12, data_2[int_2]-np.mean(data_2[int_2]), label='data_2 SHARED TIME (%s)'%id_2)
                # print(np.round(data_2, 2))
                # print(np.round(data_2, 2)==-9999.99)
                plt.legend()
                plt.show()
                # plt.figure()
                # plt.scatter(1/2.*(time_12[:-1]+time_12[1:]), z_2, label='z_2')
                # plt.legend()
                # raise Exception
            
            correlations[ii, jj] = np.corrcoef(np.vstack([data_1[int_1], data_2[int_2]]))[0,1]
            # rmse[ii, jj]         = np.sqrt(np.sum((np.abs(data_1[int_1])-np.abs(data_2[int_2]))**2)/len(time_12))
            rmse[ii, jj]         = np.sqrt(np.sum((data_1[int_1]-data_2[int_2])**2)/len(time_12))
            
            # rmse_diff[ii, jj]    = np.sqrt(np.sum((np.abs(z_1)-np.abs(z_2))**2)/len(time_12))
            rmse_diff[ii, jj]    = np.sqrt(np.sum((z_1-z_2)**2)/len(time_12))
            corr_diff[ii, jj]    = np.corrcoef(np.vstack([z_1, z_2]))[0,1]

            # print(ii, jj, 'corr %3.1f, rmse %3.1f, rmse-diff %3.1f, corr-diff %3.1f'%(correlations[ii, jj], rmse[ii, jj] ,  rmse_diff[ii, jj], corr_diff[ii, jj]))
            
            if ((np.isnan(correlations[ii,jj]) & np.isnan(rmse[ii,jj]))|(np.isnan(corr_diff[ii,jj]) & np.isnan(rmse_diff[ii,jj]))):
                print('!!! %d|%d nan detected in both correlation and rmse of record data (or its 1st difference). Danger of missing duplicate!!'%(ii, jj))
                for idd, dd in enumerate([correlations[ii,jj], rmse[ii,jj], rmse_diff[ii,jj],  corr_diff[ii,jj]]):
                    if np.isnan(dd):
                        print('nan detected in %s'%(['correlation','RMSE','RMSE of 1st difference', 'correlation of 1st difference'][idd]))
            
            # DEFINE CRITERIA:
            
            meta_crit         = (arch_1 == arch_2) & (type_1 == type_2) # archive types and proxy types must agree
            elevation_not_nan = ((~np.isnan(df['geo_meanElev'].iloc[ii]))& ~np.isnan(df['geo_meanElev'].iloc[jj]))
            elevation_dist    = np.abs(df['geo_meanElev'].iloc[ii]-df['geo_meanElev'].iloc[jj])
            #print(elevation_not_nan, elevation_dist)
            location_crit = ((distances_km[ii,jj] <= dist_tolerance_km) & 
                             (elevation_dist<=elev_tolerance if elevation_not_nan else True))
            #print('loc', location_crit)
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
                
    if save: write_csv(pot_dup_meta, ddir, fn)
    # write_csv(pot_dup_meta, ddir, fn%'meta_short')
    print('Saved indices, IDs, distances, correlations in %s'%ddir)
    print('='*30)
    print('Detected %d possible duplicates in %s.'%(n_pot_dups, df.name))
    print('Indices: %s'%(', ').join([str(pdi) for pdi in pot_dup_inds]))
    print('IDs: %s'%(', ').join([pdi[0]+' + '+pdi[1] for pdi in pot_dup_IDs]))
    print('='*30)
    return pot_dup_inds, pot_dup_IDs, distances_km, correlations
#==============================================================================
def dup_plot(df, ii, jj, id_1, id_2, time_1, time_2, time_12, data_1, data_2, int_1, int_2, 
             pot_dup_corr, keys_to_print=['originalDatabase', 'originalDataURL', 'datasetId', 'archive|proxy', 
                                          'geo_siteName', 'lat|lon|elev', 'mean|std|units', 'year' ], 
             dup_mdata_row=[]):
    """
    Plots the duplicate candidates. Plots the record data as a timeseries of anomalies in a common panel (w.r.t. shared time period) and prints out the most relevant metadata. Highlights identical metadata in orange and different metadata in green.
    """
    fs        = 10
    scale     = 0.15
    init_offs = -.5
    str_limit = 50
    
    fig  = plt.figure(figsize=(10, 8.5), dpi=75)
    grid = GS(3,1)
    ax   = plt.subplot(grid[0,0])
    #
    label1 = '%s (%d)'%(id_1, ii)#, lat_1, lon_1)   (%1.1f $^\circ$N, %1.1f $^\circ$E)
    #print('type', type(data_1), type(time_1), data_1.mask)
    ax.plot(time_1, data_1-np.mean(data_1[int_1]), color='k', marker='o',
            markersize=5,linestyle='None', label=label1)
    
    label2 = '%s (%d)'%(id_2, jj)#, lat_2, lon_2)     (%1.1f $^\circ$N, %1.1f $^\circ$E)
    ax.plot(time_2, data_2-np.mean(data_2[int_2]), color='tab:blue', marker='D',
            markersize=2,linestyle='None', label=label2)
    ax2 = ax.twinx()
    ax2.plot(time_12, (np.abs(data_1[int_1]-np.mean(data_1[int_1]))-
                       np.abs(data_2[int_2]-np.mean(data_2[int_2]))), 
             alpha=0.5, color='grey', zorder=99999)
    ax2.set_ylabel('difference')
    
    ax.legend(loc='upper right')
    ax.set_ylabel('anomalies w.r.t. %d-%d \n ('%(time_12[0], time_12[-1])+
                  df['paleoData_units'].iloc[ii]+', \n'+
                  df['paleoData_units'].iloc[jj]+')')
    ax.set_xlabel(df['yearUnits'].iloc[ii])
    ax.set_title('Possible duplicates. Correlation=%5.4f'%pot_dup_corr
                 +'. Time overlap=%d (%d'%(len(time_12), len(time_12)/len(time_1)*100)+
                 '%'+', %d'%(len(time_12)/len(time_2)*100)+'%)')

    txt = ('Metadata (differences are highlighted; only first '+
           str(str_limit)+' characters are displayed)')
    ax.text(-.1+.5, -0.3, txt, transform=ax.transAxes, ha='center', fontsize=fs)
    ax.text(-.1+0,  -0.4,'Key', transform=ax.transAxes, fontsize=fs+2)
    ax.text(-.1+.2, -0.4,'Record 1 (black)', transform=ax.transAxes, fontsize=fs+1)
    ax.text(-.1+.75,-0.4,'Record 2 (blue)',  transform=ax.transAxes, fontsize=fs+1)
    # plot the metadata 
    for ik, key in enumerate(keys_to_print):
        if key=='lat|lon|elev':
            metadata_field1 = '%3.1f|%3.1f|%3.1f'%(df['geo_meanLat'].iloc[ii],
                                                   df['geo_meanLon'].iloc[ii], 
                                                   df['geo_meanElev'].iloc[ii])
            metadata_field2 = '%3.1f|%3.1f|%3.1f'%(df['geo_meanLat'].iloc[jj],
                                                   df['geo_meanLon'].iloc[jj], 
                                                   df['geo_meanElev'].iloc[jj])
        elif key=='year':
            metadata_field1 = '%3.1f - %3.1f %s'%(np.min(df['year'].iloc[ii]), 
                                                  np.max(df['year'].iloc[ii]), 
                                                  df['yearUnits'].iloc[ii])
            metadata_field2 = '%3.1f - %3.1f %s'%(np.min(df['year'].iloc[jj]), 
                                                  np.max(df['year'].iloc[jj]), 
                                                  df['yearUnits'].iloc[jj])
        elif key=='mean|std|units':
            metadata_field1 = '%4.2f | %4.2f | %s'%(np.mean(df['paleoData_values'].iloc[ii]), 
                                                  np.std(df['paleoData_values'].iloc[ii]), 
                                                  df['paleoData_units'].iloc[ii])
            metadata_field2 = '%4.2f | %4.2f | %s'%(np.mean(df['paleoData_values'].iloc[jj]), 
                                                  np.std(df['paleoData_values'].iloc[jj]), 
                                                  df['paleoData_units'].iloc[jj])
        elif key=='archive|proxy': #'archiveType','paleoData_proxy',
            metadata_field1 = '%s|%s'%(df['archiveType'].iloc[ii],
                                       df['paleoData_proxy'].iloc[ii])
            metadata_field2 = '%s|%s'%(df['archiveType'].iloc[jj],
                                       df['paleoData_proxy'].iloc[jj])
        else:
            metadata_field1, metadata_field2 = df[key].iloc[[ii, jj]]
            
        if 'URL' in key:
            metadata_field1 = metadata_field1.replace('https://','')
            metadata_field2 = metadata_field2.replace('https://','')
            
        row = init_offs-scale*ik
        col = [-.1, -.1+.2, -.1+.75]
        
        if metadata_field1 != metadata_field2:
            plt.text(col[0], row, ' '*250,transform=ax.transAxes, fontsize=fs, 
                     bbox={#'facecolor': 'white', 
                           'alpha':.2, 'pad':2.5,# 'edgecolor': 'white'
                          })
        else:
            plt.text(col[0], row, ' '*250,transform=ax.transAxes, fontsize=fs,
                     bbox={'facecolor':'tab:orange','alpha':.2, 'pad':2.5})
        
        plt.text(col[0], row, key, transform=ax.transAxes,fontsize=fs)
        plt.text(col[1], row, str(metadata_field1)[0:str_limit],
                 transform=ax.transAxes,fontsize=fs)
        plt.text(col[2], row, str(metadata_field2)[0:str_limit],
                 transform=ax.transAxes,fontsize=fs)

        dup_mdata_row += [[metadata_field1, metadata_field2][m_ind].split('|')[s_ind]   
                          for s_ind in range(len(metadata_field1.split('|'))) 
                          for m_ind in [0,1]]
        
    return fig, dup_mdata_row
#==============================================================================
def plot_duplicates(df, save_figures=True, write_output=True):
    """
    # Generates plots of the duplicates (uses dup_plot function) and saves metadata of duplicates as csv.
    Parameters:
        df : pd.DataFrame object, holds the data and metadata.
             Define relevant keys below. Default is Pages2k keys.
             
        # specify the correct column names of the dataframe 
        'geo_meanLat', 'geo_meanLon'   = latitude and longitude  column name ('geo_meanLat', 'geo_meanLon')
        'year', 'paleoData_values'  = time key and proxy data values key ('year', 'paleoData_values')
        'archiveType', 'paleoData_proxy' = archive type key and proxy type key ('archiveType', 'paleoData_proxy')
        'datasetId'             = ID/dataset name key ('datasetId')
        'geo_siteName'           = geographical site name key ('geo_siteName')
        'originalDatabase'          = original dataset name key ('original_Database')
    Returns:
    
    """
    keys_to_print = ['originalDatabase', 'originalDataURL', 'datasetId', 'archive|proxy',
                     'geo_siteName', 'lat|lon|elev', 'mean|std|units', 'year' ]

    ddir = '%s/dup_detection/'%(df.name)+'dup_detection_candidates_'+df.name
    # ddir = '%s/dup_detection/'%(df.name)+'pot_dup_%s_'+df.name
    # load the potential duplicate data as found in find_duplicates function
    pot_dup_meta, head = read_csv(ddir, header=True)
    # pot_dup_meta, head = read_csv(ddir%'meta_short', header=True)

    pot_dup_inds   = np.array(pot_dup_meta[:, :2], dtype=int)
    pot_dup_corrs  = np.array(pot_dup_meta[:, 4], dtype=float)
    pot_dup_dists  = np.array(pot_dup_meta[:, 5], dtype=float)

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
        data_1           = df['paleoData_values'].iloc[ii]
        data_2           = df['paleoData_values'].iloc[jj]
        
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
            figsave(fig, '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+
                    '_%d_%d'%(ii,jj), add='%s/dup_detection'%df.name, form='pdf')
        # if write_output:
        #     write_csv(np.array(dup_mdata), 'dup_detection/%s/'%df.name,
        #               'pot_dup_candidates_metadata_%s'%(df.name))
            
        # Print the interpretation values 
        print('=== POTENTIAL DUPLICATE %d/%d'%(i_pot_dups, n_pot_dups)+': %s+%s ==='%(df['datasetId'].iloc[ii], df['datasetId'].iloc[jj]))
    return 

#==============================================================================


def duplicate_decisions(df, operator_details=False, choose_recollection=True, keep_all=False, 
                        plot=True, remove_identicals=True, dist_tolerance_km=8):
    """
    Go through the potential duplicates one by one and (if option selected) plot, and decide whether they are true duplicates or not. Decisions are saved as csv sheet. Duplicate free dataframe is saved too.
    Parameters:
        df : pd.DataFrame object, holds the data and metadata.
        
             
        # specify the correct column names of the dataframe 
        'geo_meanLat', 'geo_meanLon'   = latitude and longitude  column name ('geo_meanLat', 'geo_meanLon')
        'year', 'paleoData_values'  = time key and proxy data values key ('year', 'paleoData_values')
        'archiveType', 'paleoData_proxy' = archive type key and proxy type key ('archiveType', 'paleoData_proxy')
        'datasetId'             = ID/dataset name key ('datasetId')
        'geo_siteName'          = geographical site name key ('geo_siteName')
        'originalDatabase'      = original dataset name key ('original_Database')

        
        choose_recollection = True or False. Set True if you want to automatically 
                              select the site which contains recollection in site name
        keep_all            = True or False. Set True if we only want to plot the figures but not make a decision. Default is False!
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

    
    dirname = '%s/dup_detection/'%(df.name)
    filename = 'dup_decisions_%s'%df.name # name of csv file which saves duplicate candidate pairs
    # filename+='_%s'%(initials)
    date = date_time[2:10]
    filename+='_%s_%s'%(initials, date)
    # ddir = dirname+'pot_dup_%s_'+df.name
    ddir = dirname+'dup_detection_candidates_'+df.name
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # try to load data from backup!
    try:
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
    pot_dup_meta, head = read_csv(ddir, header=True)
    # pot_dup_meta, head = read_csv(ddir%'meta_short', header=True)

    pot_dup_inds   = np.array(pot_dup_meta[:, :2], dtype=int)
    pot_dup_corrs  = np.array(pot_dup_meta[:, 4], dtype=float)
    pot_dup_dists  = np.array(pot_dup_meta[:, 5], dtype=float)

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
    
    with open(dirname+filename+'_BACKUP.csv', 'w', newline='') as f:
        
        writer = csv.writer(f)
        writer.writerows(header)
        print('header for backup', header)
        writer.writerows(cols)
        print('cols for backup', cols)
        for data_row in list(data):
            writer.writerow(list(data_row))
            print('list(data_row)', list(data_row))
            dup_dec+=[list(data_row)]
        print('data for backup', data)
    
        for i_pot_dups, (ii, jj) in enumerate(pot_dup_inds):
            if load_saved_data:
                if i_pot_dups<last_index: continue
                else:
                    print(ii, jj, last_index, i_pot_dups)
            
            # data and metadata associated with the two potential duplicates
            id_1, id_2       = df['datasetId'].iloc[[ii, jj]]
            time_1, time_2   = df['year'].iloc[[ii, jj]]
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) 
            data_1           = df['paleoData_values'].iloc[ii]
            data_2           = df['paleoData_values'].iloc[jj]
            
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
            # (np.round(lat_1, 1)==np.round(lat_2, 1)) & 
            #                       (np.round(lon_1, 1)==np.round(lon_2, 1)) 
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
            correlation_perfect = (True if pot_dup_corrs[i_pot_dups]>=0.99 else False) & (len(time_1)==len(time_2))
            
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
                elif (remove_identicals & metadata_identical & (data_identical|correlation_perfect)):
                    # if most metadata and data matches except URL and/or site name, choose record according to hierarchy of databases
                    if data_identical:
                        dec_comment = 'RECORDS IDENTICAL (identical data) except for URLs and/or site Name. Automatically choose #%s.'%auto_choice
                    else:
                        dec_comment = 'RECORDS IDENTICAL (perfect correlation) except for URLs and/or site Name. Automatically choose #%s.'%auto_choice
                        
                    print(dec_comment)
                    keep = auto_choice#'1'
                    dec_type='AUTO: IDENTICAL except for URLs and/or geo_siteName.'
                    
                else:
                    if keep_all:
                        keep='b'
                        dec_type='AUTO: KEEP ALL'
                    else:
                        dec_type = 'MANUAL'
                        fig, dup_mdata_row = dup_plot(df, ii, jj, id_1, id_2, time_1, time_2, time_12, data_1, data_2, int_1, int_2, pot_dup_corrs[i_pot_dups])
                        plt.show(block=False)
                        print('Would you like to leave a comment?')
                        dec_comment = input(' Please type your comment here and/or press enter.')
                        keep = input('Keep record 1 (%s, black) [1], record 2 (%s, blue) [2], keep both [b], keep none [n] or create a composite of both records [c]?  [Type 1/2/b/n/c]:'%(id_1, id_2))
                        
                # now write down the decision
                if keep=='1':
                    print('KEEP BLUE: keep %s, remove %s.'%(id_1, id_2))
                    decisions[jj]='REMOVE'
                elif keep=='2':
                    print('KEEP BLACK: remove %s, keep %s.'%(id_1, id_2))
                    decisions[ii]='REMOVE'
                elif keep=='n':
                    print('REMOVE BOTH: remove %s, remove %s.'%(id_2, id_1))
                    decisions[ii]='REMOVE'
                    decisions[jj]='REMOVE'
                elif keep=='b':
                    print('KEEP BOTH: keep %s, keep %s.'%(id_1, id_2))
                elif keep=='c':
                    print('CREATE A COMPOSITE OF BOTH RECORDS: %s, %s.'%(id_1, id_2))
                    decisions[ii]='COMPOSITE'
                    decisions[jj]='COMPOSITE'
                    
                    
            figpath    = 'https://nzero.umd.edu:444/hub/user-redirect/lab/tree/compile_proxy_database_v2.1/%s/dup_detection/'%df.name
            figpath  += '%03d_%s_%s'%(i_pot_dups, id_1, id_2)+'_'+'_%d_%d.jpg'%(ii,jj)
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
        
    print('=====================================================================')
    print('END OF DUPLICATE DECISION PROCESS.')
    print('=====================================================================')
    print('Summary of all decisions made:')
    for ii_d in range(len(dup_dec)):
        keep_1=dup_dec[ii_d][-4]
        keep_2=dup_dec[ii_d][-3]
        print('#%d: %s record %s. %s record %s.'%(ii_d, keep_1, dup_dec[ii_d][3], 
                                                        keep_2, dup_dec[ii_d][4]))
        
    comment = '# '+input('Type your comment on your decision process here and/or press enter:')
    header  += [[comment]]
    header  += cols
    
    print(np.array(dup_dec).shape)
    write_csv(np.array(dup_dec), dirname, filename, header=header)
    print('Saved the decisions under %s.csv'%(dirname+filename))
    
    return 


def join_composites_metadata(df, comp_ID_pairs, df_decisions, header):
    
    # create composite dataframe
    df_comp = pd.DataFrame()
    
    for ii in range(len(comp_ID_pairs)):
        row = {}
        ID_1, ID_2   = comp_ID_pairs.iloc[ii][['datasetId 1', 'datasetId 2']]
        db_1, db_2   = comp_ID_pairs.iloc[ii][['originalDatabase 1', 'originalDatabase 2']]
        dec_1, dec_2 = comp_ID_pairs.iloc[ii][['Decision 1', 'Decision 2']]
        print(ID_1, ID_2)
    
        # metadata should match exactly for the following columns (check), if not choose metadata values:
        for key in ['archiveType', 'geo_meanElev', 'geo_meanLat', 'geo_meanLon',
                    'geo_siteName', 'paleoData_proxy', 'yearUnits', 'climateInterpretation_variable']:
            # print(df.at[ID_1, key], df.at[ID_2, key])
            if df.at[ID_1, key]==df.at[ID_2, key]:
                row[key]=df.at[ID_1, key]
            else:
                print('--------------------------------------------------------------------------------')
                add_dup_note = 'Metadata differs for %s in original records: %s (%s) and %s (%s). '%(key, ID_1, str(df.at[ID_1, key]), 
                                                                                 ID_2, str(df.at[ID_2, key]))
                print('Metadata different for >>>%s<<< in: %s (%s) and %s (%s). '%(key, ID_1, str(df.at[ID_1, key]), 
                                                                                 ID_2, str(df.at[ID_2, key])))
                try:
                    entry='COMPOSITE: '+df.at[ID_1, key]+' + '+df.at[ID_2, key]
                    add_dup_note += 'Metadata composited to: '+entry
                    if key=='climateInterpretation_variable':
                        if df.at[ID_1, key]=='N/A': entry=df.at[ID_2, key]
                        if df.at[ID_2, key]=='N/A': entry=df.at[ID_1, key]
                        if (df.at[ID_1, key] in df.at[ID_2, key]):  entry=df.at[ID_2, key]
                        if (df.at[ID_2, key] in df.at[ID_1, key]):  entry=df.at[ID_1, key]
                    loop=False
                except TypeError:
                    print('Can not create composites for numerical metadata! Create average instead.')
                    # av = input('Type [y] if you want to average the metadata. Otherwise type [n].')
                    # if av.lower() in ['y', 'yes']:
                    entry = np.mean([df.at[ID_1, key], df.at[ID_2, key]])                        
                    add_dup_note += 'Metadata averaged to: '+str(entry)
                    loop=False
                row[key]= entry
                print('Added the following note to DuplicateDetails:', add_dup_note)
                # loop=True
                # while loop:
                #     print('--------------------------------------------------------------------------------')
                #     add_dup_note = 'Metadata differs for %s in original records: %s (%s) and %s (%s). '%(key, ID_1, str(df.at[ID_1, key]), 
                #                                                                      ID_2, str(df.at[ID_2, key]))
                #     print('Metadata different for >>>%s<<< in: %s (%s) and %s (%s). '%(key, ID_1, str(df.at[ID_1, key]), 
                #                                                                      ID_2, str(df.at[ID_2, key])))
                #     entry = input('Keep entry for %s [1] or %s [2] or join [j] or type in new value? (Type 1/2/j/...).')
                #     if entry=='1':
                #         entry=df.at[ID_1, key]
                #         loop=False
                #         add_dup_note += 'Chose metadata: %s'%str(entry)
                #     elif entry=='2':
                #         entry=df.at[ID_2, key]
                #         loop=False
                #         add_dup_note += 'Chose metadata: %s'%str(entry)
                #     elif (entry.upper()=='JOIN')|(entry.lower()=='j') :
                #         try:
                #             entry='COMPOSITE: '+df.at[ID_1, key]+' + '+df.at[ID_2, key]
                #             add_dup_note += 'Metadata composited to: '+entry
                #             loop=False
                #         except TypeError:
                #             print('Can not create composites for numerical metadata! Create average instead.')
                #             # av = input('Type [y] if you want to average the metadata. Otherwise type [n].')
                #             # if av.lower() in ['y', 'yes']:
                #             entry = np.mean([df.at[ID_1, key], df.at[ID_2, key]])                        
                #             add_dup_note += 'Metadata averaged to: '+str(entry)
                #             loop=False
                #             # else:
                #             #     print('Please choose a different action.')
                #             #     continue
                #     else:
                #         print('Chose to manually define new value for metadata.')
                #         print('New metadata: %s'%entry)
                #         confirm = input('Please confirm by typing [y/n] and press enter.')
                #         if confirm=='y': loop=False
                #         add_dup_note += 'Metadata manually set to: %s'%str(entry)
                #     row[key]= entry
                #     print('Added the following note to DuplicateDetails:', add_dup_note)
                    
        # create new entries for the following columns:
        # create z-scores of data
        data_1, data_2 = df.at[ID_1, 'paleoData_values'], df.at[ID_2, 'paleoData_values']
    
        # year
        time_1, time_2 = df.at[ID_1, 'year'], df.at[ID_2, 'year']
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
    
        fig = plt.figure(figsize=(6, 3), dpi=300)
        
        plt.scatter(time_1, data_1, s=20, color='tab:blue', label=ID_1)
        plt.scatter(time_2, data_2, s=20, color='tab:orange', label=ID_2)
        plt.scatter(time, data, s=10, color='k', label='composite')
        plt.legend()
        plt.show()
        figsave(fig, 'composite_%s_%s'%(ID_1, ID_2), add='/%s/dup_detection/'%df.name)
        
        
        # new metadata for identification etc.
        # add climateInterpretation_variable
        row['dataSetName']      = df.at[ID_1, 'dataSetName']+', '+df.at[ID_2, 'dataSetName']
        row['originalDatabase'] = '%s_composite_standardised'%df.name
        row['originalDataURL']  = '%s: %s, %s: %s'%(ID_1, df.at[ID_1, 'originalDataURL'], ID_2, df.at[ID_2, 'originalDataURL'])
        row['paleoData_notes']  = '%s: %s, %s: %s'%(ID_1, df.at[ID_1, 'paleoData_notes'], ID_2, df.at[ID_2, 'paleoData_notes'])
        row['climateInterpretation_variableDetail']  = '%s: %s, %s: %s'%(ID_1, df.at[ID_1, 'climateInterpretation_variable'], ID_2, df.at[ID_2, 'climateInterpretation_variable'])
        row['datasetId']        = '%s_composite_st_'%df.name+ID_1+'_'+ID_2    
        row['paleoData_units']  = 'z-scores'
    
        # save details on composition process in DuplicateDetails:
        if row['DuplicateDetails']=='N/A':
            row['DuplicateDetails']=''
        row['DuplicateDetails']  += 'COMPOSITE OF two overlapping duplicate records: %s (originally from %s) and %s (originally from %s). '%(ID_1, db_1, ID_2, db_2)
        row['DuplicateDetails'] += '' # add details on how the records were merged.
        row['DuplicateDetails'] += ' Decision type: %s .'%df_decisions.iloc[ii]['Decision type']
        
        if comp_ID_pairs.iloc[ii]['Decision type']=='MANUAL': 
            operator_details = ' '.join(header[1:]).replace(' Modified ','')[:-2].replace(':','').replace('  E-Mail', '')
            row['DuplicateDetails']+=' (decision made %s).'%(operator_details)
            row['DuplicateDetails']+=' Note on decision process: %s'%comp_ID_pairs.iloc[ii]['Decision comment']
        row['DuplicateDetails'] += add_dup_note
        
        # create dataframe for composites
        df_comp = pd.concat([df_comp, pd.DataFrame({kk: [vv] for kk, vv in row.items()})], ignore_index = True, axis=0)
        
        # print(df_comp)
    return df_comp
