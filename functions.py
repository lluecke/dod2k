# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:42:11 2023

@author: lluecke

keeps all the functions we need for loading and plotting data 
for PAGES2k and B14
"""
import os
import numpy as np
from copy import deepcopy as dc
import matplotlib.pyplot as plt
import csv
import pandas as pd

def write_csv(data, filename):
    
    with open(filename+'.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)
    return
    
def write_dataframe_columns_to_csv(data, header, filename, path, ID=False):
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
    
    # save to a list of csv files (metadata, data, year)
    for fn in ['metadata', 'paleoData_values', 'year']:
        if saveto=='df.name':
            path = '/%s'%df.name
            filename = df.name+'_compact_%s'
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
    print('Saved to %s'%os.getcwd()+path+'/'+filename+'.csv')
    return

def read_compact_dataframe_columns_from_csv(key, filename, path):
    
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

def load_compact_dataframe_from_csv(df_name, readfrom='df.name', index_col=4):
    # load the compact dataframe from three csv files 
    if readfrom=='df.name':
        path = '/%s'%df_name
        filename = df_name+'_compact_%s'
    else:
        path=readfrom[0]
        filename=readfrom[1]
    
    df_meta = pd.read_csv(os.getcwd()+path+'/'+filename%'metadata'+'.csv', index_col=index_col, keep_default_na=False)
    df_data = read_compact_dataframe_columns_from_csv('paleoData_values', filename%'paleoData_values', path)
    df_year = read_compact_dataframe_columns_from_csv('year', filename%'year', path)
    # print('aaaaa')
    
    df_csv = df_meta.join(df_data).join(df_year)
    
    df_csv['datasetId'] = df_csv.index
    df_csv.index=range(len(df_csv))
    

    df_csv = df_csv[sorted(df_csv.columns)]

    
    df_csv['year'] = df_csv['year'].map(lambda x: np.array(x, dtype = np.float32))
    df_csv['paleoData_values'] = df_csv['paleoData_values'].map(lambda x: np.array(x, dtype = np.float32))

    df_csv = df_csv.astype({'archiveType': str, 
                            'dataSetName': str, 
                            'climateInterpretation_variable': str,
                            'datasetId': str, 
                            'geo_meanElev': np.float32, 
                            'geo_meanLat': np.float32, 
                            'geo_meanLon': np.float32, 
                            'geo_siteName': str, 
                            'originalDatabase': str, 
                            'originalDataURL': str, 
                            'paleoData_notes': str, 
                            'paleoData_proxy': str,
                            'paleoData_sensorSpecies': str,
                            'paleoData_units': str, 
                            'yearUnits': str})


    return df_csv


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
    Saves figure *fig* as *name*, can also add directory via *add*

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
                    format='jpg', dpi=100, bbox_inches='tight', pad_inches=0.0)
    print('saved figure in '+ addfigs+ddir+'/'+name+'.'+form)
    if close: plt.close()
    return

def cleanup(string):
    string = string.replace('#', '')
    string = string.replace('\t', '')
    string = string.replace('\n', '')
    while string.startswith(' '): string = string[1:]
    while string.endswith(' '): string = string[:-1]
    return string
def shade_percentiles(x, y, color, ax, alpha=1, lu=False, zorder=None, lw=1,
                      ups=[60, 70, 80, 90, 95], label=None):
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
    N = len(data)
    cmap         = cm.get_cmap(colormap)
    sm           = cm.ScalarMappable(cmap = colormap)
    sm.set_array(range(N))
    norm         = Normalize(vmin=minval, vmax=maxval)
    rgba         = cmap(norm(data))
    cols         = list(rgba)
    return cols

def fns(path, end='.nc', start='', other_cond='', print_dir=True):
    fn = []
    for root, dirs, files in os.walk(path):
        for fil in files:
            if fil.endswith(end) and fil.startswith(start) and (other_cond in fil):
                fn.append(fil)
    if print_dir: print(fn)
    fn = np.sort(fn)
    return fn 

def conv_nan(value):
    if value=='nan' or np.isnan(value):
        return -9999.99
    elif np.round(value, 2)==-9999.99:
        return value
    else:
        return value
def convert_to_nparray(data):
    data = np.array([conv_nan(vv) for vv in data])#, -9999.99)
    mask = np.round(data, 2)==-9999.99#np.array([False if np.round(vv, 2)!=-9999.99 else True for vv in data ])
    return np.ma.masked_array(data, mask)
