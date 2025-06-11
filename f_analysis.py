import numpy as np
import matplotlib.pyplot as plt
import functions as f
from functools import reduce
from matplotlib.gridspec import GridSpec as GS
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib




def plot_resolution(df, key, mincount=0):
    
    # print('resolutions: ')
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
        # other=0
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
        # count_res['other']=other
                
    # if mincount!=0:
    #     other=0
    #     rmv = []
    #     for kk, vv in count_res.items():
    #         if vv<mincount:
    #             other+=1
    #             rmv+=[kk]
    #     for kk in rmv: del count_res[kk]
    #     count_res['other']=other
                
    
    plt.figure(dpi=100, figsize=(5,3))
    plt.title(key)
    ax=plt.gca()
    ii=0
    rr=[]
    for kk in np.sort(list(count_res)):
        # if count<mincount: continue
        plt.bar(ii, count_res[kk])
        ii+=1
        rr+=[kk]
    ax.set_xticks(range(ii))
    ax.set_xticklabels(rr, rotation=45, ha='right', fontsize=7)
    plt.xlabel('resolution')
    plt.ylabel('count')
    plt.show()
    return 
    

    




def plot_length(df, key, mincount=0):
    
    # print('resolutions: ')
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
    plt.title(key)
    ax=plt.gca()
    ii=0
    rr=[]
    for res, count in count_res.items():
        if count<mincount: continue
        plt.bar(ii, count)
        ii+=1
        rr+=[res]
    ax.set_xticks(range(ii))
    ax.set_xticklabels(rr, rotation=45, ha='right', fontsize=7)
    plt.xlabel('length')
    plt.ylabel('count')
    plt.show()
    return 
    


def plot_coverage(df, years, key):
    
    coverage_filt = np.zeros(years.shape[0])

    miny, maxy = years[0], years[-1]
    for ii in range(len(df['year'])):
        # time_12, int_1, int_2 = np.intersect1d(years, df.iloc[ii].year, return_indices=True)
        coverage_filt[(years>=df.iloc[ii].miny)&(years<=df.iloc[ii].maxy)] += 1
        # coverage_filt[int_1]+=1
    
    fig = plt.figure(figsize=(6, 3), dpi=100)
    plt.title(key)
    ax = plt.gca()
    plt.step(years, coverage_filt, color='k', label='all records', lw=3)
    plt.xlabel('year')
    plt.ylabel('total # of records')
    
    h1, l1 = ax.get_legend_handles_labels()
    plt.legend(h1, l1, ncol=3, framealpha=0)
    plt.ylabel('# of records per archive')
    plt.show()
    return fig



def filter_resolution(df, minres, key):
    rmask = df.resolution.apply(lambda x: np.all(np.array(x)<=minres))
    print('Keep %d records with resolution <=%d. Exclude %d records.'%(len(df[rmask]), minres, len(df[~rmask])))
    
    return df[rmask]
    
def filter_record_length(df, nyears, mny, mxy, key):


    remove = []
    for ii in df.index:
        if np.sum((df.at[ii, 'year']>=mny)&(df.at[ii, 'year']<=mxy))<nyears:
            # print('No available data', ii)
            remove+=[ii]
    
    df=df.drop(labels=remove)
    # mask   = ~(df.length>=nyears)
    print('Keep %d records with nyears>=%d during %d-%d. Exclude %d records.'%(df.shape[0], nyears, mny, mxy, len(remove)))

    # n_recs_masked = len(df[~mask])
    
    
    # generate array of coverage (how many records are available each year, in total)
    # fig = plot_coverage(df[~mask], years_hom, key)
    
    return df



def filter_data_availability(df, mny, mxy):
       
    remove = []
    for ii in df.index:
        if np.sum((df.at[ii, 'year']>=mny)&(df.at[ii, 'year']<=mxy))==0:
            # print('No available data', ii)
            remove+=[ii]
    df=df.drop(labels=remove)
    print('No available data: ', remove)
    print('Keep %d records with data available between %d-%d. Exclude %d records.'%(df.shape[0], mny, mxy, len(remove)))

    return df


def homogenise_time(df, mny, mxy, minres, key):

    
    years_hom     = np.arange(mny, mxy+minres, minres)                                       #
    
    print('Homogenised time coordinate: %d-%d CE'%(years_hom[0], years_hom[-1]))
    print('Resolution: %s years'%str(np.unique(np.diff(years_hom))))

    find_shared_period(df, minmax=(mny, mxy))
    
    return df, years_hom


def homogenise_data_dimensions_old(df, years_hom, key, print_output=True, plot_output=True):
    """
    # assign the paleoData_values to the non-missing values in the homogenised data array
    # the homogenised data array is masked such as np.isin(df.year_hom.iloc[0], df.year.iloc[0])

    INPUT:
        df: dataframe
        years_hom: homogenised time coordinate

    OUTPUT:
        paleoData_values_hom:  
                masked array (nrecs, nyears) data matrix of paleoData_values, missing data is set to zero and masked out (use np.ma functions for correct data processing)
        yh:
                list (len(nrecs)) of np.ma.masked_arrays (homogenised shape of nyears) -> same content as above array, used to define new column in df
        paleoData_zscores_hom:  
                masked array (nrecs, nyears) data matrix of paleoData_zscores, missing data is set to zero and masked out (use np.ma functions for correct data processing)
        zh:
                list (len(nrecs)) of np.ma.masked_arrays (homogenised shape of nyears) -> same content as above array, used to define new column in df
    """

    mny = years_hom[0]
    mxy = years_hom[-1]
    minres = np.unique(np.diff(years_hom))[0]

    n_recs = len(df)
    
    # assign data values
    paleoData_values_hom  = np.ma.masked_array(np.zeros([n_recs, len(years_hom)]), np.zeros([n_recs, len(years_hom)]))
    paleoData_zscores_hom = np.ma.masked_array(np.zeros([n_recs, len(years_hom)]), np.zeros([n_recs, len(years_hom)]))

    yh = []
    zh = []
    
    for ijk, ii in enumerate(df.index):
        # # mask out the missing time period
        yh_base = np.ma.masked_array(np.zeros(len(years_hom)), mask=True, fill_value=np.nan)
        zh_base = np.ma.masked_array(np.zeros(len(years_hom)), mask=True, fill_value=np.nan)
        yy = df.at[ii, 'year']
        hmask = np.isin(years_hom, yy[(yy>=mny)&(yy<=mxy)]) # mask for years_hom
        
        # average data according to minres
        data_HR = df.at[ii, 'paleoData_values'][(yy>=mny)&(yy<=mxy)]
        data_LR = np.zeros(len(years_hom[hmask]))
        zsco_HR = df.at[ii, 'paleoData_zscores'][(yy>=mny)&(yy<=mxy)]
        zsco_LR = np.zeros(len(years_hom[hmask]))
        # zsco_LR = np.zeros(len(years_hom))
        for jj, yi in enumerate(years_hom[hmask]):
        # for jj, yi in enumerate(years_hom):
            ymask = (yy[(yy>=mny)&(yy<=mxy)]<=yi)&(yy[(yy>=mny)&(yy<=mxy)]>(yi-minres))
            
            data_LR[jj] = np.average(data_HR[ymask])
            zsco_LR[jj] = np.average(zsco_HR[ymask])
            
        # fill non-missing time period with data


        
        # yh_base=data_LR 
        # print(yh_base.shape, years_hom.shape)
        # print(yh_base, years_hom)
        
        yh_base[hmask]=data_LR 
        
        # # unmask values with non-missing data
        # yh_base.mask=False
        yh_base[hmask].mask=False
        # raise Exception
    
        # # repeat for z-scores
        # zh_base=zsco_LR 
        zh_base[hmask]=zsco_LR 
        # # unmask values with non-missing data
        zh_base[hmask].mask=False
        # zh_base.mask=False
        
        # # check array is correct
        if print_output:
            # print(ii, ijk, 'years_hom size: ', years_hom[hmask].shape, 'new array size: ', 
            print(ii, ijk, 'years_hom size: ', years_hom.shape, 'new array size: ', 
                  yh_base[~yh_base.mask].shape, 'resolution: ', np.unique(np.diff(yy)), 
                  # 'time coord: from %s-%s'%(yy[(yy>=mny)&(yy<=mxy)][0], yy[(yy>=mny)&(yy<=mxy)][-1])
                 )
            print(paleoData_values_hom[ijk,:].shape, yh_base.shape)  
            
        paleoData_values_hom[ijk,:]  = yh_base
        paleoData_zscores_hom[ijk,:] = zh_base
        yh.append(yh_base)
        zh.append(zh_base)

    print(paleoData_values_hom.shape)
    
    if plot_output:
        n_recs=min(len(df), 50)
        # plot paleoData_values_hom and paleoData_zscores_hom as they appear in df
        fig = plt.figure(figsize=(8,5))
        plt.suptitle(key)
        plt.subplot(221)
        plt.title('paleoData_values HOM')
        for ii in range(n_recs):
            shift = ii
            plt.plot(years_hom, paleoData_values_hom[ii,:]+shift#df.paleoData_values_hom.iloc[ii]
                     , lw=1)
        plt.xlim(mny, mxy)
            
        plt.subplot(222)
        plt.title('paleoData_values')
        for ii in range(n_recs):
            shift = ii
            plt.plot(df.year.iloc[ii],
                     df.paleoData_values.iloc[ii]+shift, lw=1)
        plt.xlim(mny, mxy)
        
        plt.subplot(223)
        plt.title('paleoData_zscores HOM')
        for ii in range(n_recs):
            shift = ii
            plt.plot(years_hom,
                     paleoData_zscores_hom[ii,:]+shift#.iloc[ii]
                     , lw=1)
        plt.xlim(mny, mxy)
        
        plt.subplot(224)
        plt.title('paleoData_zscores')
        for ii in range(n_recs):
            shift = ii
            plt.plot(df.year.iloc[ii],
                     df.paleoData_zscores.iloc[ii]+shift, lw=1)
        plt.xlim(mny, mxy)
        fig.tight_layout()
    return paleoData_values_hom, paleoData_zscores_hom, yh, zh


def homogenise_data_dimensions(df, years_hom, key, print_output=False, plot_output=True):
    """
    # assign the paleoData_values to the non-missing values in the homogenised data array
    # the homogenised data array is masked such as np.isin(df.year_hom.iloc[0], df.year.iloc[0])

    INPUT:
        df: dataframe
        years_hom: homogenised time coordinate

    OUTPUT:
        paleoData_values_hom:  
                masked array (nrecs, nyears) data matrix of paleoData_values, missing data is set to zero and masked out (use np.ma functions for correct data processing)
        yh:
                list (len(nrecs)) of np.ma.masked_arrays (homogenised shape of nyears) -> same content as above array, used to define new column in df
        paleoData_zscores_hom:  
                masked array (nrecs, nyears) data matrix of paleoData_zscores, missing data is set to zero and masked out (use np.ma functions for correct data processing)
        zh:
                list (len(nrecs)) of np.ma.masked_arrays (homogenised shape of nyears) -> same content as above array, used to define new column in df
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
        plt.suptitle(key)
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
    return paleoData_values_hom, paleoData_zscores_hom, year_hom_avbl, zsco_hom_avbl   #, yh, zh  


def homogenise_data_dimensions_old2(df, years_hom, key, print_output=False, plot_output=True):
    """
    # assign the paleoData_values to the non-missing values in the homogenised data array
    # the homogenised data array is masked such as np.isin(df.year_hom.iloc[0], df.year.iloc[0])

    INPUT:
        df: dataframe
        years_hom: homogenised time coordinate

    OUTPUT:
        paleoData_values_hom:  
                masked array (nrecs, nyears) data matrix of paleoData_values, missing data is set to zero and masked out (use np.ma functions for correct data processing)
        yh:
                list (len(nrecs)) of np.ma.masked_arrays (homogenised shape of nyears) -> same content as above array, used to define new column in df
        paleoData_zscores_hom:  
                masked array (nrecs, nyears) data matrix of paleoData_zscores, missing data is set to zero and masked out (use np.ma functions for correct data processing)
        zh:
                list (len(nrecs)) of np.ma.masked_arrays (homogenised shape of nyears) -> same content as above array, used to define new column in df
    """

    mny = years_hom[0]
    mxy = years_hom[-1]
    minres = np.unique(np.diff(years_hom))[0]

    n_recs = len(df) 
    
    # assign data values
    paleoData_values_hom  = np.ma.masked_array(np.zeros([n_recs, len(years_hom)]), np.zeros([n_recs, len(years_hom)]))
    paleoData_zscores_hom = np.ma.masked_array(np.zeros([n_recs, len(years_hom)]), np.zeros([n_recs, len(years_hom)]))

    year_hom_avbl = []
    
    for ijk, ii in enumerate(df.index):
        # create empty data arrays 

        time = df.at[ii, 'year']
        
        
        data_LR = np.zeros(len(years_hom))
        data_HR = df.at[ii, 'paleoData_values']
        
        zsco_HR = df.at[ii, 'paleoData_zscores']
        zsco_LR = np.zeros(len(years_hom))

        tt = []
        for jj, xi in enumerate(years_hom):
            window = (time>xi-minres)&(time<=xi)
            # print(xi, time[window])
            if len(time[window])==0:
                data_LR[jj] = 0#np.nan
                zsco_LR[jj] = 0#np.nan
            else:
                data_LR[jj] = np.average(data_HR[window])
                zsco_LR[jj] = np.average(zsco_HR[window])
                tt+=[time[window][-1]]
        

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
        # yh.append(yh_base)
        # zh.append(zh_base)
        year_hom_avbl.append(tt)

    print(paleoData_values_hom.shape)
    
    if plot_output:
        n_recs=min(len(df), 50)
        # plot paleoData_values_hom and paleoData_zscores_hom as they appear in df
        fig = plt.figure(figsize=(8,5))
        plt.suptitle(key)
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
    return paleoData_values_hom, paleoData_zscores_hom, year_hom_avbl#, yh, zh  


def covert_subannual_to_annual(df):
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
    # returns minimum year, maximum year of shared period
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
    # calculate z-score 
    z = x.paleoData_values-np.mean(x.paleoData_values)
    z /= np.std(x.paleoData_values)
    return z

def add_zscores_to_df(df, key, plot_output=True):
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

    # sort year and data values and obtain resolution
    df['paleoData_values']= df.apply(lambda x: x.paleoData_values[np.argsort(x.year)], axis=1)
    df['year']= df.apply(lambda x: np.round(x.year[np.argsort(x.year)], 2), axis=1)
    df['resolution']= df.year.apply(np.diff).apply(np.unique)
    
    
    return 


def calc_covariance_matrix(df):

    n_recs = len(df)
    
    covariance = np.zeros([n_recs, n_recs])
    overlap    = np.zeros([n_recs, n_recs])
    for ii in range(n_recs):
        for jj in range(ii, n_recs):
            # print(ii, jj)
            
            time_1 = df.iloc[ii].year_hom_avbl
            time_2 = df.iloc[jj].year_hom_avbl
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) # saves intersect between the records
            # print(time_12)
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
    # print(np.sum(overlap<40))
    # print('short records shape: ', overlap.shape[0]*overlap.shape[1])
    print('short records : ', overlap[overlap<40])
    return covariance, overlap


def calc_covariance_matrix_old(df):

    n_recs = len(df)
    
    covariance = np.zeros([n_recs, n_recs])
    overlap    = np.zeros([n_recs, n_recs])
    for ii in range(n_recs):
        for jj in range(ii, n_recs):
            # print(ii, jj)
            
            time_1 = df.iloc[ii].year
            time_2 = df.iloc[jj].year
            time_12, int_1, int_2 = np.intersect1d(time_1, time_2, return_indices=True) # saves intersect between the records
            # print(time_12)
            overlap[ii, jj] = len(time_12)
            overlap[jj, ii] = len(time_12)
    
            data_1 = df.iloc[ii].paleoData_zscores
            data_1 -= np.mean(data_1)
            # data_1 /= np.std(data_1)
            data_2 = df.iloc[jj].paleoData_zscores
            data_2 -= np.mean(data_2)
            # data_2 /= np.std(data_2)
            covariance[ii, jj] = np.cov(data_1[int_1], data_2[int_2], bias=True)[0,1]      #Default normalization (False) is by (N - 1), where N is the number of observations given (unbiased estimate). If bias is True, then normalization is by N.
            covariance[jj, ii] =covariance[ii, jj]
    # print(np.sum(overlap<40))
    # print('short records shape: ', overlap.shape[0]*overlap.shape[1])
    print('short records : ', overlap[overlap<40])
    return covariance, overlap

def PCA(covariance):
    U, s, Vh = np.linalg.svd(covariance) # s eigenvalues, U, Vh rotation matrices

    eigenvalues  = s
    eigenvectors = Vh
    

    return eigenvalues, eigenvectors
    
def fraction_of_explained_var(covariance, eigenvalues, n_recs, key, name):
    sorter = np.argsort(eigenvalues)[::-1] # sort eigenvalues in descending order
    
    explained_var  = eigenvalues[sorter]**2/ (n_recs - 1) 
    
    total_var = np.sum(explained_var)
    frac_explained_var = explained_var / total_var
    
    cum_frac_explained_var = np.cumsum(frac_explained_var)

    fig = plt.figure()
    plt.title(key)
    ax = plt.gca()
    plt.plot(np.arange(len(frac_explained_var))+1, frac_explained_var, label='fraction of explained variance')
    plt.xlim(-1, 10)
    plt.ylabel('fraction of explained variance')
    
    plt.xlabel('PC')
    
    ax1 = ax.twinx()
    ax1.plot(np.arange(len(frac_explained_var))+1, cum_frac_explained_var, ls=':', label='cumulative fraction of explained variance')
    plt.ylabel('cumulative fraction of explained variance') 
    
    f.figsave(fig, 'foev_%s'%key, add=name)


    return frac_explained_var

def plot_PCs(years_hom, eigenvectors, paleoData_zscores_hom, key, name):
    # PCs = np.matmul(eigenvectors, paleoData_zscores_hom.data)
    PCs = np.dot(eigenvectors.T, paleoData_zscores_hom.data)

    Dz   = paleoData_zscores_hom.data
    Dzr  = np.ma.masked_array(np.dot(eigenvectors, PCs), mask=paleoData_zscores_hom.mask)
    # rmsd = np.sqrt(np.mean((Dz - Dzr) ** 2))

    fig = plt.figure()
    plt.suptitle(key)
    ax = plt.subplot(311)
    # for ii in range(paleoData_zscores_hom.shape[0]):
    #     plt.plot(years_hom, paleoData_zscores_hom[ii,:], color='tab:blue', alpha=0.4, lw=1)
    plt.plot(years_hom, np.ma.mean(paleoData_zscores_hom, axis=0), #color='k', 
             zorder=999)
    
    ax.axes.xaxis.set_ticklabels([])
    plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
    plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
    plt.xlim(years_hom[0]-20, years_hom[-1]+20)
    plt.ylabel('paleoData_zscores')
    for ii in range(2):
        ax = plt.subplot(311+ii+1)
        plt.plot(years_hom, PCs[ii])
        if ii==1: plt.xlabel('time (year CE)')
        plt.ylabel('PC %d'%(ii+1))
        plt.axhline(0, color='k', alpha=0.5, lw=0.5)
        plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
        plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
        plt.xlim(years_hom[0]-20, years_hom[-1]+20)
        if ii==0: ax.axes.xaxis.set_ticklabels([])

    f.figsave(fig, 'PCs_%s'%key, add=name)

    plt.figure()
    for ii in range(paleoData_zscores_hom.shape[0]):
        plt.plot(paleoData_zscores_hom[ii,:], Dzr[ii,:],  alpha=0.4, lw=1)
    plt.xlabel('paleoData_zscores')
    plt.ylabel('paleoData_zscores_reconstructed')

    # plt.figure()
    # for ii in range(paleoData_zscores_hom.shape[0]):
    #     plt.plot(np.ma.mean(paleoData_zscores_hom, axis=0), np.ma.mean(Dzr[ii,:], axis=0),  
    #              alpha=0.4, lw=1)
    
    
    fig = plt.figure()
    plt.suptitle(key)
    ax = plt.subplot(211)
    for ii in range(paleoData_zscores_hom.shape[0]):
        plt.plot(years_hom, paleoData_zscores_hom[ii,:], color='tab:blue', alpha=0.4, lw=1)
    plt.plot(years_hom, np.ma.mean(paleoData_zscores_hom, axis=0), color='k', zorder=999)
    
    ax.axes.xaxis.set_ticklabels([])
    plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
    plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
    plt.xlim(years_hom[0]-20, years_hom[-1]+20)
    plt.ylabel('paleoData_zscores')
    
    ax = plt.subplot(212)
    for ii in range(Dzr.shape[0]):
        plt.plot(years_hom, Dzr[ii,:], color='tab:blue', alpha=0.4, lw=1)
    plt.plot(years_hom, np.ma.mean(Dzr, axis=0), color='k', zorder=999)
    
    # plt.plot(years_hom, np.ma.mean(Dz, axis=0))
    ax.axes.xaxis.set_ticklabels([])
    plt.axvline(years_hom[0], color='k', lw=.5, alpha=.5)
    plt.axvline(years_hom[-1], color='k', lw=.5, alpha=.5)
    plt.xlim(years_hom[0]-20, years_hom[-1]+20)
    plt.ylabel('paleoData_zscores \n (reconstructed)')
    

    n_recs = paleoData_zscores_hom.data.shape[0]
    fig = plt.figure()
    plt.suptitle(key)
    for ii in range(2):
        plt.subplot(211+ii)
        plt.plot(range(n_recs), eigenvectors[ii])
        if ii==1: plt.xlabel('rec')
        plt.ylabel('EOF %d load'%(ii+1))
        plt.axhline(0, color='k', alpha=0.5, lw=0.5)
        
    f.figsave(fig, 'EOFloading_%s'%key, add=name)
    
    return PCs, eigenvectors


def smooth(data, time, res):
    smooth_data = []
    smooth_time = []
    for ii in range(0, data.shape[0], 1):
        smooth_data += [np.mean(data[ii:ii+res])]
        smooth_time += [np.mean(time[ii:ii+res])]
    return smooth_time, smooth_data


def geo_plot(df, fs=(9,4.5), dpi=350, **kwargs):


    archive_colour, archives_sorted, proxy_marker = df_colours_markers()
    
    #%% plot the spatial distribution of all records
    proxy_lats = df['geo_meanLat'].values
    proxy_lons = df['geo_meanLon'].values
    
    # plots the map
    fig = plt.figure(figsize=fs, dpi=dpi) #fs=(13,8), dpi=350
    grid = GS(1, 3)
    
    ax = plt.subplot(grid[:, :], projection=ccrs.Robinson()) # create axis with Robinson projection of globe
    
    # ax.stock_img()
    # ax.add_feature(cfeature.LAND) # adds land features
    # ax.coastlines() # adds coastline features


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

    
    return fig

# def archive_colour(df):
    
#     cols  = f.get_colours(range(8), 'tab10', 0, 8)
#     # count archive types
#     archive_count = {}
#     for ii, at in enumerate(set(df['archiveType'])):
#         archive_count[at] = df.loc[df['archiveType']==at, 'paleoData_proxy'].count()
        
#     archive_colour = {'other': cols[-1]}
#     other_archives = []
#     major_archives = []
    
#     sort = np.argsort([cc for cc in archive_count.values()])
#     archives_sorted = np.array([cc for cc in archive_count.keys()])[sort][::-1]
#     for ii, at in enumerate(archives_sorted):
#         print(ii, at, archive_count[at])
#         if archive_count[at]>10:
#             major_archives     +=[at]
#             archive_colour[at] = cols[ii]
#         else:
#             other_archives     +=[at]
#             archive_colour[at] = cols[-1]
#     return archive_colour

def df_colours_markers(db_name='dod2k_dupfree_dupfree'):
    
    cols = [ '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB', '#44AA99']

    df = f.load_compact_dataframe_from_csv(db_name)

    
    # count archive types
    archive_count = {}
    for ii, at in enumerate(set(df['archiveType'])):
        archive_count[at] = df.loc[df['archiveType']==at, 'paleoData_proxy'].count()


    # # count proxy types
    # archive_proxy_count_short = {}
    # for ii, at in enumerate(archives_sorted):
    #     proxy_types   = df['paleoData_proxy'][df['archiveType']==at].unique()
    #     if at not in archive_proxy_count_short:
    #         archive_proxy_count_short[at]={}
    #     for pt in proxy_types:
    #         cc = df['paleoData_proxy'][(df['paleoData_proxy']==pt)&(df['archiveType']==at)].count()
            
    #         if cc<=10:
    #             if 'other %s'%at not in archive_proxy_count_short[at]:
    #                 archive_proxy_count_short[at]['other %s'%at]=0
    #             archive_proxy_count_short[at]['other %s'%at] += cc
    #         else:
    #             archive_proxy_count_short[at]['%s: %s'%(at, pt)] = cc
    
        
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
   
    colors, sm, norm = f.get_colours2(all_EOFs, 
                                colormap='RdBu_r',minval=-0.6,maxval=0.6)
                                   # minval=np.min(all_EOFs), maxval=np.max(all_EOFs)
                                    # minval=np.min([np.min(all_EOFs) -1*np.max(all_EOFs)]), 
                                    # maxval=np.max([np.max(all_EOFs) -1*np.min(all_EOFs)])
                                   #minval=np.max([np.min(all_EOFs), -np.abs(np.max(all_EOFs))]), 
                                   #maxval=np.min([np.max(all_EOFs), np.abs(np.min(all_EOFs))]) 
                                   # ) manual set to make EOF1, EOF2 colorscales equal as well as symmetric
    
    ijk=0

    for key in keys:
        
        marker  = mt[ijk]

        colors = f.get_colours(a[key]*EOFs[key][which_EOF], colormap='RdBu_r',minval=-0.6,maxval=0.6)
                               # minval=np.min(all_EOFs), maxval=np.max(all_EOFs))
                               # minval=np.min([np.min(all_EOFs) -1*np.max(all_EOFs)]), 
                               # maxval=np.max([np.max(all_EOFs) -1*np.min(all_EOFs)])
                               # minval=-0.6 # np.max([np.min(all_EOFs), -np.abs(np.max(all_EOFs))]), 
                               # maxval=+0.6 #np.min([np.max(all_EOFs), np.abs(np.min(all_EOFs))])
                               # ) # manual set to make EOF1, EOF2 colorscales equal as well as symmetric
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