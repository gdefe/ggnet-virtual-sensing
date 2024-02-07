import random
import io
import requests
import pickle
import os

# For extracting string patterns
import glob

# For interacting with tables
import pandas as pd
import numpy as np

# For interacting with NetCDF files
import xarray as xr

# For date and time definition
from datetime import timedelta

import warnings
warnings.filterwarnings('ignore')

def build_photovoltaic(n_points = 0, random_state=1234, 
                      start="20190101", end="20191231",
                      module_id = 0):
    
    print('LOAD PHOTOVOLTAIC')
    # Define the number of seconds in an hour for readable conversion from seconds to hours
    # because lead times are stored in seconds. However, code will be easier to read in hours.
    sec_in_hour = 3600
    
    # Select a module ID
    # Define the root directory of data
    if module_id>=0 and module_id<10:
        data_dir_path = f'data/Photovoltaic/module_0{module_id}'
    else:
        data_dir_path = f'data/Photovoltaic/module_{module_id}'
    coordinate_path = 'data/Photovoltaic/coordinates.nc'
    
    # Read coordinates
    coords = xr.open_dataset(coordinate_path)
    
    # Extract all NetCDF files for a particular module ID
    nc_files = glob.glob(os.path.join(data_dir_path, '*.nc'))
    nc_files.sort()
    
    # print('There are {} daily files for the module {:02d}.'.format(len(nc_files), module_id))
    
    # Establish a connection to all files in parallel
    # nc_files = list of files paths
    # decode_cf = whether to decode the dataset using the Climate and Forecast Metadata Conventions, which provide a standard for encoding geoscientific data.
    # concat_dim = dimension along which the datasets from files in the list should be concatenated
    # combine = how the function should combine the datasets: 'nested' -> each dataset is a separate dimension coordinate.
    ds_raw = xr.open_mfdataset(nc_files, decode_cf=True, concat_dim='init_time', combine='nested', parallel=False)
    ds = ds_raw.merge(coords)
    
    ds_mean = ds.median('member', skipna=True)
    
    lats = ds_mean.Ys.data
    lons = ds_mean.Xs.data
    
    # randomly sample n points
    random.seed(random_state)
    idxs = random.sample(range(len(lons)), n_points)
    lats_samp = [lats[i] for i in idxs]
    lons_samp = [lons[i] for i in idxs]
    
    # PHOTO        
    df_photo = pd.DataFrame()
    # ds_mean_daily = ds_mean.sum('lead_time')
    i_lats=[]
    i_lons=[]
    for i, (lat, lon) in enumerate(zip(lats_samp, lons_samp)):
        where = np.argwhere(ds_mean['Xs'].values==lons_samp[i])[0]
        assert len(where)==1
        i_lons.append(where[0])
        where = np.argwhere(ds_mean['Ys'].values==lats_samp[i])[0]
        i_lats.append(where[0])
    assert (i_lons == i_lats)
    indices = i_lons
    ds_mean = ds_mean.sel(grid=indices)
    ds_mean = ds_mean.drop(['Xs','Ys'])
    
    # sampled points
    cols = []
    for i, (lat, lon) in enumerate(zip(lats_samp, lons_samp)):
       cols.append(f'({round(lat,2)},{round(lon,2)})')

    # pandas dataframe
    df = ds_mean.to_dataframe()
    df = df.unstack('grid')
    df.columns = cols
    df.index = df.index.set_levels(df.index.levels[1] / sec_in_hour, level='lead_time')
    
    # sum over lead time
    ind = pd.date_range(start, end, freq='D')
    df = df.groupby(level='init_time').sum()
    # df = df.sum(axis=0, level=0)
    df = df.reindex(ind)

    # CLIMATE
    # round coordinates to 2 digits precision
    lats_samp = np.array(lats_samp).round(2)
    lons_samp = np.array(lons_samp).round(2)
    print('LOAD CLIMATE')
    climatesvar = ["T2M", "T2M_RANGE", "T2M_MAX", "WS10M", "RH2M", "PRECTOTCORR", "T2MDEW","CLOUD_AMT", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN"]
    # mmultiindex from product
    df_photo = pd.DataFrame(index=ind, columns=pd.MultiIndex.from_product([cols, climatesvar+['photovoltaic']], names=['nodes', 'channels']))

    for i, (lat, lon) in enumerate(zip(lats_samp, lons_samp)):
        print(f'{i} = ({lat},{lon})')
        
        one_climate = build_one_NASA_fromAPI(lat, lon-360, climates_var=climatesvar,
                                             temporal='daily', spatial='point',
                                             start=start, end=end)
        df_photo.loc[:,(f'({lat},{lon})','photovoltaic')] = df.loc[:,f'({lat},{lon})'].values
        for column_name, time_series in one_climate.iloc[0].items():
            df_photo.loc[:,(f'({lat},{lon})',column_name)] = time_series.values
    
    return df_photo



def build_one_NASA_fromAPI(lat:float,
                          lon:float,
                          climates_var:list,
                          temporal='daily', spatial='point',
                          start="20100101", end="20121231", 
                          community = "sb", file_format = "CSV"):
        
    df_out = pd.DataFrame(index = [f'({lat},{lon})'], columns=climates_var)
    
    v = climates_var
    complete_string = str(v[0])
    for i in range(1, len(v)):     
        complete_string += "," + v[i]
        
    # API request    
    if temporal=='climatology' or temporal=='ann_mean':
        url = "https://power.larc.nasa.gov/api/temporal/"+'climatology'+"/"+spatial+"?latitude="+str(lat)+"&longitude="+str(lon)+"&community="+community+"&parameters="+complete_string+"&format="+file_format+"&user=iclruser&header=false"
    elif temporal=='daily':
        url = "https://power.larc.nasa.gov/api/temporal/"+temporal+"/"+spatial+"?start="+start+"&end="+end+"&latitude="+str(lat)+"&longitude="+str(lon)+"&community="+community+"&parameters="+complete_string+"&format="+file_format+"&user=iclruser&header=false"
    elif temporal=='hourly':
        url = "https://power.larc.nasa.gov/api/temporal/"+temporal+"/"+spatial+"?start="+start+"&end="+end+"&latitude="+str(lat)+"&longitude="+str(lon)+"&time-standard=UTC"+"&community="+community+"&parameters="+complete_string+"&format="+file_format+"&user=iclruser&header=false"
    elif temporal=='monthly':
        start = start[:4]
        end = end[:4]
        url = "https://power.larc.nasa.gov/api/temporal/"+temporal+"/"+spatial+"?start="+start+"&end="+end+"&latitude="+str(lat)+"&longitude="+str(lon)+"&community="+community+"&parameters="+complete_string+"&format="+file_format+"&user=iclruser&header=false"
    else:
        print('invalid climate mode')
    r = requests.get(url)  
    print(" request status code =", r.status_code)
    
    # extract content
    if(r.status_code == 200):
        r = r.content
        df = pd.read_csv(io.StringIO(r.decode('utf-8')), on_bad_lines='skip')
    
        if temporal=='ann_mean':
            # reorder dataset
            df.index = list(df['PARAMETER'].values)
            df = df.drop([col for col in df.columns if col!='ANN'], axis=1)
            df = df.loc[climates_var].T
            # attach to output
            df_out.loc[f'({lat},{lon})'] = df.loc['ANN']
            
        if temporal=='climatology':
            # reorder dataset
            df = df.drop(['ANN'], axis=1)
            df.index = list(df['PARAMETER'].values)
            df = df.drop(['PARAMETER'], axis=1)
            df = df.loc[climates_var].T
            # condense columns into one row
            df['proxy'] = 1
            new_row = df.groupby('proxy').agg(lambda x:list(x)).reset_index(drop=True)
            # attach to output
            df_out.loc[f'({lat},{lon})'] = new_row.values
    
        if temporal=='monthly':
            # reorder dataset
            df = df.drop(['ANN'], axis=1)
            for var in df['PARAMETER'].unique():
                df_var = df[df['PARAMETER']==var]
                df_var.index = df_var['YEAR']
                df_var = df_var.drop(['PARAMETER','YEAR'],axis=1)
                # attach to output
                df_out.loc[f'({lat},{lon})',var] = np.concatenate(df_var.values)
                
        if temporal=='daily':
            # reorder dataset
            df = df.drop(['YEAR', 'MO', 'DY'], axis=1)
            # condense columns into one row
            df['proxy'] = 1
            new_row = df.groupby('proxy').agg(lambda x:list(x)).reset_index(drop=True)
            # attach to output
            df_out.loc[f'({lat},{lon})'] = new_row.values
    
        if temporal=='hourly':
            # reorder dataset
            df = df.drop(['YEAR', 'MO', 'DY', 'HR'], axis=1)
            # condense columns into one row
            df['proxy'] = 1
            new_row = df.groupby('proxy').agg(lambda x:list(x)).reset_index(drop=True)
            # attach to output
            df_out.loc[f'({lat},{lon})'] = new_row.values    

        if temporal!='ann_mean':
            df_out = df_out.applymap(lambda x: pd.Series(x))
        
    return df_out




# main
if __name__ == '__main__':
    module_id = 0
    seed_list = [1,2,3,4,5]  # sampling seeds
    npoints_list = [50]      # number of nodes
    df_photo = {}
    for npoints in npoints_list:
        for seed in seed_list:
            df_photo[seed] = build_photovoltaic(n_points=npoints, random_state=seed, 
                                               start="20190101", end="20191231",
                                               module_id=module_id)
        pickle.dump(df_photo, open(f'data/Photovoltaic/photovoltaic{npoints}_s.pkl', 'wb'))
    