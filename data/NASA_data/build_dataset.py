import pickle
import pandas as pd
import numpy as np
import requests
import io

def build_nasa(temporal='hourly',
               start="20190101", end="20191231"):
    # CLIMATE
    # open coordinates from excel coordinates_capitals.xlsx
    file = pd.read_excel('data/NASA_data/coordinates_capitals.xlsx')
    coords = file.loc[:, ['latitude', 'longitude']]
    cities = file.loc[:, 'location']
    lats_samp = coords['latitude'].values
    lons_samp = coords['longitude'].values
    print('LOAD CLIMATE')
    if   temporal == "daily":      
        climatesvar = ["T2M", "T2M_RANGE", "T2M_MAX",     "WS10M", "RH2M", "PRECTOTCORR", "T2MDEW", "CLOUD_AMT", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN"]
        freq='D'
    elif temporal == "hourly":      
        climatesvar = ["T2M",                             "WS10M", "RH2M", "PRECTOTCORR", "T2MDEW",              "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN"]
        freq='H'
    elif temporal == "monthly":                          
        climatesvar = ["T2M", "T2M_RANGE", "T2M_MAX_AVG", "WS2M", "RH2M", "PRECTOTCORR", "T2MDEW", "CLOUD_AMT", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN"]
        freq='M'
    # mmultiindex from product
    ind = pd.date_range(start, end+' 23:00:00' if temporal=='hourly' else end, freq=freq)
    df = pd.DataFrame(index=ind, columns=pd.MultiIndex.from_product([cities, climatesvar], names=['nodes', 'channels']))

    for i, (lat, lon, city) in enumerate(zip(lats_samp, lons_samp, cities)):
        print(f'{i} = ({lat},{lon})')
        
        one_climate = build_one_NASA_fromAPI(lat, lon, climates_var=climatesvar,
                                             temporal=temporal, spatial='point',
                                             start=start, end=end)
        for column_name, time_series in one_climate.iloc[0].items():
            df.loc[:,(city,column_name)] = time_series.values
    
    return df


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



if __name__ == '__main__':
    df = build_nasa(temporal='daily',
                    start="19910101", end="20221231")
    filename = 'data/NASA_data/clmDaily.pkl'
    pickle.dump({'df':df}, open(filename, 'wb'))
    print(f'data saved in {filename}')


# if __name__ == '__main__':
#     df = build_nasa(temporal='hourly',
#                     start="20220101", end="20221231")
#     filename = 'data/NASA_data/clmHourly.pkl'
#     pickle.dump({'df':df}, open(filename, 'wb'))
#     print(f'data saved in {filename}')