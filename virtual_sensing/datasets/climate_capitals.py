import os
import pickle
import numpy as np
import pandas as pd

from tsl import logger

from tsl.datasets.prototypes import DatetimeDataset
from tsl.utils import download_url, extract_zip
from tsl.ops.similarities import gaussian_kernel

class ClimateCapitals(DatetimeDataset):
    r"""Climatic variables readings from satellite observations and reanalizes.
    
    Data have been collected from https://power.larc.nasa.gov/ 
    in correspondence of the 235 world capitals

    'clmHourly' -> 1 years of hourly data:
        + Time steps:     8760
        + Nodes:          235
        + Channels:       7     
        + Sampling rate:  hourly 
        + Time interval:  2022-01-01 to 2022-12-31 
        + Missing values: YES
    
    'clmDaily' -> 30 years of daily data:
        + Time steps:     10958
        + Nodes:          235  
        + Channels:       10
        + Sampling rate:  daily 
        + Time interval:  1991-01-01 to 2022-12-31 
        + Missing values: YES
    
    'clmClm' -> averaged climatological values:
        + Time steps:     12 
        + Nodes:          235
        + Channels:       10
        + Sampling rate:  climatology 
        + Time interval:  average over 20-30 years (depending on the channel)
        + Missing values: NO
        
    """

    # TO DO
    # url = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"

    similarity_options = {"distance"}

    def __init__(self, name, root=None, impute_zeros=True, freq=None, normalize=True, 
                 remove_data_frac=0.0,
                 remove_vs_frac=0.0):
        self.name = name
        self.root = root
        self.norm = normalize
        self.remove_data_frac = remove_data_frac
        self.remove_vs_frac = remove_vs_frac
        # load dataset and missing values mask
        df, dist, mask = self.load(impute_zeros=impute_zeros)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name=self.name)
        self.add_covariate('dist', dist, pattern='n n')

    @property
    def raw_file_names(self):
        return [self.name + '.pkl', 'coordinates_capitals.xlsx']

    @property
    def required_file_names(self):
        return [self.name + '.pkl', 'coordinates_capitals.xlsx', self.name + '_dist.npy']

    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)

    def build(self) -> None:
        self.maybe_download()
        # Build distance matrix
        logger.info('Building distance matrix...')
        raw_coord_path = os.path.join(self.root_dir, 'coordinates_capitals.xlsx')
        coords = pd.read_excel(raw_coord_path).loc[:, ['latitude', 'longitude']]

        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(coords, to_rad=True).values

        # num_coordinates = len(coords)
        # dist = np.zeros((num_coordinates, num_coordinates))
        # # Calculate the distances and populate the distance matrix
        # for i in range(num_coordinates):
        #     for j in range(num_coordinates):
        #         coord1 = (coords['latitude'][i], coords['longitude'][i])
        #         coord2 = (coords['latitude'][j], coords['longitude'][j])
        #         distance = haversine(coord1, coord2, unit=Unit.METERS)
        #         dist[i, j] = distance

        # Save to built directory
        path = os.path.join(self.root_dir, self.name + '_dist.npy')
        np.save(path, (dist).round(3))
        # np.save(path, (dist/1e3).round(3))
        # Remove raw data
        self.clean_downloads()

    def load_raw(self):
        self.maybe_build()
        path = os.path.join(self.root_dir, self.name + '.pkl')
        # load from file
        with open(path, "rb") as file:
            df = pickle.load(file)['df']
            file.close()
        path = os.path.join(self.root_dir, self.name + '_dist.npy')
        dist = np.load(path)
        return df, dist

    def load(self, impute_zeros=True):
        df, dist = self.load_raw()
        # # check absence of nan
        # assert pd.isna(df).sum().sum() == 0

        # randomly set to nan a fraction of the datapoints
        if self.remove_data_frac > 0:
            df[np.random.rand(*df.shape) < self.remove_data_frac] = -999

        # randomly set to nan a fraction of columns
        if self.remove_vs_frac > 0:
            df.iloc[:, np.random.rand(df.shape[1]) < self.remove_vs_frac] = -999

        # replace missing values with nans for normalization
        mask = (df != -999)
        df[~mask] = np.nan
        # normalize
        if self.norm:
            df = self.normalize(df)
        # impute zeros. If not, missing values stay nan
        if impute_zeros:
            df[~mask] = 0.

        # contaminate the dataset with extreme values for debugging purposes
        # df.iloc[:,3] = -999999999
        # df.iloc[:,6] = -999999999 
        
        return df, dist, mask

    def normalize(self, df_):
        df = df_.copy()
        locations = [loc for loc in df.columns.get_level_values('nodes')]
        channels = list(set([var for var in df.columns.get_level_values('channels')]))
        for var in channels:
            # get all nodes for one channel
            one_channel_df = df.loc[:, (locations, var)]
            # mean and std avoiding missing values
            mean = np.nanmean(one_channel_df.values)
            std = np.nanstd(one_channel_df.values)
            # standardize and reassign to dataframe
            df.loc[:, (locations, var)] = (one_channel_df - mean) / std
        return df

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
