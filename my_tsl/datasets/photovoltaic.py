import os
import pickle
import numpy as np
import pandas as pd

from tsl import logger

from tsl.datasets.prototypes import DatetimeDataset
from tsl.utils import download_url, extract_zip
from tsl.ops.similarities import gaussian_kernel

class Photovoltaic(DatetimeDataset):
    r"""Climate data + photovoltaic.
        + Time steps: 365 
        + Nodes: variables    
        + Channels: 11        
        + Sampling rate: 1 day
        + Time interval:  
        + Missing values: YES
    """

    # TO DO
    # url = "https://drive.switch.ch/index.php/s/Z8cKHAVyiDqkzaG/download"

    similarity_options = {"distance"}

    def __init__(self, root=None, npoints=50, sampling_seed=1,
                 impute_zeros=True, freq=None, normalize=False):
        self.root = root
        self.norm = normalize
        self.npoints = npoints
        self.sampling_seed = sampling_seed
        # load dataset and missing values mask
        df, dist, mask = self.load(impute_zeros=impute_zeros)
        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name='photovoltaic')
        self.add_covariate('dist', dist, pattern='n n')

    @property
    def raw_file_names(self):
        return [f'photovoltaic{self.npoints}_s.pkl']

    @property
    def required_file_names(self):
        return [f'photovoltaic{self.npoints}_s.pkl', f'photovoltaic{self.npoints}_dist.npy']

    def download(self) -> None:
        path = download_url(self.url, self.root_dir)
        extract_zip(path, self.root_dir)
        os.unlink(path)

    def build(self) -> None:
        self.maybe_download()
        # Build distance matrix
        logger.info('Building distance matrix...')
        d = pd.read_pickle(os.path.join(self.root_dir, f'photovoltaic{self.npoints}_s.pkl'))
        df = d[self.sampling_seed]

        # Get coordinates
        ind = df.columns.get_level_values('nodes')
        coords = [tuple(map(float, node.strip('()').split(','))) for node in ind]
        # Create a DataFrame with two columns: 'latitude' and 'longitude'
        coords = pd.DataFrame(coords, columns=['latitude', 'longitude'])
        # Drop duplicate rows
        coords.drop_duplicates(inplace=True)
        coords.reset_index(drop=True, inplace=True)
        # lon - 360
        coords['longitude'] = coords['longitude'] - 360
        
        from tsl.ops.similarities import geographical_distance
        dist = geographical_distance(coords, to_rad=True).values

        # Save to built directory
        path = os.path.join(self.root_dir, f'photovoltaic{self.npoints}_dist.npy')
        np.save(path, dist.round(3))
        # Remove raw data
        self.clean_downloads()

    def load_raw(self):
        self.maybe_build()
        path = os.path.join(self.root_dir, f'photovoltaic{self.npoints}_s.pkl')
        # load from file
        d = pd.read_pickle(path)
        df = d[self.sampling_seed]
        # cast all elements to float32
        df = df.astype(np.float32)
        path = os.path.join(self.root_dir, f'photovoltaic{self.npoints}_dist.npy')
        dist = np.load(path)
        return df, dist

    def load(self, impute_zeros=True):
        df, dist = self.load_raw()
        # assure that channels do not contains zeros only
        df.loc[:, (df.isin([0]).any()) & (df.sum()==0)] = np.nan
        # missing values are nans at this stage
        # replace missing values with nans for normalization
        mask = ~pd.isna(df)
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
