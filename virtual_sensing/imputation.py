import numpy as np
import pandas as pd
import random

from tsl import logger
from tsl.utils.python_utils import ensure_list
from tsl.ops.imputation import to_missing_values_dataset

def reorder_columns(matrix):
    top_elements = matrix[0, :]
    sorted_indices = np.argsort(top_elements)
    reordered_matrix = matrix[:, sorted_indices]
    return reordered_matrix
def is_column_present(matrix, column):
    return np.any(np.all(matrix == column[:, np.newaxis], axis=0))
def remove_column(matrix, column):
    mask = np.all(matrix == column[:, np.newaxis], axis=0)
    return matrix[:, ~mask]


def sample_mask(shape,
                p: float = 0.002,
                p_noise: float = 0.,
                simultaneous_fault: bool = False,
                max_seq: int = 1,
                min_seq: int = 1,
                rng: np.random.Generator = None,
                verbose: bool = True):
    if rng is None:
        rand = np.random.random
        randint = np.random.randint
    else:
        rand = rng.random
        randint = rng.integers
    if verbose:
        logger.info(f'Generating mask with: p_noise={p_noise}, p_fault={p}')
    
    # noise mask, random single values
    mask_noise = np.full(shape, False)
    if p_noise!=0:
        mask_noise = rand(shape) < p_noise
    
    # fault mask, intervals in single channels
    mask_fault = np.full(shape, False)
    if p!=0:
        sampled_faults   = rand(shape) < p
        for node in range(sampled_faults.shape[1]):
            idxs = np.array(np.nonzero(sampled_faults[:,node,:]))
            # window length is sampled nodewise
            fault_len = min_seq
            if max_seq > min_seq:
                fault_len = fault_len + int(randint(max_seq - min_seq))
            if simultaneous_fault:
                # extend failures for a window for all channels
                # if one channel breaks, all break
                idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs[0]]) 
                # avoid overlaps
                idxs = np.unique(idxs_ext)              
                # clip with extremes
                idxs = np.clip(idxs, 0, shape[0] - 1)  
                mask_fault[idxs, node, :] = True
            else:
                # extend failures for a window for all channels individually
                # if one channel breaks, the others are unaffected
                times = np.concatenate([np.arange(i, i + fault_len) for i in idxs[0]])
                ch    = np.concatenate([np.ones(fault_len)*d for d in idxs[1]])
                idxs_ext = np.stack((times,ch)).astype(int)
                # avoid overlaps, remove duplicate columns
                idxs = np.unique(idxs_ext, axis=1)     
                # clip with extremes
                idxs = np.clip(idxs, 0, shape[0] - 1) 
                mask_fault[idxs[0], node, idxs[1]] = True

    mask = mask_noise | mask_fault
    
    return mask.astype('uint8')


def sample_virtual_sensors(shape,
                           val_size=0.1,        
                           test_size=0.2):
    # get T,N,D
    T, N, D = shape 
    
    # among all N*D sensors, sample val and test
    all_combinations = [(node, channel) for node in range(N) for channel in range(D)]

    # sample test
    sampled_test = random.sample(all_combinations, int(N * D * test_size))
    # remove test from all combinations
    all_combinations = list(set(all_combinations) - set(sampled_test))
    
    # sample val
    sampled_val = random.sample(all_combinations, int(N * D * val_size))
    # remove val from all combinations
    all_combinations = list(set(all_combinations) - set(sampled_val))
   
    train_s = np.array(all_combinations).T
    val_vs  = np.array(sampled_val).T
    test_vs = np.array(sampled_test).T    
    return train_s, val_vs, test_vs


def custom_val_test(shape,     
                    custom_val=[],
                    custom_test=[]):
    # get T,N,D
    T, N, D = shape 

    # among all N*D sensors, sample val and test
    all_combinations = [(node, channel) for node in range(N) for channel in range(D)]
    all_combinations = set(all_combinations)

    # sample test
    sampled_test = custom_test
    sampled_test = set(sampled_test)
    # remove test from all combinations
    # all_combinations = [elem for elem in all_combinations if elem not in sampled_test]
    all_combinations = all_combinations - sampled_test
    
    # sample val
    sampled_val = custom_val
    sampled_val = set(sampled_val)
    # remove val from all combinations
    # all_combinations = [elem for elem in all_combinations if elem not in sampled_val]
    all_combinations = all_combinations - sampled_val
   
    train_s = np.array(list(all_combinations)).T
    val_vs  = np.array(list(sampled_val)).T
    test_vs = np.array(list(sampled_test)).T    
    return train_s, val_vs, test_vs


def customize_sets(train_s,
                   val_vs,
                   test_vs,
                   custom_vs_train=[],   # list of tuples, those nodes/channel pairs will be in the train
                   custom_vs_val=[],     # list of tuples, those nodes/channel pairs will be in the val
                   custom_vs_test=[]):   # list of tuples, those nodes/channel pairs will be in the test
    for sensor in custom_vs_train:
        # if not present
        sensor = np.array(sensor)
        if not is_column_present(train_s, sensor):
            # add it
            train_s = np.column_stack((train_s, sensor))
            # remove from others
            val_vs  = remove_column(val_vs, sensor)
            test_vs = remove_column(test_vs, sensor)
            logger.info(f'sensor {sensor} moved to train')
            
    for sensor in custom_vs_val:
        # if not present
        sensor = np.array(sensor)
        if not is_column_present(val_vs, sensor):
            # add it
            val_vs = np.column_stack((val_vs, sensor))
            # remove from others
            train_s  = remove_column(train_s, sensor)
            test_vs = remove_column(test_vs, sensor)
            logger.info(f'sensor {sensor} moved to validation')
            
    for sensor in custom_vs_test:
        # if not present
        sensor = np.array(sensor)
        if not is_column_present(test_vs, sensor):
            # add it
            test_vs = np.column_stack((test_vs, sensor))
            # remove from others
            train_s = remove_column(train_s, sensor)
            val_vs  = remove_column(val_vs, sensor)
            logger.info(f'sensor {sensor} moved to test')

    return train_s, val_vs, test_vs




########################################
""" VIRTUAL SENSING """
########################################
def add_virtual_sensors(dataset,
                        p_noise=0.05,     # p of missing one value 
                        p_fault=0.01,     # p of missing one channel for a certain window
                        noise_seed=None, # seed for random mask generation
                        val_size=0.1,    # % of validation sensors
                        test_size=0.2,   # % of testing sensors
                        custom_vs_train=[],   # list of tuples, those nodes/channel pairs will be in the train
                        custom_vs_val=[],     # list of tuples, those nodes/channel pairs will be in the val
                        custom_vs_test=[],    # list of tuples, those nodes/channel pairs will be in the test
                        custom_val=[],
                        custom_test=[],
                        min_seq=1,      # window length of faults 
                        max_seq=10,
                        simultaneous_fault=False,
                        inplace=True):
    if noise_seed is None:
        noise_seed = np.random.randint(1e9)
    # Fix seed for random mask generation
    random_gen = np.random.default_rng(noise_seed)

    # Compute evaluation mask, sample faults and noise (not used)
    shape = (dataset.length, dataset.n_nodes, dataset.n_channels)
    eval_mask = sample_mask(shape,
                            p=p_fault,
                            p_noise=p_noise,
                            min_seq=min_seq,
                            max_seq=max_seq,
                            simultaneous_fault=simultaneous_fault,
                            rng=random_gen)
    # Sample virtual sensors
    if not custom_test:
        # sample virtual sensors from channels in all nodes
        train_s, val_vs, test_vs = sample_virtual_sensors(shape,
                                                          val_size=val_size,
                                                          test_size=test_size)
        # adjust if there are some custom 
        train_s, val_vs, test_vs = customize_sets(train_s, val_vs, test_vs,
                                                  custom_vs_train=custom_vs_train,   
                                                  custom_vs_val=custom_vs_val,     
                                                  custom_vs_test=custom_vs_test)
    else:
        train_s, val_vs, test_vs = custom_val_test(shape,
                                                   custom_val=custom_val,        
                                                   custom_test=custom_test)
    # reorder virtual sensors just for clarity
    train_s = reorder_columns(train_s)
    val_vs  = reorder_columns(val_vs)
    test_vs = reorder_columns(test_vs)
    
    # check that train, val and test sensors are complementary
    combined_array = np.concatenate((train_s, val_vs, test_vs), axis=1)
    unique_columns = np.unique(combined_array, axis=1)
    assert unique_columns.shape[1] == combined_array.shape[1]
    assert combined_array.shape[1] == shape[1]*shape[2]
    logger.info(f'Generating the following validation virtual sensors: {val_vs}')
    logger.info(f'Generating the following test virtual sensors: {test_vs}')    
    
    # create virutual sensing evaluation mask, completely hide channels
    ########################################################################
    val_mask  = np.full(shape, False)
    test_mask = np.full(shape, False)
    val_mask[:, val_vs[0],  val_vs[1]] = True
    test_mask[:, test_vs[0], test_vs[1]] = True

    eval_mask = eval_mask | val_mask.astype(int) | test_mask.astype(int)
    eval_mask = eval_mask & dataset.mask.astype(int)  # assure missing values are set to false in eval_mask
    ########################################################################
    
    # Store evaluation mask in dataset for easy access
    dataset.val_mask = val_mask & dataset.mask        # assure missing values are set to false in val_mask
    dataset.test_mask = test_mask & dataset.mask      # assure missing values are set to false in test_mask

    # Convert to missing values dataset
    dataset = to_missing_values_dataset(dataset, eval_mask, inplace)

    dataset.p_noise = p_noise
    dataset.p_fault = p_fault
    dataset.train_s = train_s
    dataset.val_vs = val_vs
    dataset.test_vs = test_vs

    return dataset





def prediction_dataframe(y, index, columns=None, aggregate_by='mean'):
    """Aggregate batched predictions in a single DataFrame.

    @param (list or np.ndarray) y: the list of predictions.
    @param (list or np.ndarray) index: the list of time indexes coupled with the predictions.
    @param (list or pd.Index) columns: the columns of the returned DataFrame.
    @param (str or list) aggregate_by: how to aggregate the predictions in case there are more than one for a step.
    - `mean`: take the mean of the predictions
    - `central`: take the prediction at the central position, assuming that the predictions are ordered chronologically
    - `smooth_central`: average the predictions weighted by a gaussian signal with std=1
    @return: pd.DataFrame df: the evaluation mask for the DataFrame
    """
    B = y.shape[0]
    T = y.shape[1]
    N = y.shape[2]
    D = y.shape[3]
    dfs = [pd.DataFrame(data=data.reshape((T,N*D)), index=idx,
                        columns=columns) for data, idx in zip(y, index)]
    # dfs = [pd.DataFrame(data=data.reshape(data.shape[:2]),index=idx,
    #                     columns=columns) for data, idx in zip(y, index)]
    df = pd.concat(dfs)
    preds_by_step = df.groupby(df.index)
    # aggregate according passed methods
    aggr_methods = ensure_list(aggregate_by)
    dfs = []
    for aggr_by in aggr_methods:
        if aggr_by == 'mean':
            dfs.append(preds_by_step.mean())
        elif aggr_by == 'central':
            dfs.append(preds_by_step.aggregate(lambda x: x[int(len(x) // 2)]))
        elif aggr_by == 'smooth_central':
            from scipy.signal import gaussian
            dfs.append(
                preds_by_step.aggregate(
                    lambda x: np.average(x, weights=gaussian(len(x), 1))))
        elif aggr_by == 'last':
            # first imputation has missing value in last position
            dfs.append(preds_by_step.aggregate(lambda x: x[0]))
        else:
            raise ValueError("aggregate_by can only be one of "
                             "['mean', 'central', 'smooth_central', 'last']")
    if isinstance(aggregate_by, str):
        return dfs[0]
    return dfs
