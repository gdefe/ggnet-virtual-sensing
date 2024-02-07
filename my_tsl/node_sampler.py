import torch
from tsl.data.batch import StaticBatch
from itertools import islice

def sample_nodes_groups(batch, n):
    input_list = list(range(batch.x.size()[2]))
    groups = [list(islice(input_list, i, i+n)) for i in range(0, len(input_list), n)]
    return groups

def sample_nodes_rnd(batch, n):
    # Sample n random nodes from nodes
    return list(torch.randperm(batch.x.size()[2])[:n])

def sample_nodes(batch_, sampled_indices):
    """Sample n nodes from a batch of graphs.
    Args:
        batch_ (tsl.data.StaticBatch): Batch of graphs.
        n (int): Number of nodes to sample.
    Returns:
        (tsl.data.StaticBatch, torch.Tensor): Sampled batch of graphs and indices of sampled nodes."""
    if sampled_indices == range(batch_.num_nodes):
        return batch_
    else:
        raise NotImplementedError
        # batch = batch_.clone()
        # sampled_data = StaticBatch(input={'x': batch.input.x[:, :, sampled_indices, :], 
        #                                 'mask': batch.input.mask[:, :, sampled_indices, :],
        #                                 'sampled_idx': sampled_indices},
        #                             target={'y': batch.target.y[:, :, sampled_indices, :]},
        #                             mask = batch.input.mask[:, :, sampled_indices, :],
        #                             val_mask = batch.val_mask[:, :, sampled_indices, :],
        #                             test_mask = batch.test_mask[:, :, sampled_indices, :],
        #                             has_mask = batch.has_mask,
        #                             transform = batch.transform)
        # if hasattr(batch, 'eval_mask'):
        #     sampled_data.eval_mask = batch.eval_mask[:, :, sampled_indices, :]
        # if hasattr(batch, 'original_mask'):
        #     sampled_data.original_mask = batch.original_mask[:, :, sampled_indices, :]
        
        # sampled_data.sampled = True

        # return sampled_data