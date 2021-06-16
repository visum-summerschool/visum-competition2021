import ast
import json
from collections import OrderedDict

import numpy as np
import torch


def fix_random_seeds(seed):
    """
    Fix random seeds for reproducibility.

    Args:
        seed (int): an integer representing the seed value

    Returns:
        None

    Notes: random seeds are only fixed when the specified seed is an integer.
    """

    if isinstance(seed, int):
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)


def check_isrgb(im):
    """ Check if im is an RGB image. Otherwise, convert from grayscale to a 3-channel stacked image """
    
    im_sze = im.shape
    
    if len(im_sze) != 3:
        im = np.stack((im,) * 3, axis=-1)
    
    return im


def pdist(vectors):
    """ Distance matrix computation: pairwise distance between vectors in batch
    Args:
        vectors (torch.tensor): batch of vector/embedding of shape [batch_size x embedding_size]

    Returns:
        (torch.tensor): distance matrix of shape [batch_size x batch_size].

    """

    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def tokens_str(tokens, word_map):
    """ Converter function - from list of tokens to list of strings/words

    Args:
        tokens (list): list of tokens.
        word_map (dict): dictionary mapping from word to token id.

    Returns:
        (list): list of strings.

    """

    return [list(word_map.keys())[list(word_map.values()).index(i)] for i in tokens]


def from_np_array(array_string):
    """ Converter function - from array_string to numpy array

    Args:
        array_string (str): array_string.

    Returns:
        (numpy.array): numpy array.

    """

    array_string = ','.join(array_string.replace('[ ', '[').split())
    return arraystr_to_array(array_string)


def arraystr_to_array(array_string):
    """ Convert a array_string to numpy array

    Args:
       array_string (str): array_string.

    Returns:
       (numpy.array): numpy array.

    """

    return np.array(ast.literal_eval(array_string))


def cat2id(df):
    """ mapping from category to id

    Args:
       df (dataframe): df.

    Returns:
       category_map (dict): category_map = {'category':category_id}.

    Notes: df needs to have column category (eg: df_products)
    """

    category_id = 0
    category_map = {}

    # create category map from unique categories in the dataframe
    for cat in df['category'].unique():
        category_map[cat] = category_id
        category_id += 1

    return category_map


def read_json(fname):
    with open(fname, 'rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
