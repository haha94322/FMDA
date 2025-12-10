import torch

import pickle
import numpy as np
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def todevice(tensor, device):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        assert isinstance(tensor[0], torch.Tensor)
        return [todevice(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)

def save_obj(obj, filename, verbose=True):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    if verbose:
        logger.info("Saved object to %s." % filename)


def load_obj(filename, verbose=True):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    if verbose:
        logger.info("Load object from %s." % filename)
    return obj