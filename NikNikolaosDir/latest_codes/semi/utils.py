import os
import logging
import numpy as np

import torch

def makedirs(_dir):
    if not os.path.exists(_dir):
        os.makedirs(_dir)

def create_logger(save_path, file_type):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    file_name = os.path.join(save_path, file_type + '_log.txt')

    if not getattr(logger, 'handler_set', False):

        cs = logging.StreamHandler()
        cs.setLevel(logging.DEBUG)
        logger.addHandler(cs)

        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(logging.INFO)

        logger.addHandler(fh)

        logger.handler_set = True

    return logger


def log_sum_exp(logits, mask=None, inf=1e7):
    if mask is not None:
        logits = logits * mask - inf * (1.0 - mask)
        max_logits = logits.max(1)[0]
        return ((logits - max_logits.expand_as(logits)).exp() * mask).sum(1).log().squeeze() + max_logits.squeeze()
    else:
        max_logits = logits.max(1)[0]
        return ((logits - max_logits.unsqueeze(1).expand_as(logits)).exp()).sum(1).log().squeeze() + \
            max_logits.squeeze()

def log_sum_exp_0(logits):
    max_logits = logits.max()
    return (logits - max_logits.expand_as(logits)).exp().sum().log() + max_logits

def tensor2Var(tensor, requires_grad=False):
    if torch.cuda.is_available() and not tensor.is_cuda:
        tensor = tensor.cuda()
    if not requires_grad:
        tensor.requires_grad = False

    return tensor

def np2Var(array, requires_grad=False):
    tensor = torch.from_numpy(array)

    return tensor2Var(tensor, requires_grad=requires_grad)
