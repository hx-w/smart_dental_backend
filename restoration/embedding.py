# -*- coding: utf-8 -*-

import os

import numpy as np
import torch

import lib
from .networks import dataset
from .preload import models


def embedding_impl(hash_t: str):
    DentalType = 'Enamel_15'
    
    mat_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.DATASET.value)
    
    dt = dataset.PointCloud_wo_FreePoints(instance_idx=0, pointcloud_path=mat_path, **models[DentalType]['meta_params'])
    dtt = dataset.PointCloudMulti([dt])
    
    _embed = models[DentalType]['model'].template_code.clone().detach().unsqueeze(0).repeat(1, 1)
    _embed.requires_grad = True
    
    trans = torch.nn.Parameter(torch.zeros([1, 3], dtype=_embed.dtype).cuda(), requires_grad=True)
    optim = torch.optim.Adam(lr=models[DentalType]['meta_params']['lr'], params=[_embed, trans])
    
    for i in range(len(dtt)):
        model_input, gt = dtt[i]
        model_input = {k: v.cuda() for k, v in model_input.items()}
        gt = {k: v.cuda() for k, v in gt.items()}
        model_input['coords'] = model_input['coords'] + trans.unsqueeze(1)
        losses = models[DentalType]['model'].embedding(_embed, model_input, gt)
        train_loss = sum(list(losses.values()))
        optim.zero_grad()
        train_loss.backward()
        optim.step()

    fitted_embedding = _embed.detach().cpu().numpy()
    fitted_trans = trans.detach().cpu().numpy()
    
    embedding_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.EMBEDDING.value)
    
    np.savez(embedding_path, fitted_embedding, fitted_trans)
