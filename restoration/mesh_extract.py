# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from scipy.io import loadmat

import lib
from .networks import sdf_meshing
from .preload import models


def mesh_extract_impl(hash_t: str, label: str) -> str:
    DentalType = f'Enamel_{label}'
    
    embedding_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.EMBEDDING.value)
    
    arrs = np.load(embedding_path)
    embedding, trans = arrs['arr_0'], arrs['arr_1']
    
    restore_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.RESTORED.value)
    
    transf_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, '_local_transform.mat')
    tmat = loadmat(transf_path)['local_transform']

    sdf_meshing.create_mesh(
        models[DentalType]['model'],
        restore_path,
        embedding = torch.from_numpy(embedding.astype(np.float32)).cuda(),
        N = 128,
        get_color = False,
        scale = 0.1, # 缩小0.1，即放大10
        offset = trans,
        transf = tmat
    )
    
    del embedding, trans
    
    return restore_path
