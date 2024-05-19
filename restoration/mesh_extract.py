# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

import lib
from .networks import sdf_meshing
from .preload import models


def mesh_extract_impl(hash_t: str):
    DentalType = 'Enamel_15'
    
    embedding_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.EMBEDDING.value)
    
    arrs = np.load(embedding_path)
    embedding, trans = arrs['arr_0'], arrs['arr_1']
    
    restore_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.RESTORED.value)
    
    sdf_meshing.create_mesh(
        models[DentalType]['model'],
        restore_path,
        embedding = torch.from_numpy(embedding.astype(np.float32)).cuda(),
        N = 128,
        get_color = False,
        scale = 0.1, # 缩小0.1，即放大10
        offset = trans
    )
    
    