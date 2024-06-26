# -*- coding: utf-8 -*-

import os
import trimesh
import numpy as np
from scipy.io import savemat

import lib
from . import preload

def preprocess_impl(hash_t: str, label: int):
    mesh_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.RAW_INPUT.value)
    
    raw_mesh = trimesh.load(mesh_path, force='mesh')
    # connected components get largest
    mesh = sorted(raw_mesh.split(only_watertight=False).tolist(), key=lambda x: x.vertices.shape[0])[-1] 
    
    # registration
    tmat, _ = trimesh.registration.mesh_other(mesh, preload.models[f'Enamel_{label}']['standard'], 50, icp_first=5, icp_final=10)
    mesh.apply_transform(tmat)

    # scale to 1/10
    mesh.apply_scale(1. / 10)
    
    # only need surface pnts
    sample_point_count = 500000
    points, face_indices = mesh.sample(sample_point_count, return_index=True)
    normals = mesh.face_normals[face_indices]
    
    points_near_surface = np.concatenate([points, normals], axis=1)

    dataset_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.DATASET.value)
    savemat(dataset_path, {'p': points_near_surface})

    transf_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, '_local_transform.mat')
    savemat(transf_path, {'local_transform': np.linalg.inv(tmat)})