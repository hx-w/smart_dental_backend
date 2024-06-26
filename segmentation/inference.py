# -*- coding: utf-8 -*-

import os
import trimesh
import lib
from .preload import model, standard_jaw


def inference_impl(hash_t: str, jaw_kind='upper', do_reg=True):
    jaw_path = os.path.join(lib.DentalFileT.BASE.value, hash_t, lib.DentalFileT.RAW_INPUT.value)
    
    jaw_mesh = trimesh.load(jaw_path, force='mesh')

    if do_reg:
        jaw_mesh.apply_obb()
        
        tmat, _ = trimesh.registration.mesh_other(jaw_mesh, standard_jaw, 50, icp_first=5, icp_final=10)

        jaw_mesh.apply_transform(tmat)
        
        print('[seg / reg] finish')
    
    labels = model.process(jaw_mesh, jaw_kind)

    print('[seg] finish')
    return labels
