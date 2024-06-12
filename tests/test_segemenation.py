# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import trimesh
from segmentation import inference_impl
import lib


def test_segmentation():
    token = '68c24f65122911c271cadd63ef0af689e0049994'
    
    # for ind in range(3):
    #     _start = time.time()
    #     labels = inference_impl(token, 'upper', bool(ind % 2))
    #     _end = time.time()

    #     print(len(labels), _end - _start)

    
    jaw_path = os.path.join(lib.DentalFileT.BASE.value, token, lib.DentalFileT.RAW_INPUT.value)
    mesh = trimesh.load(jaw_path, force='mesh')
    labels = inference_impl(token, 'upper', True)
    
    valid_mask = np.array(labels) == 15

    # # mesh.visual.vertex_colors[valid_mask] = [255, 0, 0, 255]

    retrived_faces = []

    for fid, face in enumerate(mesh.faces):
        if valid_mask[face[0]] and valid_mask[face[1]] and valid_mask[face[2]]:
            retrived_faces.append(fid)

    retrived_faces = np.array(retrived_faces)
    print(retrived_faces)

    retrived_mesh = mesh.submesh([retrived_faces], append=True)

    retrived_mesh.export('tests/static/retrived_15.obj')

if __name__ == '__main__':
    test_segmentation()
