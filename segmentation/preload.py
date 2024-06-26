# -*- coding: utf-8 -*-

import os
import trimesh
import lib
from .inference_pipeline_mid import InferencePipeLine
from .predict_utils import ScanSegmentation

inference_config = {
    "fps_model_info":{
        "model_parameter" :{
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": "segmentation/ckpts/0707_cosannealing_val"
    },

    "boundary_model_info":{
        "model_parameter":{
            "input_feat": 6,
            "stride": [1, 4, 4, 4, 4],
            "nstride": [2, 2, 2, 2],
            "nsample": [36, 24, 24, 24, 24],
            "blocks": [2, 3, 4, 6, 3],
            "block_num": 5,
            "planes": [32, 64, 128, 256, 512],
            "crop_sample_size": 3072,
        },
        "load_ckpt_path": "segmentation/ckpts/0711_bd_cbl_aug_test_val"
    },

    "boundary_sampling_info":{
        "bdl_ratio": 0.7,
        "num_of_bdl_points": 20000,
        "num_of_all_points": 24000,
    },
}

model = ScanSegmentation(InferencePipeLine(inference_config))

print('== PyTorch [segmentation] models loaded ==')

_standard_jaw_path = os.path.join(lib.DentalFileT.STANDARD.value, lib.DentalFileT.STANDARD_JAW.value)

standard_jaw = trimesh.load(_standard_jaw_path, force='mesh')
