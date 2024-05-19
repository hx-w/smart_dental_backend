# -*- coding: utf-8 -*-

import os
import yaml
import torch
from .networks.net import FlexField

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


models = {
    'Enamel_15': None
}

for mname in models.keys():
    models[mname] = {
        'model': None,
        'meta_params': None
    }
    
    with open(os.path.join('restoration', 'ckpts', mname, 'config.yml'), 'r') as stream:
        models[mname]['meta_params'] = yaml.safe_load(stream)

    
    models[mname]['model'] = FlexField(**models[mname]['meta_params'])
    state_dict = torch.load(models[mname]['meta_params']['checkpoint_path'])
    filtered_state_dict = { k: v for k, v in state_dict.items() if k.find('detach')==-1 }
    
    models[mname]['model'].load_state_dict(filtered_state_dict)
    models[mname]['model'].cuda()


print('== PyTorch models loaded ==')
