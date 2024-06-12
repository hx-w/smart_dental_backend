# -*- coding: utf-8 -*-

import os
import hashlib
from enum import Enum, unique


@unique
class DentalFileT(Enum):
    BASE: str = 'static'
    RAW_INPUT: str = 'basic.ply'
    
    # for restoration
    DATASET: str = 'dataset.mat'
    EMBEDDING: str = 'embedding.npz'
    RESTORED: str = 'restored.ply'
    
    # for segmentation
    STANDARD_JAW: str = 'standard_jaw.ply'


def hash_bytes(inp_bytes: bytes) -> str:
    _salt = 'deadbeef'.encode('utf-8')
    return hashlib.sha1(inp_bytes + _salt).hexdigest()

def check_filepath_exist(hash_t: str, file_name: str) -> bool:
    return os.path.isfile(os.path.join(DentalFileT.BASE.value, hash_t, file_name))

async def save_raw_file(hash_t: str, file_bytes: bytes):
    _basedir = os.path.join(DentalFileT.BASE.value, hash_t)
    os.makedirs(_basedir, exist_ok=True)

    with open(os.path.join(_basedir, DentalFileT.RAW_INPUT.value), 'wb') as wf:
        wf.write(file_bytes)
