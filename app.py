# -*- coding: utf-8 -*-

from fastapi import (
    FastAPI, UploadFile, File,
    HTTPException, Depends
)
from fastapi.responses import FileResponse

import lib
from authentication import verify_token
import restoration
import segmentation


smtapp = FastAPI()


@smtapp.post('/register')
async def api_register_geometry(file: UploadFile = File(...), _=Depends(verify_token)):
    file_bytes = await file.read()
    hash_t = lib.hash_bytes(file_bytes)
    
    await lib.save_raw_file(hash_t, file_bytes)

    del file_bytes
    return {'token': hash_t}

@smtapp.post('/segmentation')
async def api_dental_segementation(token: str, jaw_kind: str, do_registration: bool=True, _=Depends(verify_token)):
    if not lib.check_filepath_exist(token, lib.DentalFileT.RAW_INPUT.value):
        raise HTTPException(status_code=403, detail='Token expired')
    if jaw_kind not in ['lower', 'upper']:
        raise HTTPException(status_code=403, detail='Jaw kind should be `lower` or `upper`')
    try:
        labels = segmentation.inference_impl(token, jaw_kind, do_registration)
    except Exception as e:
        raise HTTPException(status_code=501, detail=f'Preprocess err: {e}')
    
    return {'labels': labels}

@smtapp.post('/restoration/preprocess')
async def api_dental_restoration_preprocess(token: str, label: str, _=Depends(verify_token)):
    if not lib.check_filepath_exist(token, lib.DentalFileT.RAW_INPUT.value):
        raise HTTPException(status_code=403, detail='Token expired')
    try:
        restoration.preprocess_impl(token, label)
    except Exception as e:
        raise HTTPException(status_code=501, detail=f'Preprocess err: {e}')
    
    return {'timecost': 0}

@smtapp.post('/restoration/embedding')
async def api_dental_restoration_embedding(token: str, label: str, _=Depends(verify_token)):
    if not lib.check_filepath_exist(token, lib.DentalFileT.DATASET.value):
        raise HTTPException(status_code=403, detail='Token expired')
    try:
        restoration.embedding_impl(token, label)
    
    except Exception as e:
        raise HTTPException(status_code=501, detail=f'Embedding err: {e}')
    
    return {'timecost': 0}

@smtapp.post('/restoration/extract')
async def api_dental_restoration_extract(token: str, label: str, _=Depends(verify_token)):
    if not lib.check_filepath_exist(token, lib.DentalFileT.EMBEDDING.value):
        raise HTTPException(status_code=403, detail='Token expired')
    try:
        ripsta_path = restoration.mesh_extract_impl(token, label)
    
    except Exception as e:
        raise HTTPException(status_code=501, detail=f'Extract err: {e}')
    
    return FileResponse(ripsta_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(smtapp, host="127.0.0.1", port=8000)
