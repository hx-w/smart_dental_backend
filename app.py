# -*- coding: utf-8 -*-

from fastapi import (
    FastAPI, UploadFile, File,
    HTTPException, Depends
)

import lib
from authentication import verify_token
import restoration


smtapp = FastAPI()

@smtapp.get('/')
async def root():
    return {"message": "Hello World"}

@smtapp.post('/register')
async def api_register_geometry(file: UploadFile = File(...), _=Depends(verify_token)):
    file_bytes = await file.read()
    hash_t = lib.hash_bytes(file_bytes)
    
    await lib.save_raw_file(hash_t, file_bytes)

    del file_bytes
    return {'token': hash_t}

@smtapp.post('/dental/restoration/preprocess')
async def api_dental_restoration_preprocess(token: str, _=Depends(verify_token)):
    if not lib.check_filepath_exist(token, lib.DentalFileT.RAW_INPUT.value):
        raise HTTPException(status_code=403, detail='Token expired')
    
    try:
        restoration.preprocess_impl(token)
    except Exception as e:
        raise HTTPException(status_code=501, detail=f'Preprocess err: {e}')
    
    return {'timecost': 0}

@smtapp.post('/dental/restoration/embedding')
async def api_dental_restoration_embedding(token: str, _=Depends(verify_token)):
    if not lib.check_filepath_exist(token, lib.DentalFileT.DATASET.value):
        raise HTTPException(status_code=403, detail='Token expired')
    
    try:
        restoration.embedding_impl(token)
    
    except Exception as e:
        raise HTTPException(status_code=501, detail=f'Preprocess err: {e}')
    
    return {'timecost': 0}

@smtapp.post('/dental/restoration/extract')
async def api_dental_restoration_extract(token: str, _=Depends(verify_token)):
    if not lib.check_filepath_exist(token, lib.DentalFileT.EMBEDDING.value):
        raise HTTPException(status_code=403, detail='Token expired')
    
    try:
        restoration.mesh_extract_impl(token)
    
    except Exception as e:
        raise HTTPException(status_code=501, detail=f'Preprocess err: {e}')
    
    return {'timecost': 0}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(smtapp, host="127.0.0.1", port=8000)
