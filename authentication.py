# -*- coding: utf-8 -*-

from fastapi import Header, HTTPException, Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials


security = HTTPBasic()

def verify_token(credentials: HTTPBasicCredentials = Depends(security)) -> bool:
    '''
    A very simple impl for only 1 token
    '''
    _uname = 'GET MY TOKEN'
    _upswd = 'b19aa580-90e1-4f32-b065-e5f4c1b9c2cd'
    
    if credentials.username != _uname or credentials.password != _upswd:
        raise HTTPException(status_code=401, detail='Authentication failed')

    return True
