import asyncio
from fastapi import FastAPI
import time


app = FastAPI(title='AgriVoice API')


@app.get('/sync')
def sync_endpoint():
    '''
    Blocking endpoint — blocks a worker for 1 second.
    '''
    
    time.sleep(1)
    return {
        'mode': 'sync',
        'waited': '1 second'
    }
    
    
@app.get('/async')
async def async_endpoint():
    '''
    Non-blocking endpoint — frees the worker while waiting.
    '''
    
    await asyncio.sleep(3)
    return {
        'mode': 'sync',
        'waited': '1 second'
    }
    
    
    