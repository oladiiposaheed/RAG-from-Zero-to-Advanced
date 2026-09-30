'''
generate a unique ID per request, time it, log it, and add both as headers.
'''


import time
import uuid
from fastapi import FastAPI, Request

app = FastAPI(title='AgriVoice API')

@app.middleware('http')
async def request_middleware(request: Request, call_next):
    '''
    Assign a unique ID to each request, time it, and log it.
    '''
    
    # 1. Generate a short unique ID (8 characters)
    request_id = str(uuid.uuid4())[:8]
    
    # 2. Attach the ID to the request so endpoints can use it
    request.state.request_id = request_id
    
    # 3. Record the start time
    start_time = time.time()
    
    # 4. Run the endpoint
    response = await call_next(request)
    
    # 5. Calculate elapsed time in milliseconds
    elapsed_ms = (time.time() - start_time) * 1000
    
    # 6. Log everything on one line
    print(f'[{request_id}] {request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.1f}ms)')

    # 7. Add both headers to the response
    response.headers['X-Request-ID'] = request_id
    response.headers['X-Process-Time'] = f'{elapsed_ms:.1f}ms'

    return response


@app.get('/')
def read_root(request: Request):
    '''Root endpoint — shows the request ID in the body too.'''
    return {
        'message': 'Hello, AgriVoice!',
        'request_id': request.state.request_id,
    }


@app.get('/slow')
def slow_endpoint(request: Request):
    '''Pretend to do heavy work.'''
    time.sleep(1)
    return {
        'status': 'done',
        'request_id': request.state.request_id,
    }
    