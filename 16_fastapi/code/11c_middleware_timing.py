import time
from fastapi import FastAPI, Request

app = FastAPI(title='AgriVoice API')

@app.middleware('hhtp')
async def timing_middleware(request: Request, call_next):
    '''Measure how long each request takes.'''
    
    # 1. Record the start time (in seconds)
    start_time = time.time()
    
    # 2. Run the endpoint
    response = await call_next(request)
    
    # 3. Record the end time
    end_time = time.time()

    # 4. Calculate the elapsed time
    elapsed = end_time - start_time

    # 5. Convert to milliseconds (1 second = 1000 ms)
    elapsed_ms = elapsed * 1000

    # 6. Print the result
    print(f'⏱️  {request.method} {request.url.path} -> {elapsed_ms:.1f}ms')

    # 7. Also add it as a header
    response.headers['X-Process-Time'] = f'{elapsed_ms:.1f}ms'

    return response



@app.get('/')
def read_root():
    return {'message': 'Hello, AgriVoice!'}

@app.get('/slow')
def slow_endpoint():
    '''Pretend to do heavy work — takes about 1 second.'''
    time.sleep(1)
    return {'status': 'done after 1 second'}