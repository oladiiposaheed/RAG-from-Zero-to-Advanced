from fastapi import FastAPI, Request

app = FastAPI(title='AgriVoice API')

# A counter that lives outside the middleware
request_count = 0

@app.middleware('http')
async def counting_middleware(request: Request, call_next):
    '''Count every request '''
    
    # Use the global counter
    global request_count
    
    # 1. Increment the counter
    request_count = request_count + 1
    print(f'Request #{request_count}: {request.method} {request.url.path}')
    
    # 2. Run the endpoint
    response = await call_next(request)
    
    # 3. Add the count as a header
    response.headers['X-Request-Count'] = str(request_count)
    
    # 4. Return the response
    return response


@app.get('/')
def read_root():
    return {'message': 'Hello, AgriVoice!'}


@app.get('/health')
def health():
    return {'status': 'healthy'}


@app.get('/count')
def show_count():
    '''
    Return how many requests have hit the server
    '''
    
    return {
        'total_requests': request_count
    }
    
    
@app.get('/headers')
def show_headers():
    '''Return the count — also check the X-Request-Count header in DevTools.
    '''
    return {'count_from_body': request_count}