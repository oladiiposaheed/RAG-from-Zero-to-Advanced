'''
Middleware is code that runs for EVERY request, before and after your endpoint.
'''

from fastapi import FastAPI, Request

app = FastAPI(title='AgriVoice API')

@app.middleware('http')
async def simple_logger(request: Request, call_next):
    '''
    Print a message for every request
    '''
    
    # 1. This runs BEFORE the endpoint
    print(f'Incoming: {request.method} {request.url.path}')
    
    # 2. Run the actual endpoint
    response = await call_next(request)
    
    # 3. This runs AFTER the endpoint
    print(f'Outgoing: {response.status_code}')

    # 4. Return the response
    return response


@app.get('/')
def read_root():
    
    return {
        'message': 'Hello, AgriVoice!'
    }

@app.get('/health')
def health():
    return {
        'status': 'healthy'
    }