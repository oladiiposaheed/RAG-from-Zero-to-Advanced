from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def read_root():
    return {'message': 'Hello, AgriVoice!'}


# Add a Health Endpoint
@app.get('/health')
def health_check():
    '''
    Health check endpoint — for monitoring.
    '''
    
    return {
        'status': 'healthy',
        'service': 'agrivoice-api'
    }
    