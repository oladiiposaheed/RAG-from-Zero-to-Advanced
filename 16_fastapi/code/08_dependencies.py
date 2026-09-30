from fastapi import FastAPI, Depends


app = FastAPI(title='AgriVoice API')

def get_config():
    '''Shared config — used by every endpoint.'''
    
    return {
        'top_k': 4,
        'language': 'english',
    }
    
@app.get('/ask')
def ask(question: str, config: dict = Depends(get_config)):
    '''Answer using the shared config.'''
    
    return {
        'question': question,
        'top_k': config['top_k'],
        'language': config['language'],
    }
    

@app.get('/search')
def search(query: str, config: dict = Depends(get_config)):
    '''Search using the shared config.'''
    
    return {
        'query': query,
        'top_k': config['top_k'],
        'language': config['language']
    }
    
    