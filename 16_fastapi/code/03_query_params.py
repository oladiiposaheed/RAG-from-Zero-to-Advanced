from fastapi import FastAPI

app = FastAPI(title='AgriVoice API')

@app.get('/search')
def search(
    q: str,
    limit: int = 10,
    language: str = 'english'
):
    '''
    Search with optional limit and language filters.
    '''
    
    return {
        'query': q,
        'limit': limit,
        'language': language,
        'results': [f'result {i+1} for {q}' for i in range(limit)],
    }
    