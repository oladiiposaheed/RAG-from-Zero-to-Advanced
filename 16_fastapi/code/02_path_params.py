from fastapi import FastAPI


app = FastAPI(title='AgriVoice API')

@app.get('/')
def read_root():
    return {
        'message': 'Hello AgriVoice!',
    }
    
@app.get('/items/{item_id}')
def read_item(item_id):
    '''
    Return an item by ID — the ID comes from the URL path.
    '''
    
    return {
        'item_id': item_id,
        'type': type(item_id).__name__,
    }
    
@app.get('/crops/{crop_name}')
def read_crop(crop_name: str):
    '''
    Return info about a crop by name — string path parameter.
    '''
    
    return {
        'crop': crop_name,
        'advice': f'{crop_name.title()} grows well in Nigeria',
    }
    
    