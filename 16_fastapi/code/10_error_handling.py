from fastapi import FastAPI, HTTPException


app = FastAPI(title='AgriVoice API')

# Fake disease database
DISEASES = {
    'cassava-mosaic': 'Use disease-free cuttings, plant resistant varieties.',
    'maize-smut': 'Plant resistant hybrids, rotate crops, remove galls.',
    'rice-blast': 'Use resistant varieties, apply fungicides.',
}


@app.get('/disease/{disease_name}')
def get_disease(disease_name: str, severity: str = 'moderate'):
    '''
    Look up a disease — returns 404 for unknown, 400 for bad severity.
    '''
    
    # Validate severity first
    if severity not in ('mild', 'moderate', 'severe'):
        raise HTTPException(
            status_code=400,
            detail=f'Invalid severity "{severity}". Allowed: mild, moderate, severe',
        )
    
    # Check if the disease exists in database
    if disease_name not in DISEASES:
        raise HTTPException(
            status_code=404,
            detail=f'Disease "{disease_name}" not found',
        )
    
    return {
        'disease': disease_name,
        'treatment': DISEASES[disease_name],
    }    
    