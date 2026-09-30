from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title='AgriVoice API')

class AskRequest(BaseModel):
    '''
    Request body for /ask — with validation.
    '''
    
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description='The user question',
    )
    top_k: int = Field(4, ge=1, le=20, description='Number of chunks to retrieve')
    language: str = Field('english', description='Response language',)
    

@app.post('/ask')
def ask(request: AskRequest):
    '''
    Answer a question
    '''
    
    return {
        'question': request.question,
        'top_k': request.top_k,
        'language': request.language,
        'answer': f'[Would answer in {request.language} with {request.top_k} chunks]'
    }
    
    
    