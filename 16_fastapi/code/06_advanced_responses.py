from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title='AgriVoice API')


class Chunk(BaseModel):
    '''
    A single retrieved chunk
    '''
    
    text: str
    source: str
    score: float = Field(ge=0.0, le=1.0)
    
    
class AskResponse(BaseModel):
    '''
    Full /ask response — nested chunks and metadata.
    '''
    
    question: str
    answer: str
    language: str
    chunks: List[Chunk]
    request_id: str = id
    

@app.post('/ask', response_model=AskResponse)
def ask(question: str, language: str = 'english'):
    '''
    # Simulate retrieved chunks
    '''
    
    chunks = [
        Chunk(text='Use disease-free cuttings', source='crop_disease.pdf', score=0.95),
        Chunk(text='Plant resistant varieties', source='agriculture.html', score=0.89),
        Chunk(text='Remove infected plants early', source='crop_disease.pdf', score=0.82),
    ]
    
    return AskResponse(
        question=question,
        answer=f'To treat cassava mosaic disease in {language}...',
        language=language,
        chunks=chunks,
        request_id='req_abc123',
    )
