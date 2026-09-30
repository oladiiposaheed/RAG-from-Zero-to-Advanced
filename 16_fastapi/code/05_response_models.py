from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title='AgriVoice API')

class AskResponse(BaseModel):
    '''
    The shape of every /ask response.
    '''
    
    question: str
    answer: str
    top_k: int
    
    
@app.post('/ask', response_model=AskResponse)
def ask(question: str):
    '''
    Answer a question — returns a typed AskResponse
    '''
    
    return AskResponse(
        question=question,
        answer=f'Answer to: {question}',
        top_k=4,
    )
    