import asyncio
from fastapi import FastAPI


app = FastAPI(title='AgriVoice API')

async def fake_openai_call(question: str) -> str:
    '''Simulate a slow OpenAI call.'''
    
    # Pause without blocking the event loop
    await asyncio.sleep(1)
    return f'LLM answer to: {question}'


async def fake_chroma_call(question: str) -> list:
    '''Simulate a slow Chroma query.'''
    await asyncio.sleep(1)
    return [f'LLM answer to: {question}']


@app.get('/ask')
async def ask(question: str):
    '''
    Call OpenAI and Chroma CONCURRENTLY
    '''
    
    # Start both tasks at the same time
    llm_task = asyncio.create_task(fake_openai_call(question))
    chroma_task = asyncio.create_task(fake_chroma_call(question))
    
    # Wait for both to finish
    llm_answer, chunks = await asyncio.gather(llm_task, chroma_task)

    return {
        'answer': llm_answer,
        'chunks': chunks,
        'total_wait': '1 second'
    }    


