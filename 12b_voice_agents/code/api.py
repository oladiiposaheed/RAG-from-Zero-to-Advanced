'''
AgriVoice API — FastAPI wrapper.

Run:
    uvicorn api:app --reload --port 5000
'''

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from faster_whisper import WhisperModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import os
import tempfile
from pathlib import Path
import uuid
import time
import requests


# Load .env
load_dotenv()


# Global state — loaded once at startup, reused by every request
state = {}


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class TextAskRequest(BaseModel):
    '''User input — question and target language.'''
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description='The farmer question in any supported language',
    )
    language: str = Field(
        'English',
        description='Response language: English, Yoruba, Hausa, Igbo, Nigerian Pidgin',
    )


class TextAskResponse(BaseModel):
    '''AI output — answer plus metadata.'''
    question_english: str
    answer: str
    language: str
    chunks_used: int


# ─────────────────────────────────────────────
# Lifespan — startup + shutdown
# ─────────────────────────────────────────────

# Load once at startup, reuse forever
@asynccontextmanager
async def lifespan(app):
    '''
    Runs once when the server starts and once when it shuts down.
    Load all heavy resources here so requests are fast.
    '''

    print('🚀 AgriVoice API starting...')

    # OpenAI client — used by translation and fallback TTS
    state['client'] = OpenAI()

    # LLM used by the RAG chain
    state['llm'] = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    # Embeddings for the Chroma store
    state['embeddings'] = OpenAIEmbeddings()

    # Domain knowledge base (farming docs)
    state['vectorstore'] = Chroma(
        persist_directory=r'C:\Users\USER\rag_course\chroma_db_domain',
        embedding_function=state['embeddings'],
    )
    state['retriever'] = state['vectorstore'].as_retriever(search_kwargs={'k': 4})

    # Load faster-whisper inside lifespan
    # Local STT model — transcribes farmer audio
    state['stt_model'] = WhisperModel('base', device='cpu', compute_type='int8')
    
    # RAG prompt — short answers because they will be spoken aloud
    rag_prompt = ChatPromptTemplate.from_messages([
        ('system',
         'You are AgriVoice, a farming assistant for Nigerian farmers. '
         'Answer using ONLY the context below. Keep it under 3 sentences. '
         'If the answer is not in the context, say "I don\'t know".'),
        ('human', 'Context:\n{context}\n\nQuestion: {question}'),
    ])

    # The full RAG chain
    state['rag_chain'] = rag_prompt | state['llm'] | StrOutputParser()

    # Terminology store — correct crop/disease translations
    state['terminology_store'] = Chroma(
        persist_directory=r'C:\Users\USER\rag_course\chroma_db_terminology',
        embedding_function=state['embeddings'],
        collection_name='terminology',
    )
    state['terminology_retriever'] = state['terminology_store'].as_retriever(
        search_kwargs={'k': 5}
    )

    # Confirm what loaded
    print(f'✅ Domain chunks:     {state["vectorstore"]._collection.count()}')
    print(f'✅ Terminology docs:  {state["terminology_store"]._collection.count()}')
    print('✅ AgriVoice API ready')

    yield

    print('AgriVoice API shutting down')


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(
    title='AgriVoice API',
    description='Voice-first farming assistant for Nigerian farmers.',
    version='0.1.0',
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# Translation helpers
# ─────────────────────────────────────────────

# Retrieve terminology from Chroma for correct crop names
def get_relevant_terms(text: str) -> str:
    docs = state['terminology_retriever'].invoke(text)
    return '\n'.join(f'- {doc.page_content}' for doc in docs)


# Translate English text to a Nigerian language
def translate(text: str, target_lang: str) -> str:
    if target_lang == 'English':
        return text

    terms = get_relevant_terms(text)
    system = (
        f'You are an expert translator for Nigerian languages.\n'
        f'Translate the following English text to {target_lang}.\n\n'
        f'MANDATORY VOCABULARY:\n{terms}\n\n'
        f'RULES:\n'
        f'1. Output ONLY {target_lang} — no English, except proper nouns.\n'
        f'2. Use correct tone marks and diacritics.\n'
        f'3. Output ONLY the translation. No explanations.\n'
    )
    response = state['client'].chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# Translate FROM a Nigerian language TO English
def translate_to_english(text: str, source_lang: str) -> str:
    if source_lang == 'English':
        return text

    terms = get_relevant_terms(text)
    system = (
        f'You are an expert translator for Nigerian languages.\n'
        f'Translate the following {source_lang} text to English.\n\n'
        f'MANDATORY VOCABULARY:\n{terms}\n\n'
        f'RULES:\n'
        f'1. Output ONLY English.\n'
        f'2. Use the terminology above to identify crops correctly.\n'
        f'3. Output ONLY the translation.\n'
    )
    response = state['client'].chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


# The /voice/ask endpoint

# Generate speech with YarnGPT — async job flow
def yarngpt_tts(text: str, voice: str) -> bytes:
    
    yarngpt_key = os.getenv('YARNGPT_API_KEY')
    
    submit = requests.post(
        'https://yarngpt.ai/api/v1/tts',
        headers={
            'Authorization': f'Bearer {yarngpt_key}',
            'Content-Type': 'application/json',
            'Idempotency-Key': str(uuid.uuid4()),
        },
        json={'text': text, 'voice': voice},
        timeout=30,
    )
    if submit.status_code != 202:
        raise RuntimeError(f'YarnGPT submit failed: {submit.status_code}')
    
    job_id = submit.json()['job_id']
    
    for _ in range(10):
        time.sleep(2)
        status = requests.get(
            f'https://yarngpt.ai/api/v1/status/{job_id}',
            headers={'Authorization': f'Bearer {yarngpt_key}'},
            timeout=30,
        )
        
        data = status.json()
        s = data.get('status')
        
        if s == 'completed':
            audio = requests.get(data['audio_url'], timeout=30)
            return audio.content
        
        if s == 'failed':
            raise RuntimeError(f'YarnGPT job failed: {data}')
        
    raise TimeoutError('YarnGPT job timed out')
                          
# Language → YarnGPT voice mapping
YARNGPT_VOICES = {
    'English': 'idera',
    'Yoruba': 'idera',
    'Igbo': 'chinenye',
    'Hausa': 'umar',
    'Nigerian Pidgin': 'tayo',
}

# Speak text — try YarnGPT, fall back to OpenAI TTS 
def tts_speak(text: str, language: str = 'English') -> bytes:
    try:
        voice = YARNGPT_VOICES.get(language, 'idera')
        print(f'   🔊 YarnGPT ({voice})...')
        return yarngpt_tts(text, voice)
    
    except Exception as e:
        print(f'   ⚠️  YarnGPT failed: {e}')
        print('   🔊 Falling back to OpenAI TTS...')
        
    response = state['client'].audio.speech.create(
        model='tts-1',
        voice='alloy',
        input=text
    )
    return response.content


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

# Root — quick check
@app.get('/')
def read_root():
    return {
        'service': 'agrivoice-api',
        'status': 'running',
        'docs': '/docs',
    }


# Health check — for monitoring and load balancers
@app.get('/health')
def health_check():
    return {
        'status': 'healthy',
        'service': 'agrivoice-api',
        'version': '0.1.0',
        'domain_chunks': state['vectorstore']._collection.count(),
        'terminology_docs': state['terminology_store']._collection.count(),
    }


# Text question → multilingual answer
@app.post('/text/ask', response_model=TextAskResponse)
def text_ask(request: TextAskRequest):
    # 1. Translate question to English
    question_en = translate_to_english(request.question, request.language)

    # 2. Retrieve context, generate English answer
    docs = state['retriever'].invoke(question_en)
    context = '\n\n'.join(doc.page_content for doc in docs)
    answer_en = state['rag_chain'].invoke({
        'context': context,
        'question': question_en,
    })

    # 3. Translate answer back to the user's language
    answer_local = translate(answer_en, request.language)

    # 4. Return everything
    return TextAskResponse(
        question_english=question_en,
        answer=answer_local,
        language=request.language,
        chunks_used=len(docs),
    )
    

# Supported languages for validation
SUPPORTED_LANGS = list(YARNGPT_VOICES.keys())

# Audio file in -> audio file out
@app.post(
    '/voice/ask',
    response_class=Response,
    responses={200: {'content': {'audio/mpeg': {}}}},
)
async def voice_ask(
    audio: UploadFile = File(..., description='WAV or MP3 of the farmer speaking'),
    language: str = Form('English', description='Target response language')
):
    # Validate language
    if language not in SUPPORTED_LANGS:
        raise HTTPException(
            status_code=400,
            detail=f'Unsupported language: {language}. '
            f'Use on of: {SUPPORTED_LANGS}',
        )
        
    # Save the upload to a temp file — faster-whisper needs a file path
    suffix = Path(audio.filename or 'input.wav').suffix or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
        
    try:
        # 1. STT — transcribe the audio
        segments, info = state['stt_model'].transcribe(tmp_path, beam_size=5)
        question = ' '.join(seg.text.strip() for seg in segments)
        print(f'📝 STT ({info.language}): {question}')
        
        if not question.strip():
            raise HTTPException(status_code=400, detail='No speech detected')
        
        # 2. Translate → English
        question_en = translate_to_english(question, language)
        
        # 3. RAG in English
        docs = state['retriever'].invoke(question_en)
        context = '\n\n'.join(doc.page_content for doc in docs)
        answer_en = state['rag_chain'].invoke({
            'context': context,
            'question': question_en,
        })
        
        # 4. Translate answer -> user's language
        answer_local = translate(answer_en, language)
        
        # 5. TTS — speak it
        audio_bytes = tts_speak(answer_local, language=language)
        
        return Response(content=audio_bytes, media_type='audio/mpeg')
    
    finally:
        # Always clean up the temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        
