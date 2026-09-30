'''
File upload — receive images on the server.
'''

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException


app = FastAPI(title='AgriVoice API')

# Folder to save uploaded files
UPLOAD_DIR = Path(r'C:\Users\USER\rag_course\16_fastapi\uploads')
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed image types
ALLOWED_TYPES = {'image/jpeg', 'image/png', 'image/jpg'}

@app.post('/upload')
async def upload_images(file: UploadFile = File(...)):
    '''
    Receive an image and save it to disk
    '''
    
    # 1. Validate the file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f'Invalid type: {file.content_type} Allowd: {ALLOWED_TYPES}',
        )
    
    # 2. Read the file bytes
    contents = await file.read()
    
    # 3. Save it to disk
    save_path = UPLOAD_DIR / file.filename
    save_path.write_bytes(contents)
    
    # 4. Return info about what was saved
    return {
        'filename': file.filename,
        'size_bytes': len(contents),
        'content_type': file.content_type,
        'saved_to': str(save_path),
    }
    
    