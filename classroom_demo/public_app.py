"""Preset-only public classroom workbench. No uploads or shared inspection history."""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from threading import Lock
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageEnhance
from industrial_inspection.pipeline import inspect_image

ROOT=Path(__file__).resolve().parent
CASES={'tile':('tile','crack_000.png'), 'bottle':('bottle','broken_large_000.png')}
SLOTS=Lock()
app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)


@app.get('/health')
def health():
    return {'status':'ok','edition':'public-classroom-demo'}


@app.get('/',response_class=HTMLResponse)
def home():
    return HTMLResponse((ROOT/'index.html').read_text(encoding='utf-8'))


@app.get('/sample/{case}')
def sample(case:str):
    if case not in CASES:raise HTTPException(404)
    product, file=CASES[case]
    return FileResponse(ROOT/'samples'/product/file, media_type='image/png')


def encode(image):
    b=BytesIO()
    image.thumbnail((640,640))
    image.save(b,format='JPEG',quality=89)
    return 'data:image/jpeg;base64,'+base64.b64encode(b.getvalue()).decode('ascii')


@app.post('/run/{case}')
def run(case:str):
    if case not in CASES:raise HTTPException(404)
    if not SLOTS.acquire(blocking=False):raise HTTPException(429,'A demonstration is running. Try again in a moment.')
    try:
        product, file=CASES[case]
        with Image.open(ROOT/'samples'/product/file) as opened:im=opened.convert('RGB')
        outputs=[('Feature memory','patch',im),('Pixel baseline','pixel',im)] if case=='tile' else [
                 ('Original photo','patch',im),('Brightness 0.1','patch',ImageEnhance.Brightness(im).enhance(.1))]
        rows=[]
        for name,method,image in outputs:
            overlay,report=inspect_image(image,ROOT/'models'/f'{product}_{method}')
            rows.append({'name':name,'image':encode(image.copy()),'overlay':encode(overlay.copy()),'report':report})
        return JSONResponse({'id':uuid.uuid4().hex[:8],'case':case,'rows':rows},headers={'Cache-Control':'no-store'})
    finally:
        SLOTS.release()

