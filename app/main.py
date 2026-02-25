from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import os
import logging
import traceback
from .model_utils import CaptionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Image Caption Generator")

# Global variable for lazy loading
caption_gen = None

# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "model.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(MODELS_DIR, "feature_extractor.keras")

STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def get_caption_generator():
    global caption_gen

    if caption_gen is None:
        logger.info("Loading models lazily...")

        if not all(os.path.exists(p) for p in [MODEL_PATH, TOKENIZER_PATH, FEATURE_EXTRACTOR_PATH]):
            logger.error("Model files missing!")
            raise Exception("Model files not found")

        caption_gen = CaptionGenerator(
            MODEL_PATH,
            TOKENIZER_PATH,
            FEATURE_EXTRACTOR_PATH
        )

        logger.info("Models loaded successfully.")

    return caption_gen


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        caption_generator = get_caption_generator()

        contents = await file.read()
        logger.info(f"Generating caption for image of size {len(contents)} bytes")

        caption = caption_generator.generate_caption(contents)

        logger.info(f"Generated caption: {caption}")
        return {"caption": caption}

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")