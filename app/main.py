from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from contextlib import asynccontextmanager
import os
import logging
import traceback
from .model_utils import CaptionGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable for the caption generator
caption_gen = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global caption_gen
    
    # Use absolute path for models directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    
    MODEL_PATH = os.path.join(MODELS_DIR, "model.keras")
    TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
    FEATURE_EXTRACTOR_PATH = os.path.join(MODELS_DIR, "feature_extractor.keras")
    
    logger.info(f"Checking for models in {MODELS_DIR}")
    
    if all(os.path.exists(p) for p in [MODEL_PATH, TOKENIZER_PATH, FEATURE_EXTRACTOR_PATH]):
        try:
            logger.info("Loading models...")
            caption_gen = CaptionGenerator(MODEL_PATH, TOKENIZER_PATH, FEATURE_EXTRACTOR_PATH)
            logger.info("Models loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            logger.error(traceback.format_exc())
            caption_gen = None
    else:
        logger.error(f"Error: Model files not found in {MODELS_DIR}")
        for p in [MODEL_PATH, TOKENIZER_PATH, FEATURE_EXTRACTOR_PATH]:
            if not os.path.exists(p):
                logger.error(f"Missing file: {p}")
    
    yield
    # Clean up on shutdown if needed
    caption_gen = None

app = FastAPI(title="Image Caption Generator", lifespan=lifespan)

# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_caption(file: UploadFile = File(...)):
    if not caption_gen:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        logger.info(f"Generating caption for image of size {len(contents)} bytes")
        caption = caption_gen.generate_caption(contents)
        logger.info(f"Generated caption: {caption}")
        return {"caption": caption}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Note: Using 127.0.0.1 for local access
    uvicorn.run(app, host="127.0.0.1", port=8000)
