import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io
import logging
import traceback

logger = logging.getLogger(__name__)

class CaptionGenerator:
    def __init__(self, model_path, tokenizer_path, feature_extractor_path, max_length=36, img_size=224):
        self.max_length = max_length
        self.img_size = img_size
        
        # Load the models and tokenizer once during initialization
        try:
            logger.info(f"Loading caption model from {model_path}")
            self.caption_model = load_model(model_path)
            
            logger.info(f"Loading feature extractor from {feature_extractor_path}")
            self.feature_extractor = load_model(feature_extractor_path)
            
            logger.info(f"Loading tokenizer from {tokenizer_path}")
            with open(tokenizer_path, "rb") as f:
                self.tokenizer = pickle.load(f)
        except Exception as e:
            logger.error(f"Error initializing models or tokenizer: {e}")
            logger.error(traceback.format_exc())
            raise

    def preprocess_image(self, image_bytes):
        # Load image from bytes
        img = load_img(io.BytesIO(image_bytes), target_size=(self.img_size, self.img_size))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        return img

    def generate_caption(self, image_bytes):
        try:
            img = self.preprocess_image(image_bytes)
            logger.debug("Image preprocessed successfully")
            
            image_features = self.feature_extractor.predict(img, verbose=0)
            logger.debug("Image features extracted")

            in_text = "startseq"
            for i in range(self.max_length):
                sequence = self.tokenizer.texts_to_sequences([in_text])[0]
                sequence = pad_sequences([sequence], maxlen=self.max_length)
                yhat = self.caption_model.predict([image_features, sequence], verbose=0)
                yhat_index = np.argmax(yhat)
                word = self.tokenizer.index_word.get(yhat_index, None)
                
                if word is None:
                    logger.warning(f"Word index {yhat_index} not found in tokenizer at step {i}")
                    break
                
                in_text += " " + word
                if word == "endseq":
                    break
            
            caption = in_text.replace("startseq", "").replace("endseq", "").strip()
            return caption
        except Exception as e:
            logger.error(f"Error during caption generation: {e}")
            logger.error(traceback.format_exc())
            raise
