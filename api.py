import io
import json
import logging
import os
import re
import time

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

IMG_SIZE = 224

app = FastAPI(title="NutriSafe ML API")

# ── Nutrition model (TensorFlow) ──────────────────────────────────────────────

def build_nutrition_model():
    base = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dense(4, activation="linear"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.build((None, IMG_SIZE, IMG_SIZE, 3))
    model.load_weights("best_nutrition_model.h5")
    return model

nutrition_model = build_nutrition_model()

# ── Gemini (ingredient detection) ────────────────────────────────────────────

api_key = os.getenv("GEMINI_API_KEY", "")
gemini_client = None
if api_key:
    gemini_client = genai.Client(api_key=api_key)
    log.info("Gemini model loaded.")
else:
    log.warning("GEMINI_API_KEY tidak ditemukan. Deteksi bahan dinonaktifkan.")

INGREDIENT_PROMPT = """Lihat gambar makanan ini dengan seksama.
Identifikasi semua bahan makanan yang terlihat pada gambar.
Kembalikan HANYA array JSON berisi nama bahan dalam Bahasa Indonesia, tanpa teks lain.
Contoh output: ["Nasi", "Ayam Goreng", "Wortel", "Kacang", "Susu"]
Jika tidak ada makanan yang terdeteksi, kembalikan array kosong: []"""

def detect_mime_type(image_bytes: bytes) -> str:
    if image_bytes[:2] == b'\xff\xd8':
        return "image/jpeg"
    if image_bytes[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_bytes[:4] == b'RIFF' and image_bytes[8:12] == b'WEBP':
        return "image/webp"
    return "image/jpeg"

MAX_RETRIES = 3
RETRY_DELAYS = [1, 2, 4]  # detik (exponential backoff)

def detect_ingredients(image_bytes: bytes) -> tuple[list[str], bool]:
    """Returns (ingredients, gemini_failed)."""
    if gemini_client is None:
        log.warning("Gemini client None, skip deteksi bahan.")
        return [], True

    mime_type = detect_mime_type(image_bytes)
    img_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[INGREDIENT_PROMPT, img_part],
            )
            text = response.text.strip()
            log.info(f"Gemini raw response (attempt {attempt}): {text}")

            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if isinstance(result, list):
                    return result, False
            log.warning(f"Gemini response tidak mengandung JSON array: {text}")
            return [], False
        except Exception as e:
            is_last = attempt == MAX_RETRIES
            if not is_last and ("503" in str(e) or "UNAVAILABLE" in str(e) or "429" in str(e)):
                delay = RETRY_DELAYS[attempt - 1]
                log.warning(f"Gemini attempt {attempt} gagal ({e}), retry dalam {delay}s...")
                time.sleep(delay)
            else:
                log.warning(f"Gemini gagal setelah {attempt} attempt: {e}")
                return [], True

    return [], True

# ── Helpers ───────────────────────────────────────────────────────────────────

def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def get_status_gizi(calories: float) -> str:
    if calories < 300:
        return "Kurang"
    elif calories <= 700:
        return "Seimbang"
    return "Lebih"

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "gemini": gemini_client is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img_array = preprocess(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File bukan gambar yang valid: {e}")

    # Nutrition dari TF model
    pred = nutrition_model.predict(img_array)[0]
    calories     = round(max(0.0, float(pred[0])), 2)
    fat          = round(max(0.0, float(pred[1])), 2)
    carbohydrate = round(max(0.0, float(pred[2])), 2)
    protein      = round(max(0.0, float(pred[3])), 2)

    # Ingredient detection dari Gemini (dengan retry)
    ingredients, gemini_failed = detect_ingredients(contents)

    return {
        "calories":             calories,
        "fat":                  fat,
        "carbohydrate":         carbohydrate,
        "protein":              protein,
        "status_gizi":          get_status_gizi(calories),
        "detected_ingredients": ingredients,
        "gemini_failed":        gemini_failed,
    }
