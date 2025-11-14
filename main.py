import os
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io

from database import db, create_document, get_documents
from schemas import AuraPhoto

app = FastAPI(title="Aura Vision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Aura Vision Backend is running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response

class AuraResponse(BaseModel):
    aura_label: str
    aura_score: float
    dominant_colors: List[str]
    notes: str
    record_id: str


def extract_dominant_colors(image: Image.Image, k: int = 5) -> List[str]:
    # Downscale for speed
    img = image.copy()
    img.thumbnail((256, 256))
    # Convert to RGB and get colors
    img = img.convert("RGB")
    pixels = list(img.getdata())

    # Simple k-means like quantization using PIL getcolors fallback
    max_colors = 256 * 256
    colors = img.getcolors(maxcolors=max_colors)
    if not colors:
        # Fallback: sample pixels
        step = max(1, len(pixels)//5000)
        colors = {}
        for i in range(0, len(pixels), step):
            colors[pixels[i]] = colors.get(pixels[i], 0) + 1
        colors = [(v, k) for k, v in colors.items()]

    # Sort by frequency
    colors.sort(reverse=True, key=lambda x: x[0])
    top = [c[1] for c in colors[:k]]

    def to_hex(rgb):
        return '#%02x%02x%02x' % rgb

    return [to_hex(tuple(color)) for color in top]


aura_palette_map = [
    ("Mystic Violet", "#7c3aed"),
    ("Electric Indigo", "#6366f1"),
    ("Neon Azure", "#0ea5e9"),
    ("Solar Amber", "#f59e0b"),
    ("Cyber Magenta", "#db2777"),
    ("Quantum Jade", "#10b981"),
]


def classify_aura(hex_colors: List[str]) -> (str, float, str):
    # Map to nearest of our curated palette by simple RGB distance
    def hex_to_rgb(h: str):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def dist(a, b):
        return sum((a[i]-b[i])**2 for i in range(3))

    best_label = "Mystic Violet"
    best_dist = float('inf')
    target_rgb = [hex_to_rgb(c) for c in hex_colors if len(c) == 7]
    if not target_rgb:
        target_rgb = [(124, 58, 237)]

    for label, hexv in aura_palette_map:
        p = hex_to_rgb(hexv)
        d = min(dist(p, t) for t in target_rgb)
        if d < best_dist:
            best_dist = d
            best_label = label

    # Score normalized
    score = max(0.1, 1.0 - min(best_dist / (255**2*3), 0.9))

    notes_map = {
        "Mystic Violet": "Intuitive, visionary, deeply creative.",
        "Electric Indigo": "Focused, evolving, embracing change.",
        "Neon Azure": "Calm clarity, empathetic, communicative.",
        "Solar Amber": "Warm leadership, optimism, grounded energy.",
        "Cyber Magenta": "Bold expression, passion, magnetic presence.",
        "Quantum Jade": "Healing, growth, balanced vitality.",
    }
    notes = notes_map.get(best_label, "Harmonic aura detected.")

    return best_label, float(round(score, 3)), notes


@app.post("/api/analyze", response_model=AuraResponse)
async def analyze_aura(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    dominant = extract_dominant_colors(image, k=5)
    label, score, notes = classify_aura(dominant)

    # Persist a minimal record
    try:
        record = AuraPhoto(
            user_id=None,
            image_url="data:upload/inline",  # In this environment, we won't store externally
            dominant_colors=dominant,
            aura_label=label,
            aura_score=score,
            notes=notes,
        )
        record_id = create_document("auraphoto", record)
    except Exception:
        # If DB not available, just return without persistence
        record_id = ""

    return AuraResponse(
        aura_label=label,
        aura_score=score,
        dominant_colors=dominant,
        notes=notes,
        record_id=record_id,
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
