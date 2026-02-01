"""
FastAPI Backend for OACA Invoice Extraction
Uses fine-tuned LayoutLMv3 model for NER on invoice images
"""
import os
import json
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import pytesseract
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

from .schemas import InvoiceData, ExtractionResponse, HealthResponse

# Configuration
MODEL_PATH = Path(__file__).parent.parent / "data_oaca" / "layoutlmv3-oaca" / "checkpoint-200"
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Global model references
model = None
processor = None


def load_model():
    """Load the LayoutLMv3 model and processor"""
    global model, processor
    
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model checkpoint not found at {MODEL_PATH}")
    
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_PATH))
    
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Warning: Could not load model: {e}")
    yield


app = FastAPI(
    title="OACA Invoice Extraction API",
    description="Extract structured data from OACA invoices using LayoutLMv3",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_ocr(image: Image.Image) -> dict:
    """Run Tesseract OCR and extract words with bounding boxes"""
    # Configure Tesseract path if on Windows
    if os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
    
    # Get OCR data with bounding boxes
    ocr_data = pytesseract.image_to_data(image, lang='fra', output_type=pytesseract.Output.DICT)
    
    words = []
    boxes = []
    width, height = image.size
    
    for i, word in enumerate(ocr_data['text']):
        if word.strip():  # Skip empty strings
            x = ocr_data['left'][i]
            y = ocr_data['top'][i]
            w = ocr_data['width'][i]
            h = ocr_data['height'][i]
            
            # Normalize to 0-1000 scale for LayoutLMv3
            normalized_box = [
                int(1000 * (x / width)),
                int(1000 * (y / height)),
                int(1000 * ((x + w) / width)),
                int(1000 * ((y + h) / height))
            ]
            # Clamp to [0, 1000]
            normalized_box = [max(0, min(1000, coord)) for coord in normalized_box]
            
            words.append(word)
            boxes.append(normalized_box)
    
    return {"words": words, "boxes": boxes}


def extract_entities(image: Image.Image, words: list, boxes: list) -> dict:
    """Run model inference and extract named entities using word-level mapping"""
    global model, processor
    
    if model is None or processor is None:
        raise RuntimeError("Model not loaded")
    
    device = next(model.parameters()).device
    
    # Process input
    encoding = processor(
        image, 
        words, 
        boxes=boxes, 
        return_tensors="pt", 
        truncation=True, 
        padding="max_length", 
        max_length=512,
        return_offsets_mapping=False
    )
    
    # Get word_ids to map tokens back to original words
    word_ids = encoding.word_ids(batch_index=0)
    
    encoding_tensors = {k: v.to(device) for k, v in encoding.items()}
    
    # Inference
    with torch.no_grad():
        outputs = model(**encoding_tensors)
    
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    id2label = model.config.id2label
    
    # Map token predictions to word-level predictions
    # Use the first subtoken's prediction for each word
    word_predictions = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in word_predictions:
            pred_label = id2label[predictions[token_idx]]
            word_predictions[word_id] = pred_label
    
    # Build entities from original words using BIO tags
    final_entities = {}
    current_entity_type = None
    current_entity_words = []
    
    for word_idx in sorted(word_predictions.keys()):
        label = word_predictions[word_idx]
        
        if label == "O":
            # End current entity if exists
            if current_entity_type and current_entity_words:
                entity_text = " ".join(current_entity_words)
                if current_entity_type not in final_entities:
                    final_entities[current_entity_type] = []
                final_entities[current_entity_type].append(entity_text)
            current_entity_type = None
            current_entity_words = []
            continue
        
        # Parse BIO label
        if "-" in label:
            prefix, entity_type = label.split("-", 1)
        else:
            prefix = "B"
            entity_type = label
        
        if prefix == "B":
            # Begin new entity - save previous if exists
            if current_entity_type and current_entity_words:
                entity_text = " ".join(current_entity_words)
                if current_entity_type not in final_entities:
                    final_entities[current_entity_type] = []
                final_entities[current_entity_type].append(entity_text)
            
            current_entity_type = entity_type
            current_entity_words = [words[word_idx]]
        elif prefix == "I" and current_entity_type == entity_type:
            # Continue current entity
            current_entity_words.append(words[word_idx])
        else:
            # I- tag without matching B- tag, treat as new entity
            if current_entity_type and current_entity_words:
                entity_text = " ".join(current_entity_words)
                if current_entity_type not in final_entities:
                    final_entities[current_entity_type] = []
                final_entities[current_entity_type].append(entity_text)
            
            current_entity_type = entity_type
            current_entity_words = [words[word_idx]]
    
    # Flush last entity
    if current_entity_type and current_entity_words:
        entity_text = " ".join(current_entity_words)
        if current_entity_type not in final_entities:
            final_entities[current_entity_type] = []
        final_entities[current_entity_type].append(entity_text)
    
    return final_entities


def find_total_fallback(words: list) -> str:
    """Heuristic fallback to find total amount if model fails"""
    keywords = ["TOTAL", "FACTURE", "NET", "PAYER", "SOMME"]
    for i, word in enumerate(words):
        if any(kw in word.upper() for kw in keywords):
            # Look at the next few words for something that looks like a number
            for j in range(i + 1, min(i + 6, len(words))):
                candidate = words[j].replace(",", ".")
                # Look for something with digits and potentially a decimal
                if any(c.isdigit() for c in candidate) and ('.' in candidate or len(candidate) > 3):
                    return words[j]
    return None


def map_entities_to_invoice(entities: dict, all_words: list = None) -> InvoiceData:
    """Map extracted entities to InvoiceData schema with fallback support"""
    # Try to get from AI detected entities
    total = " | ".join(entities.get("TOTAL", []))
    
    # If not detected, try fallback
    if (not total or total == "None") and all_words:
        fallback_total = find_total_fallback(all_words)
        if fallback_total:
            total = fallback_total
            
    return InvoiceData(
        invoice_number=" | ".join(entities.get("INVOICE_NO", [])) or None,
        invoice_date=" | ".join(entities.get("DATE", [])) or None,
        client_name=" | ".join(entities.get("CLIENT", [])) or None,
        total_amount=total or None,
        currency=" | ".join(entities.get("CURRENCY", [])) or None,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API and model health"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/extract", response_model=ExtractionResponse)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Extract structured data from an uploaded invoice image.
    
    Accepts: PNG, JPG, JPEG images
    Returns: Extracted invoice fields (invoice_number, date, client, total, currency)
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load and process image
        image = Image.open(tmp_path).convert("RGB")
        
        # Run OCR
        ocr_result = run_ocr(image)
        
        if not ocr_result["words"]:
            return ExtractionResponse(
                success=False,
                message="No text detected in image",
                data=None
            )
        
        # Extract entities
        entities = extract_entities(image, ocr_result["words"], ocr_result["boxes"])
        
        # Map to schema
        invoice_data = map_entities_to_invoice(entities, ocr_result["words"])
        
        # Cleanup
        os.unlink(tmp_path)
        
        return ExtractionResponse(
            success=True,
            message="Extraction successful",
            data=invoice_data,
            raw_entities=entities
        )
    
    except Exception as e:
        return ExtractionResponse(
            success=False,
            message=f"Extraction failed: {str(e)}",
            data=None
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
