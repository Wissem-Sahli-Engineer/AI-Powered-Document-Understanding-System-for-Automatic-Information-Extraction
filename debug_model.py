import torch
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
from pathlib import Path
import os

# Setup
MODEL_PATH = Path("data_oaca/layoutlmv3-oaca/checkpoint-200")
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

def debug_inference(image_path):
    print(f"Processing {image_path}...")
    
    # Load Model
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model = LayoutLMv3ForTokenClassification.from_pretrained(str(MODEL_PATH))
    
    # OCR
    image = Image.open(image_path).convert("RGB")
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    words = []
    boxes = []
    width, height = image.size
    
    all_ocr_text = []
    for i, word in enumerate(ocr_data['text']):
        if word.strip():
            all_ocr_text.append(word)
            x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
            # Normalize 0-1000
            normalized_box = [
                int(1000 * (x / width)),
                int(1000 * (y / height)),
                int(1000 * ((x + w) / width)),
                int(1000 * ((y + h) / height))
            ]
            normalized_box = [max(0, min(1000, c)) for c in normalized_box]
            words.append(word)
            boxes.append(normalized_box)
            
    print("\n--- OCR WORDS ---")
    print(" ".join(all_ocr_text))
            
    # Inference
    encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**encoding)
        
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    id2label = model.config.id2label
    word_ids = encoding.word_ids(batch_index=0)
    
    # DEBUG: Capture all non-O predictions
    results = {
        "ocr_text": " ".join(all_ocr_text),
        "predictions": []
    }
    seen_words = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is not None and word_id not in seen_words:
            pred = id2label[predictions[idx]]
            if pred != "O":
                results["predictions"].append({"word": words[word_id], "label": pred})
            seen_words.add(word_id)
            
    import json
    with open("debug_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to debug_results.json")
            
if __name__ == "__main__":
    debug_inference("data_oaca/invoices/invoice_130.png")
