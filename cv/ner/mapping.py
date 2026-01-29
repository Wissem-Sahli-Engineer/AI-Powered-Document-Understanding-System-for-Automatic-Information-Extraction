import json
import os
import re


# 1. SETUP PATHS
ANNOTATIONS_DIR = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\annotations\ner"
OCR_DIR = r"C:\Users\wgsom\Desktop\AI_Project\cv\ocr"
OUTPUT_DIR = r"C:\Users\wgsom\Desktop\AI_Project\cv\ner"

# Define your labels
LABEL_MAP = {
    "O": 0,
    "B-INVOICE_NO": 1, "I-INVOICE_NO": 2,
    "B-DATE": 3, "I-DATE": 4,
    "B-CLIENT": 5, "I-CLIENT": 6,
    "B-TOTAL": 7, "I-TOTAL": 8,
    "B-CURRENCY": 9, "I-CURRENCY": 10
}
    
def normalize(text):
    return re.sub(r"\s+", "", str(text).lower().strip())

def create_dataset_split(split_name):
    split_ann_dir = ANNOTATIONS_DIR / split_name
    split_ocr_dir = OCR_DIR / split_name
    output_file = OUTPUT_DIR / f"{split_name}.jsonl"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Process each annotation file
    for ann_file in split_ann_dir.glob("*.json"):
        ocr_file = split_ocr_dir / ann_file.name
        
        if not ocr_file.exists():
            print(f"⚠️ Missing OCR for {ann_file.name}, skipping.")
            continue

        with open(ann_file, 'r', encoding='utf-8') as f:
            manual = json.load(f)
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr = json.load(f)

        # Mapping Dictionary (Manual label -> target text)
        targets = {
            "INVOICE_NO": normalize(manual.get("invoice_number", "")),
            "DATE": normalize(manual.get("invoice_date", "")),
            "CLIENT": normalize(manual.get("client_name", "")),
            "TOTAL": normalize(manual.get("total_amount", "")),
            "CURRENCY": normalize(manual.get("currency", ""))
        }

        tokens = []
        bboxes = []
        ner_tags = []

        # Iterate through OCR words and match
        for word_data in ocr["words"]:
            text = word_data["text"]
            norm_text = normalize(text)
            
            # Default label is 'O'
            assigned_label = "O"
            
            # Check against each target
            for label_name, target_val in targets.items():
                if target_val and norm_text in target_val and len(norm_text) > 1:
                    # Simple logic: First match becomes B-
                    # (In a more complex version, we check sequence for I- tags)
                    assigned_label = f"B-{label_name}"
                    break 

            tokens.append(text)
            bboxes.append([
                word_data["left"], 
                word_data["top"], 
                word_data["left"] + word_data["width"], 
                word_data["top"] + word_data["height"]
            ])
            ner_tags.append(LABEL_MAP.get(assigned_label, 0))

        # Save this invoice as one entry
        results.append({
            "id": ann_file.stem,
            "tokens": tokens,
            "bboxes": bboxes,
            "ner_tags": ner_tags,
            "image_path": str(BASE_DIR / "data_preprocessed" / split_name / (ann_file.stem + ".png"))
        })

    # Write to JSONL (one JSON object per line)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    
    print(f"✅ Created {split_name}.jsonl with {len(results)} samples.")

# RUN FOR ALL SPLITS
for split in ["train", "val", "test"]:
    create_dataset_split(split)