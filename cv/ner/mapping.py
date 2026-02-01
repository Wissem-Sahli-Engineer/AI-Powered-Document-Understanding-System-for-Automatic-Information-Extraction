from pathlib import Path
import json
import re

# 1. SETUP YOUR PATHS
# These now point to the "root" of your splits
BASE_ANN = Path(r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\annotations")
BASE_OCR = Path(r"C:\Users\wgsom\Desktop\AI_Project\cv\ocr")
OUTPUT_DIR = Path(r"C:\Users\wgsom\Desktop\AI_Project\cv\ner")

LABEL_MAP = {
    "O": 0, "B-INVOICE_NO": 1, "I-INVOICE_NO": 2, "B-DATE": 3, "I-DATE": 4,
    "B-CLIENT": 5, "I-CLIENT": 6, "B-TOTAL": 7, "I-TOTAL": 8, "B-CURRENCY": 9, "I-CURRENCY": 10
}

def normalize(text):
    return re.sub(r"\s+", "", str(text).lower().strip()).replace(",", ".")

def process_split(split_name):
    ann_folder = BASE_ANN / split_name
    ocr_folder = BASE_OCR / split_name
    output_file = OUTPUT_DIR / f"{split_name}.jsonl"
    
    if not ann_folder.exists():
        print(f"⚠️ Skipping {split_name}: Folder not found.")
        return

    results = []
    ann_files = list(ann_folder.glob("*.json"))
    
    for ann_file in ann_files:
        # Looking for the OCR version of the SAME invoice
        ocr_file = ocr_folder / ann_file.name
        if not ocr_file.exists(): 
            print(f"⚠️ Missing OCR for {ann_file.name} in {split_name}")
            continue

        with open(ann_file, 'r', encoding='utf-8') as f: manual = json.load(f)
        with open(ocr_file, 'r', encoding='utf-8') as f: ocr = json.load(f)

        tokens = [w["text"] for w in ocr["words"]]
        bboxes = [[w["left"], w["top"], w["left"]+w["width"], w["top"]+w["height"]] for w in ocr["words"]]
        ner_tags = [0] * len(tokens)

        # Mapping targets from your annotation file
        targets = {
            "INVOICE_NO": manual.get("invoice_number"),
            "DATE": manual.get("invoice_date"),
            "CLIENT": manual.get("client_name"),
            "TOTAL": manual.get("total_amount"),
            "CURRENCY": manual.get("currency")
        }

        found_labels = set()
        for label_name, target_value in targets.items():
            if not target_value: continue
            target_words = str(target_value).split()
            norm_target_words = [normalize(w) for w in target_words]
            n = len(norm_target_words)

            for i in range(len(tokens) - n + 1):
                window = [normalize(tokens[j]) for j in range(i, i + n)]
                if window == norm_target_words and label_name not in found_labels:
                    # BIO Tagging: B- for first, I- for the rest
                    ner_tags[i] = LABEL_MAP[f"B-{label_name}"]
                    for j in range(1, n): ner_tags[i+j] = LABEL_MAP[f"I-{label_name}"]
                    found_labels.add(label_name)
                    break 

        results.append({"id": ann_file.stem, "tokens": tokens, "bboxes": bboxes, "ner_tags": ner_tags})

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results: f.write(json.dumps(item) + "\n")
    print(f"✅ Created {output_file.name} with {len(results)} invoices.")

# Run for your splits
for split in ["train", "val", "test"]:
    process_split(split)