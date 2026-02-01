import json
from PIL import Image
import torch
from transformers import AutoProcessor, LayoutLMv3ForTokenClassification

# 1. Load Model
model_path = r"C:\Users\wgsom\Desktop\AI_Project\layoutlmv3-oaca\checkpoint-200" 
processor = AutoProcessor.from_pretrained(model_path)
model = LayoutLMv3ForTokenClassification.from_pretrained(model_path).to("cuda")

# 2. Load Image and OCR JSON
image_path = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\val\invoice_151.png"
json_path = r"C:\Users\wgsom\Desktop\AI_Project\cv\ocr\val\invoice_151.json"

image = Image.open(image_path).convert("RGB")
width, height = image.size

with open(json_path, 'r') as f:
    ocr_data = json.load(f)

# 3. Extract and Normalize Boxes
# Assuming your JSON has a list of objects with "word" and "bbox" [x1, y1, x2, y2]
width = ocr_data['dimensions']['width']
height = ocr_data['dimensions']['height']

words = []
boxes = []

for item in ocr_data['words']:
    words.append(item['text'])
    
    # Your JSON provides: left, top, width, height
    # LayoutLMv3 needs: [x1, y1, x2, y2]
    x1 = item['left']
    y1 = item['top']
    x2 = x1 + item['width']
    y2 = y1 + item['height']
    
    # Normalize pixel coordinates to 0-1000 scale
    normalized_box = [
        int(1000 * (x1 / width)),
        int(1000 * (y1 / height)),
        int(1000 * (x2 / width)),
        int(1000 * (y2 / height))
    ]
    # Clamp to [0, 1000]
    normalized_box = [max(0, min(1000, x)) for x in normalized_box]
    boxes.append(normalized_box)

# 4. Feed to Processor (No need for Tesseract now!)
print(f"Number of words: {len(words)}")
encoding = processor(image, words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
if torch.cuda.is_available():
    encoding = encoding.to("cuda")

# 5. Inference
print("Starting inference...")
with torch.no_grad():
    outputs = model(**encoding)
print("Inference done.")

predictions = outputs.logits.argmax(-1).squeeze().tolist()
id2label = model.config.id2label

# 6. Parse and Print Results
final_entities = {}
current_entity_label = None
current_entity_tokens = []

for i, pred_id in enumerate(predictions):
    label = id2label[pred_id]
    
    # 1. Skip non-entity tokens ('O')
    if label == "O":
        if current_entity_label:
            # Save current
            text = processor.tokenizer.decode(current_entity_tokens).strip()
            # Filter special tokens just in case
            if text not in ["<s>", "</s>"]:
                if current_entity_label not in final_entities:
                    final_entities[current_entity_label] = []
                final_entities[current_entity_label].append(text)
            
            # Reset
            current_entity_label = None
            current_entity_tokens = []
        continue

    # 2. Extract type
    if "-" in label:
        prefix, entity_type = label.split("-", 1)
    else:
        prefix = "B"
        entity_type = label
    
    # 3. Aggregation Logic
    # If same type, continue (ignore B/I distinction to merge fragmented tokens)
    if current_entity_label == entity_type:
         current_entity_tokens.append(encoding["input_ids"][0][i])
    
    # New type
    else:
        # Save previous if exists
        if current_entity_label:
            text = processor.tokenizer.decode(current_entity_tokens).strip()
            if text not in ["<s>", "</s>"]:
                if current_entity_label not in final_entities:
                    final_entities[current_entity_label] = []
                final_entities[current_entity_label].append(text)

        # Start new
        current_entity_label = entity_type
        current_entity_tokens = [encoding["input_ids"][0][i]]

# Flush last
if current_entity_label:
    text = processor.tokenizer.decode(current_entity_tokens).strip()
    if text not in ["<s>", "</s>"]:
        if current_entity_label not in final_entities:
            final_entities[current_entity_label] = []
        final_entities[current_entity_label].append(text)

# Print nice summary
print("\n" + "="*30)
print("EXTRACTED DATA")
print("="*30)
for label, values in final_entities.items():
    # Join distinct occurrences with " | "
    # Note: Because we merged consecutive B-tags, "distinct" now means separated by 'O' or other labels.
    full_text = " | ".join(values)
    print(f"{label}: {full_text}")

