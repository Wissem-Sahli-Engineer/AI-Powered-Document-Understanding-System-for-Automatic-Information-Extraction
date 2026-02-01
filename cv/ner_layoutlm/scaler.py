import json
import os
from pathlib import Path
from tqdm import tqdm  # strict optional, but helpful if installed. I'll stick to standard lib to be safe or check imports.

# Define Paths
# Adjust these absolute paths if your workspace structure changes
BASE_NER_DIR = Path(r"C:\Users\wgsom\Desktop\AI_Project\cv\ner")
BASE_OCR_DIR = Path(r"C:\Users\wgsom\Desktop\AI_Project\cv\ocr")
OUTPUT_DIR = Path(r"C:\Users\wgsom\Desktop\AI_Project\cv\ner_layoutlm")

def scale_bbox(box, width, height):
    """
    Scales a bounding box [x1, y1, x2, y2] from absolute pixels to 0-1000 scale.
    Clamps values between 0 and 1000.
    """
    x1, y1, x2, y2 = box
    
    # Scale coordinates
    # Use max(1, width) to avoid division by zero
    w_scale = 1000 / max(1, width)
    h_scale = 1000 / max(1, height)
    
    x1_s = int(x1 * w_scale)
    y1_s = int(y1 * h_scale)
    x2_s = int(x2 * w_scale)
    y2_s = int(y2 * h_scale)
    
    # Clamp to [0, 1000]
    x1_s = max(0, min(1000, x1_s))
    y1_s = max(0, min(1000, y1_s))
    x2_s = max(0, min(1000, x2_s))
    y2_s = max(0, min(1000, y2_s))
    
    # Ensure logical consistency for LayoutLM (x2 >= x1, y2 >= y1)
    # Some implementations prefer strictly x2 >= x1, others don't mind.
    # We will just clamp.
    
    return [x1_s, y1_s, x2_s, y2_s]

def process_split(split_name):
    """
    Process a single split (train, val, or test).
    Reads {split_name}.jsonl from BASE_NER_DIR
    Lookups dimensions in BASE_OCR_DIR/{split_name}/{id}.json
    Writes scaled output to OUTPUT_DIR/{split_name}.jsonl
    """
    input_file = BASE_NER_DIR / f"{split_name}.jsonl"
    output_file = OUTPUT_DIR / f"{split_name}.jsonl"
    ocr_split_dir = BASE_OCR_DIR / split_name
    
    if not input_file.exists():
        print(f"⚠️ Input file not found: {input_file}")
        return

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {split_name}...")
    processed_count = 0
    missing_ocr_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            data = json.loads(line)
            file_id = data.get("id") # e.g., "invoice_1"
            
            # Construct path to original OCR file to get dimensions
            # Assuming OCR file is named "{file_id}.json"
            ocr_path = ocr_split_dir / f"{file_id}.json"
            
            if not ocr_path.exists():
                print(f"  ❌ OCR file missing for ID: {file_id} at {ocr_path}")
                missing_ocr_count += 1
                continue
            
            # Read image dimensions
            try:
                with open(ocr_path, 'r', encoding='utf-8') as f_ocr:
                    ocr_data = json.load(f_ocr)
                    # Check for dimensions key
                    dims = ocr_data.get("dimensions", {})
                    img_width = dims.get("width")
                    img_height = dims.get("height")
                    
                    if not img_width or not img_height:
                         print(f"  ❌ Missing dimensions in OCR file for ID: {file_id}")
                         missing_ocr_count += 1
                         continue

            except Exception as e:
                print(f"  ❌ Error reading OCR file {ocr_path}: {e}")
                missing_ocr_count += 1
                continue
            
            # Scale BBoxes
            original_bboxes = data.get("bboxes", [])
            scaled_bboxes = []
            
            for bbox in original_bboxes:
                scaled_box = scale_bbox(bbox, img_width, img_height)
                scaled_bboxes.append(scaled_box)
            
            # Update data object
            data["bboxes"] = scaled_bboxes
            # Optionally add original image size if needed for debugging
            # data["original_image_size"] = [img_width, img_height]
            
            image_path = Path(r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\data_preprocessed") / split_name / f"{file_id}.png"
            data["image_path"] = str(image_path)

            # Write to output
            fout.write(json.dumps(data) + "\n")
            processed_count += 1

    print(f"✅ Finished {split_name}. Processed: {processed_count}, Missing OCR: {missing_ocr_count}")

def main():
    splits = ["train", "val", "test"]
    
    print("🚀 Starting Coordinate Scaling...")
    for split in splits:
        process_split(split)
    print(f"🎉 All Done! Scaled files are in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
