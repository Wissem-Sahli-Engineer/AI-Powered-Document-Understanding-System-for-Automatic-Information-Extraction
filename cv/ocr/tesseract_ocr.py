import pytesseract
import cv2
import json
import os
from natsort import natsorted

# 1. Path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def run_ocr_for_ner(image_path, output_json_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Could not read image: {image_path}")
        return

    config = "--oem 3 --psm 6"
    data = pytesseract.image_to_data(img, lang="fra+eng", config=config, output_type=pytesseract.Output.DICT)

    ocr_results = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            word_data = {
                "text": data['text'][i],
                "left": data['left'][i],
                "top": data['top'][i],
                "width": data['width'][i],
                "height": data['height'][i],
                "conf": data['conf'][i]
            }
            ocr_results.append(word_data)

    final_output = {
        "file_name": os.path.basename(image_path),
        "dimensions": {"width": img.shape[1], "height": img.shape[0]},
        "words": ocr_results
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

# 2. Main Processing Loop
input_root = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\data_preprocessed"
output_root = r"C:\Users\wgsom\Desktop\AI_Project\cv\ocr"
splits = ['train', 'val', 'test']

for split in splits:
    image_dir = os.path.join(input_root, split)
    json_dir = os.path.join(output_root, split)

    # Create the output folder if it doesn't exist
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    if os.path.exists(image_dir):
        # Sort files naturally (1, 2, 10 instead of 1, 10, 2)
        images = natsorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        
        print(f"--- Processing {split} split ({len(images)} files) ---")
        for img_name in images:
            img_path = os.path.join(image_dir, img_name)
            # Change extension from .png to .json
            json_name = os.path.splitext(img_name)[0] + ".json"
            json_path = os.path.join(json_dir, json_name)
            
            run_ocr_for_ner(img_path, json_path)
            print(f"✅ OCR Done: {img_name}")

print("\n🚀 All folders processed successfully!")