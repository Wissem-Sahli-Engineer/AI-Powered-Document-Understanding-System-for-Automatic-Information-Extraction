import cv2
import json
import os

def visualize_ocr(image_path, json_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Load the big JSON file you just created
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Draw a box for every word
    for word in data['words']:
        x = word['left']
        y = word['top']
        w = word['width']
        h = word['height']
        
        # Draw a green rectangle (thickness 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the result to see it
    cv2.imwrite(output_path, img)
    print(f"✅ Visualization saved to: {output_path}")

# --- TEST IT ON ONE FILE ---
img_test = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\data_preprocessed\train\invoice_1.png"
json_test = r"C:\Users\wgsom\Desktop\AI_Project\cv\ocr\train\invoice_1.json"
out_test = r"C:\Users\wgsom\Desktop\AI_Project\ocr_check.png"

visualize_ocr(img_test, json_test, out_test)