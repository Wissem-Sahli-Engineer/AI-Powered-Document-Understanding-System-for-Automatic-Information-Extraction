import cv2
import os

def safe_preprocess(input_base, output_base):
    splits = ['train', 'val', 'test']
    
    for split in splits:
        in_path = os.path.join(input_base, split)
        out_path = os.path.join(output_base, split)
        
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        images = [f for f in os.listdir(in_path) if f.lower().endswith('.png')]
        
        for img_name in images:
            img = cv2.imread(os.path.join(in_path, img_name))
            if img is None: continue

            # 1. Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 2. Gentle Denoising (3x3 is very safe)
            # This removes "salt and pepper" noise without blurring text
            clean = cv2.GaussianBlur(gray, (3, 3), 0)

            # 3. Save (Maintain original 2480 x 3509 resolution)
            cv2.imwrite(os.path.join(out_path, img_name), clean)
            print(f"Processed: {split}/{img_name}")

# --- RUN IT ---
# Use different folders so you don't overwrite your original high-res color images!
path_raw_split = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca"
path_preprocessed = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\data_preprocessed"

safe_preprocess(path_raw_split, path_preprocessed)