import os
import json
from natsort import natsorted 

# Define paths
base_data_path = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca"
output_base = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\annotations\ner"

# Define your specific split requirements
splits = {
    "train": 35,
    "val": 7,
    "test": 8
}

# The JSON structure you want
template = {
    "invoice_number": "",
    "invoice_date": "20/07/2023",
    "client_name": "",
    "total_amount": "",
    "currency": "TND"
}

total_created = 0

for split_name, count in splits.items():
    image_folder = os.path.join(base_data_path, split_name)
    
    # Check if the image folder exists
    if not os.path.exists(image_folder):
        print(f"⚠️ Warning: Folder {image_folder} not found. Skipping.")
        continue

    # Get all png files in that folder
    images = [f for f in os.listdir(image_folder) if f.endswith(".png")]
    
    # --- FIXED SORTING HERE ---
    images = natsorted(images) 
    # Now it will be: [invoice_1, invoice_2, ... invoice_10, invoice_100]

    # Take only the number you requested (35, 7, or 8)
    selected_images = images[:count]

    print(selected_images,"\n")

    for img_name in selected_images:
        # Create the JSON filename (invoice_1.png -> invoice_1.json)
        json_name = os.path.splitext(img_name)[0] + ".json"
        save_path = os.path.join(output_base, json_name)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Write the empty JSON file
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=4)
        
        total_created += 1

print(f"✅ Success! Created {total_created} JSON templates in {output_base}")


