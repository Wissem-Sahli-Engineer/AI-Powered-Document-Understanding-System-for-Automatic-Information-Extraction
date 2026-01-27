import os 
from PIL import Image


# input => the folder where .tif files are
# ouput => the folder where .png will be 
# name => new file's name, "file" is set by default ! 

def convertor(input,output,name = "file"):

    input = input.replace("\\", "/")
    output = output.replace("\\", "/")
    

    if not os.path.exists(output):
        os.makedirs(output)

    files = [f for f in os.listdir(input) if f.lower().endswith(('.tif','.tiff'))]
    files.sort()

    f,s=0,0
    for i, filename in enumerate(files,start=1):
        try:
            with Image.open(os.path.join(input,filename)) as img:
                new_name = f"{name}{i}.png"
                output_path = os.path.join(output, new_name)

                img.convert("RGB").save(output_path, "PNG")

                print(f"Converted: {filename} -> {new_name}")
                s=s+1
        except Exception as e:
            print(f"Error converting {filename}: {e}")
            f=f+1

    print(f"Processing complete, {s} converted to .png")
    print("Number of files that couldn't be converted to .png",f)

path_in = r"C:\Users\wgsom\Desktop\AI_Project\data\raw\public"
path_out = r"C:\Users\wgsom\Desktop\AI_Project\data\images\non_invoices\all"

convertor(path_in,path_out,"non_invoices")