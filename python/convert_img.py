import os 
from PIL import Image
from pdf2image import convert_from_path, pdfinfo_from_path

# input => the folder where .tif files are
# ouput => the folder where .png will be 
# name => new file's name, "file" is set by default ! 

def convertor_tif_png(input,output,name = "file"):

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


def convertor_pdf_png(input,output,name = "file"):

    input = input.replace("\\", "/")
    output = output.replace("\\", "/")
    
    poppler = r"C:\Program Files\poppler-25.12.0\Library\bin"

    if not os.path.exists(output):
        os.makedirs(output)

    if not os.path.exists(input):
        print(f"❌ ERROR: The PDF file was not found at that path!")
        return
    
    try:
        info = pdfinfo_from_path(input, poppler_path=poppler)
        total_pages = info["Pages"]
        print(f"--- Process Started ---")
        print(f"Total pages to convert: {total_pages}")

        # Convert pages one by one in a loop
        for i in range(1, total_pages + 1):
            page = convert_from_path(
                input, 
                dpi=300, 
                first_page=i, 
                last_page=i, 
                poppler_path=poppler
            )
            
            img_name = f"{name}_{i}.png"
            save_path = os.path.join(output, img_name)
            page[0].save(save_path, 'PNG')
            
            print(f"✅ [{i}/{total_pages}] Saved: {img_name}")

        print(f"--- Finished! ---")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
    

path_in =r"C:\Users\wgsom\Desktop\AI_Project\OACA.pdf"
path_out = r"C:\Users\wgsom\Desktop\AI_Project\data_oaca\invoices"

convertor_pdf_png(path_in,path_out,"invoice")