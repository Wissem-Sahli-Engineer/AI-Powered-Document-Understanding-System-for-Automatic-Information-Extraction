data collection Build a realistic, diverse dataset ,
place them raw folder : real_invoices and public contains other documents 

nb : raw_data : read-only

Normalize Image Format : Convert ALL .tif → .png 
 using python (Pillow) 
 ( creation of a function :
 convertor( inputfolderpath , outputfolderpath, desired name for the file-"file" is default)
 )
Split Documents by ROLE

split the all folder to test-val-train using python ( splitfolder) ratio 0.7 - 0.15 -0.15