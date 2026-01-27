data collection Build a realistic, diverse dataset ,
place them raw folder : real_invoices and public contains other documents 

nb : raw_data : read-only

Normalize Image Format : Convert ALL .tif → .png 
 using python (Pillow) 
 ( creation of a function :
 convertor_tif_png( inputfolderpath , outputfolderpath, desired name for the file-"file" is default)
 )
Split Documents by ROLE

split the all folder to test-val-train using python ( splitfolder) ratio 0.7 - 0.15 -0.15

annotation : 125 invoice , Required fields.txt
( Invoice number
Date
Total amount
Currency
Supplier name (if visible) ) 

each invoice gets a JSON exemple : 
{
  "invoice_number": "INV-2024-019",
  "invoice_date": "2024-12-10",
  "total_amount": "1240.50",
  "currency": "TND",
}
store in data/annotations/ner/
nb : Filename must match image name.

Got 150 invoices from OACA they all clean adn structuted i ll start using them 

Problem 150 in 1 pdf : each page is an invoice ! 

So pdf => .png files 

using python (pdf2image ) agian create a fuction named convertor_pdf_png, downloading poppler (So python could count the pages) and copy the path of its bin ( insdie the liraby folder ) after unexrating from .zip ( downlink : https://github.com/oschwartz10612/poppler-windows/releases/ ) 