from shutil import move
from splitfolders import ratio
import splitfolders
import splitfolders

input_inv = "C:/Users/wgsom/Desktop/AI_Project/data/images/invoices"
output_inv = "C:/Users/wgsom/Desktop/AI_Project/data/images/invoices"

splitfolders.ratio(input_inv, seed = 42,
                   output = output_inv,
                   ratio= (0.7,0.15,0.15 ),
                   group_prefix=None, move = False)

input_non_inv = "C:/Users/wgsom/Desktop/AI_Project/data/images/non_invoices"
output_non_inv = "C:/Users/wgsom/Desktop/AI_Project/data/images/non_invoices"

splitfolders.ratio(input_non_inv, seed = 42,
                   output = output_non_inv,
                   ratio= (0.7,0.15,0.15 ),
                   group_prefix=None, move = False)

