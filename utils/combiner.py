'''
############### Data Combiner for Trailer Network Data ################

Utility script to combine data from different directories to a singular
.csv file

#######################################################################
'''

#%%
import os
import pandas as pd
import csv 

parent_directory_path = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\processed'
sub_directories_path = os.listdir(parent_directory_path)

output_directory_path = 'C:\\Users\\pzt0029\\Documents\\Networks\\trailer_pose_network\\trailer_pose_network\\data\\combined\\combined_data.csv'

counter = 0

with open(output_directory_path, 'w') as file:
    wr = csv.writer(file, quoting = csv.QUOTE_ALL)
    for folder in sub_directories_path:
        subfolders = os.listdir(parent_directory_path+"\\"+folder)
        for subfolder in subfolders:
            path = parent_directory_path+"\\"+folder+"\\"+subfolder+"\\"
            print(path)
            csv_file = path+subfolder+"_training.csv"

            with open(csv_file, "r") as f:
                reader = csv.reader(f)

                for i, line in enumerate(reader):
                    left_image =  parent_directory_path+"\\"+folder+"\\"+subfolder+"\\images\\LRMC\\"+line[0]+".jpg"
                    right_image = parent_directory_path+"\\"+folder+"\\"+subfolder+"\\images\\RRMC\\"+line[1]+".jpg"
                    line[0] = left_image
                    line[1] = right_image
                    
                    wr.writerow(line)
                    

file.close()