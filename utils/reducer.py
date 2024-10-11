import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv



input_path = '/home/tahn/Software/Networks/trailer_pose_network/trailer_pose_network/data/combined/combined_data_augmented.csv'
output_path = '/home/tahn/Software/Networks/trailer_pose_network/trailer_pose_network/data/reduced/reduced_data.csv'
counter = 0

df = pd.read_csv(input_path, index_col=0)

print(df.head())
column_names = df.columns

n = df[column_names[15]].value_counts().min()

df = df.sample(frac=1)

df = df.groupby(column_names[15]).head(300)
print(column_names[15])

print(len(df.index))

df.to_csv(output_path)


# with open(output_path, 'w') as file:
#     wr = csv.writer(file, quoting = csv.QUOTE_ALL)
#     for folder in sub_directories_path:
#         subfolders = os.listdir(parent_directory_path+"/"+folder)
#         for subfolder in subfolders:
#             path = parent_directory_path+"/"+folder+"/"+subfolder+"/"
#             print(path)
#             csv_file = path+subfolder+"_training.csv"

#             with open(csv_file, "r") as f:
#                 reader = csv.reader(f)

#                 for i, line in enumerate(reader):
#                     left_image =  parent_directory_path+"/"+folder+"/"+subfolder+"/images/LRMC/"+line[0]+".jpg"
#                     right_image = parent_directory_path+"/"+folder+"/"+subfolder+"/images/RRMC/"+line[1]+".jpg"
#                     line[0] = left_image
#                     line[1] = right_image
                    
#                     wr.writerow(line)
                    

# file.close()
               
#         # self.df = pd.read_csv("/"+path+self.csv_file)
#         # self.df = shuffle(self.df)
#         # self.left_image_path ="/"+path+"/images/LRMC/"
#         # self.right_image_path ="/"+path+"/images/RRMC/"

        # self.column_names = self.df.columns