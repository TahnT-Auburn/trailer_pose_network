import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pandas
from sklearn.utils import shuffle
from PIL import Image
import torchvision.transforms as transforms
import matplotlib

import torchvision.transforms.functional as TF
import math
from torch import optim

from models.mango_net import mango_net

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





model_params={
        "shape_in": (3,512,512), 
        "initial_filters": 32,    
        "num_fc1": 300,
        "dropout_rate": 0.25}
model = mango_net(model_params)


state_dict = torch.load("weights/mango_net.pth")
model.load_state_dict(state_dict)
model.eval()
model = model.to(device)


tf=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.ToTensor()
])






def get_pitch(input_img):
    input_img = tf(input_img)
    image_ten = input_img.to(device)
    pred_mu = model(image_ten)
    return pred_mu.item()



# Create a VideoCapture object and read from input file 
cap = cv2.VideoCapture('/home/gavlab/Trailer_Pitch_Estimate/TrainingData/RoadRunner/TrainingData/Raw/INT/INT1/videos/RRMC.mp4') 
cap2 = cv2.VideoCapture('/home/gavlab/Trailer_Pitch_Estimate/TrainingData/RoadRunner/TrainingData/Raw/INT/INT1/videos/LRMC.mp4') 
truth_file = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/RoadRunner/TrainingData/processed/INT/INT1/INT1_training.csv"

# cap = cv2.VideoCapture('/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/source/videos/RRMC.mp4') 
# cap2 = cv2.VideoCapture('/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/source/videos/LRMC.mp4') 
# truth_file = "/home/gavlab/Trailer_Pitch_Estimate/TrainingData/procedure1/training/procedure1_training_data.csv"


i=0


i_array = []
truth_array = []
pred_array =[]
pred_sig_array = []
# Check if camera opened successfully 
if (cap.isOpened()== False): 
    print("Error opening video file") 


truth_df = pd.read_csv(truth_file)
column_names = truth_df.columns
# Read until video is completed 
while(cap.isOpened()): 
      
# Capture frame-by-frame 
    ret, frame = cap.read() 
    ret2, frame2 = cap2.read()
    truth = truth_df[column_names[9]][i]
    i_array.append(i)  
    truth_array.append(truth*180/3.1415926)

    i = i +1
    if ret == True: 
   
        #Concatenate the Two Images
        image = cv2.hconcat([frame, frame2])
        pred_mu = get_pitch(image)

        image = cv2.resize(image, (1000,421))

        pred_array.append(pred_mu*180/3.14159)
        # pred_sig_array.append(pred_sig)
        image = cv2.putText(image, str(round(truth*180/3.14159,5)), (350,200), cv2.FONT_HERSHEY_SIMPLEX ,  2, (255,0,0), 2, cv2.LINE_AA) 
        image = cv2.putText(image, str(round(pred_mu*180/3.14159,5)), (350,300), cv2.FONT_HERSHEY_SIMPLEX ,  2, (0,0,255), 2, cv2.LINE_AA) 
        cv2.imshow("Predictions", image)
        cv2.waitKey(3)
        print("---")
        print(truth*180/3.1415926)
        print(pred_mu*180/3.1415926)
        # print(pred_sig)

    # Press Q on keyboard to exit 
        if cv2.waitKey(25) & 0xFF == ord('q'): 
            break
        
# Break the loop 
    else: 
        break
 
    if i ==len(truth_df)-3:
            break
  
  
# When everything done, release 
# the video capture object 
cap.release() 
  
# Closes all the frames 
plt.figure(200)

plt.plot(i_array, pred_array, 'b', label = "Estimated")
plt.plot(i_array, truth_array, 'r', label="True Pitch")
# plt.fill_between(i_array, np.array(pred_array)-np.array(pred_sig_array), np.array(pred_array)+np.array(pred_sig_array),color='grey',alpha=0.6, label="Uncertainty Bounds")

plt.xlabel("Frame Number")  # add X-axis label
plt.ylabel("Pitch [DEG]")  # add Y-axis label
plt.title("1500 Frames from Procedure 1 Data")  # add title
plt.legend()
plt.savefig("Estimates.png")

error = np.array(truth_array)-np.array(pred_array)
plt.figure(300)
plt.plot(i_array, error, color='orange', label="Error")

plt.xlabel("Frame Number")  # add X-axis label
plt.ylabel("Error [DEG]")  # add Y-axis label
plt.title("Error")  # add title
plt.savefig("Error.png")