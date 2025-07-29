#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#%%
# load data
TEST_CSV = "D:\\TestingData\\simulation\\processed\\FF\\FF1\\FF1.csv"
df = pd.read_csv(TEST_CSV)

#%%
# define conversion functions
def tangent_to_body_frame_translation(pose1, pose2):
    """
    Convert from tangent plane poses to body frame relative translation
    
    Args:
        pose1: (X1, Y1, yaw1) - starting pose
        pose2: (X2, Y2, yaw2) - ending pose
        
    Returns:
        (dx_body, dy_body) - translation in body frame of pose1
    """
    X1, Y1, yaw1 = pose1
    X2, Y2, yaw2 = pose2
    
    # World frame displacement
    dx_world = X2 - X1
    dy_world = Y2 - Y1
    
    # Rotate into body frame of pose1
    cos_yaw = np.cos(-yaw1)
    sin_yaw = np.sin(-yaw1)
    
    dx_body = cos_yaw * dx_world - sin_yaw * dy_world
    dy_body = sin_yaw * dx_world + cos_yaw * dy_world
    
    # Relative yaw change
    dyaw = yaw2 - yaw1
    
    # Normalize yaw to [-pi, pi]
    dyaw = np.arctan2(np.sin(dyaw), np.cos(dyaw))
    
    return dx_body, dy_body, dyaw
    
def body_to_tangent_frame_translation(pose1, dx_body, dy_body):
    """
    Convert from body frame translation to tangent plane displacement
    
    Args:
        pose1: (X1, Y1, yaw1) - starting pose
        dx_body, dy_body: translation in body frame
        
    Returns:
        (dx_world, dy_world) - translation in world/tangent frame
    """
    X1, Y1, yaw1 = pose1
    
    # Rotate from body frame to world frame
    cos_yaw = np.cos(yaw1)  # Note: positive yaw1
    sin_yaw = np.sin(yaw1)  # Note: positive yaw1
    
    dx_world = cos_yaw * dx_body - sin_yaw * dy_body
    dy_world = sin_yaw * dx_body + cos_yaw * dy_body
    
    X2 = X1 + dx_world
    Y2 = Y1 + dy_world
    
    return X2, Y2

def compute_abs_pos_error(coords1, coords2):
    X1,Y1 = coords1
    X2,Y2 = coords2
    
    error = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)
    return error

if __name__ == "__main__":
    X_est_array = []
    Y_est_array = []
    for i in range(1,len(df)):
        pose_prev = (df.iloc[i-1]["X"], df.iloc[i-1]["Y"], df.iloc[i-1]["yaw"])
        pose_current = (df.iloc[i]["X"], df.iloc[i]["Y"], df.iloc[i]["yaw"])
        
        dx_body, dy_body, dyaw = tangent_to_body_frame_translation(pose_prev, pose_current)
        
        X_est, Y_est = body_to_tangent_frame_translation(pose_prev, dx_body, dy_body)
        X_est_array.append(X_est)
        Y_est_array.append(Y_est)
    
    X_est_array.insert(0,df.iloc[0]["X"])
    Y_est_array.insert(0,df.iloc[0]["Y"])
    
    # visualize
    plt.plot(X_est_array, Y_est_array)
    plt.plot(df["X"], df["Y"], '--')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(["Est", "Truth"])
    plt.show()
    
    error = compute_abs_pos_error((df["X"],df["Y"]), (X_est_array, Y_est_array))
    plt.plot(error)
    plt.ylabel("Position Error")
    plt.show()