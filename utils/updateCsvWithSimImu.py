'''
Utility script to update simulation CSVs with custom simulated IMU signals.
'''
#%%
import glob
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import scipy.io
from pathlib import Path
import matplotlib.pyplot as plt

IN_ROOT_DIR = "D:\TestingData\simulation\processed"
TS_ROOT_DIR = "D:\TestingData\simulation\Raw"
OUT_ROOT_DIR = "D:\TestingData\simulation\processed_alt_imu_1"
WRITE_CSV = True
VISUALIZE = False

def simulate_imu_advanced(
    accel: list,
    gyro: list,
    accel_bias_sigma: tuple[float, float, float],
    accel_bias_tau: tuple[float, float, float],
    accel_rw_sigma: tuple[float, float, float],
    gyro_bias_sigma: tuple[float, float, float],
    gyro_bias_tau: tuple[float, float, float],
    gyro_rw_sigma: tuple[float, float, float],
    dt: float,
    L: int
):
    """Simulates IMU from clean accel and gyro signals with 

    Args:
        accel : ndarray, shape (N, 3)
            Clean acceleration signals [ax, ay, az] in m/s^2
        gyro : ndarray, shape (N, 3)
            Clean gyroscope signals [wx, wy, wz] in rad/s
        accel_bias_sigma : tuple of 3 floats
            Standard deviation of FOGM bias for each accel axis (m/s^2)
        accel_bias_tau : tuple of 3 floats
            Time constant of FOGM bias for each accel axis (seconds)
        accel_rw_sigma : tuple of 3 floats
            Standard deviation of random walk for each accel axis (m/s^2)
        gyro_bias_sigma : tuple of 3 floats
            Standard deviation of FOGM bias for each gyro axis (rad/s)
        gyro_bias_tau : tuple of 3 floats
            Time constant of FOGM bias for each gyro axis (seconds)
        gyro_rw_sigma : tuple of 3 floats
            Standard deviation of random walk for each gyro axis (rad/s)
        dt : float
            Time step in seconds
    """
    
    # Convert tuples to arrays for easier manipulation
    accel = np.array(accel)
    gyro = np.array(gyro)
    accel_bias_sigma = np.array(accel_bias_sigma)
    accel_bias_tau = np.array(accel_bias_tau)
    accel_rw_sigma = np.array(accel_rw_sigma)
    gyro_bias_sigma = np.array(gyro_bias_sigma)
    gyro_bias_tau = np.array(gyro_bias_tau)
    gyro_rw_sigma = np.array(gyro_rw_sigma)
    
    # Initialize output arrays
    accel_noisy = np.zeros_like(accel)
    gyro_noisy = np.zeros_like(gyro)
    
    # Initialize FOGM bias states (starting from steady-state distribution)
    accel_bias = np.random.randn(3) * accel_bias_sigma
    gyro_bias = np.random.randn(3) * gyro_bias_sigma
    
    # FOGM parameters
    accel_phi = np.exp(-dt / accel_bias_tau)  # State transition
    gyro_phi = np.exp(-dt / gyro_bias_tau)
    
    # Process noise for FOGM (to maintain steady-state variance)
    accel_bias_noise_std = accel_bias_sigma * np.sqrt(1 - accel_phi**2)
    gyro_bias_noise_std = gyro_bias_sigma * np.sqrt(1 - gyro_phi**2)
    
    # Simulate IMU measurements
    for i in range(L):
        # Update FOGM bias (First-Order Gauss-Markov process)
        accel_bias = accel_phi * accel_bias + np.random.randn(3) * accel_bias_noise_std
        gyro_bias = gyro_phi * gyro_bias + np.random.randn(3) * gyro_bias_noise_std
        
        # Generate random walk (white noise)
        accel_noise = np.random.randn(3) * accel_rw_sigma
        gyro_noise = np.random.randn(3) * gyro_rw_sigma
        
        # Apply errors to clean signals
        accel_noisy[:,i] = accel[:,i] + accel_bias + accel_noise
        gyro_noisy[:,i] = gyro[:,i] + gyro_bias + gyro_noise

        # add gravity term to Az signal
        accel_noisy[2,i] -= 9.81 # positive for NED convention 
        
    return accel_noisy, gyro_noisy # [3,L]

if __name__ == '__main__':
    # define directories
    in_root_dir = IN_ROOT_DIR
    ts_root_dir = TS_ROOT_DIR
    out_root_dir = OUT_ROOT_DIR

    search_pattern = os.path.join(in_root_dir, "**/*.csv")
    csv_files = glob.glob(search_pattern, recursive=True)

    for csv_file in csv_files:
        
        # dataframe
        df = pd.read_csv(csv_file, dtype={"SUBSET": str}, header='infer')
        L = len(df)
        
        # search for corresponding trucksim file for clean accels and ang vels
        set = df['SET'].iloc[0]
        subset = df['SUBSET'].iloc[0]
        ts_dir = ts_root_dir + '\\' + set + '\\' + subset + '\\' + subset + '_TS.mat'
        ts_mat = scipy.io.loadmat(ts_dir)
        # generate true linear accelerations and angular rates
        lin_accel = [
            ts_mat['Ax'].squeeze()*9.81,
            ts_mat['Ay'].squeeze()*9.81,
            ts_mat['Az'].squeeze()*9.81
        ]
        ang_vel = [
            np.deg2rad(ts_mat['AVx'].squeeze()),
            np.deg2rad(ts_mat['AVy'].squeeze()),
            np.deg2rad(ts_mat['AVz'].squeeze()),
        ]
        dt = round(np.mean(np.diff(ts_mat['T_Event'].squeeze())),3)
        stop=1
        accel, gyro = simulate_imu_advanced( # low grade
            lin_accel,
            ang_vel,
            accel_bias_sigma=(0.05, 0.05, 0.05),
            accel_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
            accel_rw_sigma = (0.10, 0.10, 0.10),  # m/s^2 (white noise)
            gyro_bias_sigma = (0.005, 0.005, 0.005),  # rad/s (about 0.1 deg/s or 360 deg/hr)
            gyro_bias_tau = (300.0, 300.0, 300.0),  # seconds (5 minutes)
            gyro_rw_sigma = (0.0005, 0.0005, 0.0005),  # rad/s (about 0.02 deg/s white noise)
            dt=dt,
            L=L,
        )
        # accel, gyro = simulate_imu_advanced( # VN-100
        #     lin_accel,
        #     ang_vel,
        #     accel_bias_sigma=(0.0004, 0.0004, 0.0004),
        #     accel_bias_tau = (1000.0, 1000.0, 1000.0),  # seconds (5 minutes)
        #     accel_rw_sigma = (0.00137, 0.00137, 0.00137),  # m/s^2 (white noise)
        #     gyro_bias_sigma = (0.00145, 0.00145, 0.00145),  # rad/s (about 0.1 deg/s or 360 deg/hr)
        #     gyro_bias_tau = (1000.0, 1000.0, 1000.0),  # seconds (5 minutes)
        #     gyro_rw_sigma = (0.000061, 0.000061, 0.000061),  # rad/s (about 0.02 deg/s white noise)
        #     dt=dt,
        #     L=L,
        # )
        new_imu_accel_x = accel[0,:]
        new_imu_accel_y = accel[1,:]
        new_imu_accel_z = accel[2,:]
        new_imu_gyro_x = gyro[0,:]
        new_imu_gyro_y = gyro[1,:]
        new_imu_gyro_z = gyro[2,:]
        
        # replace imu fields in df
        df['imu_accel_x'] = new_imu_accel_x
        df['imu_accel_y'] = new_imu_accel_y
        df['imu_accel_z'] = new_imu_accel_z
        df['imu_gyro_x'] = new_imu_gyro_x
        df['imu_gyro_y'] = new_imu_gyro_y
        df['imu_gyro_z'] = new_imu_gyro_z
        
        # write output csv
        if WRITE_CSV:
            # retrieve output file
            out_dir = Path(out_root_dir + '\\' + set + '\\' + subset + '\\' + subset + '.csv')
            out_dir.parent.mkdir(parents=True, exist_ok=True) # make directory if it doesn't exist
            df.to_csv(out_dir, index=False)
            print(f'Processed and wrote new csv to: {out_dir}')
        
        # plots
        if VISUALIZE:
            t = df['t']
            plt.figure('Accel')
            plt.suptitle('Sim Accelerometer vs. TS')
            plt.subplot(311)
            plt.plot(t, lin_accel[0], '--k')
            plt.plot(t, new_imu_accel_x)
            plt.plot(t, df['imu_accel_x'])
            plt.legend(['TS', 'New', 'Old'])
            plt.subplot(312)
            plt.plot(t, lin_accel[1], '--k')
            plt.plot(t, new_imu_accel_y)
            plt.plot(t, df['imu_accel_y'])
            plt.subplot(313)
            plt.plot(t, lin_accel[2] - 9.81, '--k')
            plt.plot(t, new_imu_accel_z)
            plt.plot(t, df['imu_accel_z'])
            plt.show()
            
            plt.figure('Accel Error')
            plt.suptitle('Sim Accelerometer Error vs. TS')
            plt.subplot(311)
            plt.plot(t,  lin_accel[0] - new_imu_accel_x)
            plt.plot(t,  lin_accel[0] - df['imu_accel_x'])
            plt.legend(['New', 'Old'])
            plt.subplot(312)
            plt.plot(t, lin_accel[1] - new_imu_accel_y)
            plt.plot(t, lin_accel[1] - df['imu_accel_y'])
            plt.subplot(313)
            plt.plot(t, (lin_accel[2] - 9.81) - new_imu_accel_z)
            plt.plot(t, (lin_accel[2] - 9.81) - df['imu_accel_z'])
            plt.show()
            
            plt.figure('Gyro')
            plt.suptitle('Sim Gyro vs. TS')
            plt.subplot(311)
            plt.plot(t, ang_vel[0], '--k')
            plt.plot(t, new_imu_gyro_x)
            plt.plot(t, df['imu_gyro_x'])
            plt.legend(['TS', 'New', 'Old'])
            plt.subplot(312)
            plt.plot(t, ang_vel[1], '--k')
            plt.plot(t, new_imu_gyro_y)
            plt.plot(t, df['imu_gyro_y'])
            plt.subplot(313)
            plt.plot(t, ang_vel[2], '--k')
            plt.plot(t, new_imu_gyro_z)
            plt.plot(t, df['imu_gyro_z'])
            plt.show()
            
            plt.figure('Gyro Error')
            plt.suptitle('Sim Gyro Error vs. TS')
            plt.subplot(311)
            plt.plot(t, ang_vel[0] - new_imu_gyro_x)
            plt.plot(t, ang_vel[0] - df['imu_gyro_x'])
            plt.legend(['New', 'Old'])
            plt.subplot(312)
            plt.plot(t, ang_vel[1]- new_imu_gyro_y)
            plt.plot(t, ang_vel[1]- df['imu_gyro_y'])
            plt.subplot(313)
            plt.plot(t, ang_vel[2] - new_imu_gyro_z)
            plt.plot(t, ang_vel[2] - df['imu_gyro_z'])
            plt.show()
        
        stop=1