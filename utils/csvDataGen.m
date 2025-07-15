%%
clc
clear
close all

%%
% set parent directories of matfiles
% parent_dir = 'D:\Tahn\2025_04_22\synced_data\mat';
parent_dir = 'D:\Tahn\2025_03_07_w_camera\synced_data\a2_sensors_2025_03_07_02';
% currSetDir = 'D:\Tahn\2025_03_07_w_camera\synced_data\a2_sensors_2025_03_07_01';

fileList = dir(fullfile(parent_dir, '*.mat'));

% set directory of camera folders
% cam_parent_dir = 'D:\Tahn\2025_03_07_w_camera\synced_data';
% cam_set_base = '\a2_sensors_2025_03_07_0';
% count = 0;

% set write variables
write_csv = false;
out_path = 'D:\Tahn\2025_03_07_w_camera\synced_data\csv';
out_name = '\2025_03_07_02.csv';

% loop through mat files
for i = 1:length(fileList)
    if i > 1
        break
    end

    % generate the current set directory
    % count = count+1;
    % currSetDir = append(cam_parent_dir, cam_set_base, int2str(count));

    filename = fullfile(parent_dir, fileList(i).name);
    disp(['processing: ',filename])

    % load mat file
    mat = load(filename);
    
    t = mat.tractor_etal.time.zeroed/60;
    L = length(t);

    % solve for articulation angle and rate
    tractor_quat = [mat.tractor_etal.orientation(4,:); mat.tractor_etal.orientation(1:3,:)]; % reported quaternion (rearanged to [w,x,y,z]
    tractor_eul = quat2eul(tractor_quat', 'XYZ');
    tractor_eul(:,3) = pi/2 - unwrap(tractor_eul(:,3));
    

    trailer_quat = [mat.trailer_etal.orientation(4,:); mat.trailer_etal.orientation(1:3,:)]; % reported quaternion (rearanged to [w,x,y,z]
    trailer_eul = quat2eul(trailer_quat', 'XYZ');
    trailer_eul(:,3) = pi/2 - unwrap(trailer_eul(:,3));
    
    % solve for articulation angle and rate
    hitch = trailer_eul(:,3) - tractor_eul(:,3);
    hitch_rate = -mat.trailer_etal.angTwist(3,:)' - -mat.tractor_etal.angTwist(3,:)';
    
    % generate paths to images
    LRMC = append(parent_dir,'\cameras','\LRMC','\LRMC') + string(1:L) + '.jpg';
    RRMC = append(parent_dir,'\cameras','\RRMC','\RRMC') + string(1:L) + '.jpg';
    img_dirs = [LRMC',RRMC'];
    
    %sanity check
    % figure()
    % plot(t,rad2deg(hitch_rate))

    % generate csv
    data_mat{i} = [img_dirs, hitch, hitch_rate];
    tot_mat = vertcat(data_mat{:});
    if write_csv
        writematrix(tot_mat, append(out_path, out_name));
    end
end