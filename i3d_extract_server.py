import os
import numpy as np
from i3d import *
from utils.visualization_util import *

## USE THIS '_server' EXTENSION IF YOU WANT TO USE THE ENTIRE DATASET

# PATHS: './video_paths.txt', cfg.path_all_videos
# AIM: extract from 1 video, n features. Merge n features to end with 32 features and save them in a .txt file

path_all_videos = cfg.path_all_videos

def run_i3d():
    '''
    Starting from videos, extract features using the I3D model pre-trained on the Kinetics dataset and save them as .txt files.
    How?
        Each video is divided into n clips of 16 frames, where n = int(np.round(total_number_of_frames/16)).
        Each clip is given as input to the I3D model, that returns as output an array of 1024 floats. This array is 1 feature. 
        (n, 1024) features are then merged to end up with (32, 1024) features per each video.

    1. './video_paths.txt' = file with all the 19 names of the folders that contain videos by category 
    2. cfg.I3D_path = file where done_check.txt is saved (optional)
    3. cfg.path_all_videos = folder that contains all the 19 subfolders with videos by category
    '''
    
    # List with the paths of all the videos, that are divided into 19 categories
    input_path = [line.strip() for line in open('./video_paths.txt', 'r')]
    input_path = [os.path.join(path_all_videos, i) for i in input_path]
    
    # Create a folder per each category where features will be saved
    feat_output_path = [line.strip()+'_Features_I3D' for line in open('./video_paths.txt', 'r')]
    feat_output_path = [os.path.join(path_all_videos, i) for i in feat_output_path]
    
    assert len(input_path) == len(feat_output_path)
    
    for path in feat_output_path:
        if not os.path.exists(path):
            os.makedirs(path)

    print('I3D Features directories initialized')
    done_check = []

    for path in range(len(input_path)):
        for filename in os.listdir(input_path[path]):
            if filename.endswith('.mp4'):
                video_name = os.path.join(input_path[path], filename)       
                name = os.path.basename(video_name).split('.')[0]           

                # Read video: create n clips of 16 frames
                video_clips, num_frames = get_video_clips(video_name)
                print("Number of clips in the video : ", len(video_clips))
                
                # Initialize I3D model 
                feature_extractor = i3d_feature_extractor()
                
                # Extract features
                rgb_features = []
                for i, clip in enumerate(video_clips):
                    clip = np.array(clip)
                    if len(clip) < params.frame_count:
                        continue

                    clip = preprocess_input(clip)
                    rgb_feature = feature_extractor.predict(clip)[0] 
                    rgb_feature = rgb_feature.reshape(1024,)
                    rgb_features.append(rgb_feature)
                    print("Processed clip : {} / {}".format(i, len(video_clips)))
                    
                # Bag features: from n to 32 features per each video
                rgb_features = np.array(rgb_features)
                rgb_feature_bag = interpolate(rgb_features, params.features_per_bag)
                
                # Save each bag of features as .txt
                save_path = os.path.join(feat_output_path[path], name + '.txt')
                np.savetxt(save_path, rgb_feature_bag)
                done_check.append(video_name)
            
            # Can save the names of those videos whose features have been already extracted
            #cat_1 = done_check[0].split('/')[-1].strip('mp4')
            #cat_2 = ''.join(filter(str.isalpha, cat_1))[:-1]
            #with open(os.path.join(cfg.I3D_path, 'done_check_'+cat_2+'.txt'), 'w') as f0:
                #print(done_check, file=f0)
        #print(path)
            
        
if __name__ == '__main__':
   run_i3d()
