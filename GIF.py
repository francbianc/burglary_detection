import os
import numpy as np
import configuration as cfg
from utils.visualization_util import *

# PATHS: cfg.input_folder, './num_frames.txt', cfg.score_path, cfg.gif_path

visualization_names = [i.replace('.mp4', '') for i in os.listdir(cfg.input_folder) if i.endswith('.mp4')]

all_frames = [i.strip().split() for i in open('./num_frames.txt', 'r')]
all_frames_dict = {l[0].split('/')[1]:l[1:] for l in all_frames}

def run_visualization():
    for video_name in visualization_names: 
        video_path = os.path.join(cfg.input_folder, video_name+'.mp4')
        num_frames = int(all_frames_dict[video_name+'.txt'][0])
        predictions = [float(i.strip()) for i in open(os.path.join(cfg.score_path, video_name+'.txt'), 'r')]
        predictions = extrapolate(predictions, num_frames)
        save_path = os.path.join(cfg.gif_path, video_name + '.gif')
        visualize_predictions(video_path, predictions, save_path)
        print('Executed Successfully - ' + video_name + '.gif saved')
    

if __name__ == '__main__':
   run_visualization()