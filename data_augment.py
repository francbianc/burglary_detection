import os
import random
from moviepy.editor import VideoFileClip, vfx

# PATHS: videos_to_flip_path

# Put all the videos you want to flip horizzontally into an unique folder
videos_to_flip_path = '/Volumes/DISK/UCF_Crimes/Videos/Flipping'
names = [i for i in os.listdir(videos_to_flip_path)]

for n in names: 
    path = os.path.join(videos_to_flip_path, n)
    clip = VideoFileClip(path)
    reversed_clip = clip.fx(vfx.mirror_x)
    # Save the flipped video in the same folder, with a 'flip_' prefix in the name
    reversed_clip.write_videofile(os.path.join(videos_to_flip_path, 'flip_'+n.split('/')[-1]))