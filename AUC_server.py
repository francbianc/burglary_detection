import numpy as np
import os
from utils.visualization_util import *
from sklearn import metrics
import configuration as cfg

# PATHS: cfg.path_all_features, cfg.Ann_path, cfg.NamesAnn_path, './num_frames.txt'
# VARIABLES: cfg.train_exp_name, cfg.use_i3d

path_all_features = cfg.path_all_features
experiment_name = cfg.train_exp_name

# Score is a folder in path_all_features, created by test_detect_server.py, that contains 310 txt files of scores
score_path = os.path.join(path_all_features, 'Scores_'+experiment_name)

all_annotations = [line.strip().split() for line in open(cfg.Ann_path, 'r')]
all_annotations_dict = {l[0].split('.')[0]:l[-4:] for l in all_annotations}

all_ann_names = [line.strip().replace('.mp4', '.txt').replace('/', '_Features/') for line in open(cfg.NamesAnn_path, 'r')]
all_frames = [i.strip().split() for i in open('./num_frames.txt', 'r')]
all_frames_dict = {l[0]:l[1:] for l in all_frames}
if cfg.use_i3d: 
   all_ann_names = [line.strip().replace('.mp4', '.txt').replace('/', '_Features_I3D/') for line in open(cfg.NamesAnn_path, 'r')]
   all_frames_dict = {l[0].replace('/', '_I3D/'):l[1:] for l in all_frames}

# Divide test videos: Sultani vs. Giovanni to compute different AUC metrics
gios = [i.strip().replace('.mp4', '') for i in open('./video_gio_names.txt', 'r')]
gio_annotations_dict = {k:v for k,v in all_annotations_dict.items() if k in gios}
assert len(gio_annotations_dict) == 20
sul_annotations_dict = {k:v for k,v in all_annotations_dict.items() if k not in gios}
assert len(sul_annotations_dict) == 290

gio_ann_names = [i for i in all_ann_names if i.split('/')[1].replace('.txt', '') in gios]
assert len(gio_ann_names ) == 20
sul_ann_names = [i for i in all_ann_names if i.split('/')[1].replace('.txt', '') not in gios]
assert len(sul_ann_names) == 290

gio_frames_dict = {k:v for k,v in all_frames_dict.items() if k.split('/')[1].replace('.txt', '') in gios}
assert len(gio_frames_dict ) == 20
sul_frames_dict = {k:v for k,v in all_frames_dict.items() if k.split('/')[1].replace('.txt', '') not in gios}

def run_test(all_annotations_dict, all_frames_dict, all_ann_names):
   no_video = 1
   frm_counter = 0
   All_Detect = np.zeros(100000000)
   All_GT = np.zeros(100000000)
   
   for filename in all_ann_names:
      if filename.endswith('.txt'):
         video_name = os.path.join(path_all_features, filename)
         name = os.path.basename(video_name).split('.')[0]
         #print(name)
         name_txt = name + '.txt'
         
         scores = os.path.join(score_path, name_txt)
         score = [line.strip() for line in open(scores, 'r')]
         # list of 32 str (each str contains 1 score) 

         C3D_file = [line.strip() for line in open(video_name, 'r')]
         # list of 32 str (each str contains 4096 floats)

         Ann = all_annotations_dict[name] 
         # list of 4 str (each str contains 1 annot as str)

         num_frames = int(all_frames_dict[filename][0])
         # integer
         
         # assign to each frame the anomaly score of the feature it belongs to
         num_features = len(C3D_file)     # must be 32
         num_frames_C3D = num_features*16 # as the features were computed for every 16 frames
         Detection_score_32shots = np.zeros(num_frames_C3D)
         Thirty2_shots = np.round(np.linspace(0, num_features, 32))

         l = range(len(Thirty2_shots))
         p_c = -1
         for c_shots, n_shots in zip (l, l[1:]):
            p_c = p_c + 1
            ss = Thirty2_shots[c_shots]
            ee = Thirty2_shots[n_shots] - 1
            #print('ss:', ss, 'ee:', ee)
            #print('c_shots:', c_shots, 'n_shots:', n_shots)

            if c_shots == len(Thirty2_shots):
               ee=Thirty2_shots[n_shots]

            if ee<ss:
               Detection_score_32shots[(int(ss))*16:(int(ss))*16+16+1] = score[p_c]
               #print(ee < ss)
            else:
               Detection_score_32shots[(int(ss))*16:(int(ee))*16+16+1] = score[p_c]
               #print(ee > ss)

         #print(num_frames)
         #print(len(Detection_score_32shots))
         if num_frames > len(Detection_score_32shots):
            Final_score = np.append(Detection_score_32shots, np.repeat(Detection_score_32shots[-1], [num_frames-len(Detection_score_32shots)]))
            GT=np.zeros(num_frames)
         else:
            Final_score = Detection_score_32shots
            GT=np.zeros(len(Detection_score_32shots))

         # check the temporal annotation
         t_txt = [int(i) for i in Ann]
         
         for y in range(0,3,2):
            if t_txt[y] >= 0:
               st_fr = max(int(float(t_txt[y])), 0)
               end_fr = min(int(float(t_txt[y+1])), num_frames)
               GT[st_fr:end_fr+1] = 1

         All_Detect[frm_counter:frm_counter+len(Final_score)] = Final_score
         All_GT[frm_counter:frm_counter+len(Final_score)] = GT
         print('Video ', no_video, ' successfully processed!')
         no_video = no_video + 1
         frm_counter = frm_counter+len(Final_score)

   All_Detect = (All_Detect[0:frm_counter])
   All_GT = All_GT[0:frm_counter]
   tot_scores = All_Detect
   si = tot_scores.argsort()[::-1]
   tp = All_GT[si] > 0
   fp = All_GT[si] == 0
   tp = np.cumsum(tp)
   fp = np.cumsum(fp)
   nrpos = sum(All_GT)
   rec = tp / nrpos
   fpr = fp / sum(All_GT == 0)
   prec = tp / (fp + tp)
   AUC1 = np.trapz(rec, fpr)
   print('AUC1: ', AUC1)

   fpr, tpr, thresholds = metrics.roc_curve(All_GT, All_Detect)
   AUC2 = metrics.auc(fpr, tpr)
   print('AUC2: ', AUC2)
   return AUC1, AUC2

print('>> AUC with 310 test videos')
AUC1_all, AUC2_all = run_test(all_annotations_dict, all_frames_dict, all_ann_names)

print('>> AUC with 290 SULTANI test videos')
AUC1_sul, AUC2_sul = run_test(sul_annotations_dict, sul_frames_dict, sul_ann_names)

print('>> AUC with 20 GIOSS test videos')
AUC1_gio, AUC2_gio = run_test(gio_annotations_dict, gio_frames_dict, gio_ann_names)