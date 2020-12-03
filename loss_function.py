import tensorflow as tf

# AIM: define the customized loss function of Sultani et al. (2019)

def custom_objective(y_true, y_pred):
    """
    Implementation of Sultani et al. (2019) custom loss function. 
    Assumption: Multiple Instance Learning.
        That is we only know the video-level labels, not the frame-level ones. A video is normal or contains anomaly somewhere, but we do not know where. 
        This is intriguing because we can easily annotate a large number of videos by only assigning video-level labels.
    Elements: 
        1. Hinge Loss: max [0, (1 − highest_score_abnormal + highest_score_normal)] where score_normal = S_n, score_abnormal = S_a 
        2. Smoothing constraint: sum[(S_a_i − S_a_i+1)^2] where i = 1, ... 32
        3. Sparsity constraint: sum[S_a_i] 
    
    Parameters
    ----------
    y_true : tf.tensor
        True labels of the 32 features extracted with the C3D or I3D model
    y_pred : tf.tensor
        Predicted labels of the 32 features extracted with the C3D or I3D model (sometimes referred to as scores)

    Returns
    ----------
    float
        Value of the loss function computed from a batch of 60 videos, 30 normal and 30 abnormal
    """
    
    # Flatten tensors 
    y_true = tf.reshape(y_true, [-1]) 
    y_pred = tf.reshape(y_pred, [-1]) 

    n_seg = 32              # number of features per video 
    nvid = 60               # batch size
    n_exp = int(nvid / 2)   
    Num_d = int(32 * nvid)  # total number of features of a batch 

    # Create some tensors of ones 
    sub_max = tf.ones_like(y_pred)
    sub_sum_labels = tf.ones_like(y_true)
    sub_sum_l1 = tf.ones_like(y_true)
    sub_l2 = tf.ones_like(y_true)
    
    # For each video in a batch (ii = 0,1,2,...59)
    for ii in range(0, nvid):
        # Take the corresponding 32 true labels and sum them up
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = tf.concat([sub_sum_labels, [tf.math.reduce_sum(mm)]], 0)   

        # Take the corresponding 32 predicted labels, take the biggest prediction and sum them up
        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = tf.concat([sub_max, [tf.reduce_max(Feat_Score, 0)]], 0)           
        sub_sum_l1 = tf.concat([sub_sum_l1, [tf.math.reduce_sum(Feat_Score)]], 0)   
        
        # SMOOTHING CONSTRAINT
        # Compute: (𝑆_𝑖 − 𝑆_𝑖+1) where S stands for score (= predicted label) 
        z1 = tf.ones_like(Feat_Score)
        z2 = tf.concat([z1, Feat_Score], 0)
        z3 = tf.concat([Feat_Score, z1], 0)
        z_22 = z2[31:]  # 1 and 32 predictions  [1, s1, s2, ... s32]
        z_44 = z3[:33]  # 32 predictions and 1  [s1, s2, ... s32, 1]
        z = z_22 - z_44
        z = z[1:32]
        
        # Compute: ∑(𝑆_𝑖 − 𝑆_𝑖+1)^2
        z = tf.math.reduce_sum(tf.math.square(z))                                   
        sub_l2 = tf.concat([sub_l2, [z]], 0)                                        

    # Remove from the tensors concatenated before the ones 
    sub_score = sub_max[Num_d:] # contains the highest predicted scores per each video in the batch 
    F_labels = sub_sum_labels[Num_d:]
    
    # SMOOTHING AND SPARSITY CONSTRAINTS
    sub_sum_l1 = sub_sum_l1[Num_d:]
    sub_sum_l1 = sub_sum_l1[:n_exp] # Input for the Smoothing constraint 
    sub_l2 = sub_l2[Num_d:]
    sub_l2 = sub_l2[:n_exp] # Input for the Sparsity constraint 
    
    # Get indices of normal and abnormal videos 
    indx_nor = tf.where(tf.equal(F_labels, 32))  # normal are positioned where the sum is = 32                                    
    indx_abn = tf.where(tf.equal(F_labels, 0))   # abnormal are positioned where the sum is = 0                                

    # n_exp = 30
    n_Nor = n_exp

    # HINGE LOSS
    # Get 32*30 normal and 32*30 abnormal scores of the whole batch 
    Sub_Nor = tf.gather_nd(sub_score, indx_nor)                                     
    Sub_Abn = tf.gather_nd(sub_score, indx_abn)                                     
    z = tf.ones_like(y_true)

    for ii in range(0, n_Nor):
        # Compute: max [0, (1 − max_S_a + max_S_n)] per each combination*
        sub_z = tf.reduce_max(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = tf.concat([z, [tf.math.reduce_sum(sub_z)]], 0)                            

    # COMPLETE LOSS FUNCTION
    lambda_1 = 0.00008
    lambda_2 = 0.00008
    z = z[Num_d:]
    z = tf.math.reduce_mean(z) + lambda_1 * tf.math.reduce_sum(sub_sum_l1) + lambda_2 * tf.math.reduce_sum(sub_l2)
    
    #* per each combination means that we want to compute the difference between each highest_S_a and each highest_S_n in the batch. 
    # Therefore, we take all the 30 highest abnormal scores and at each iteration we subtract them with one of the 30 highest normal scores.  
    return z
