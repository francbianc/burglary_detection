import tensorflow as tf

def custom_objective(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1]) 
    y_pred = tf.reshape(y_pred, [-1]) 

    n_seg = 32
    nvid = 60
    n_exp = int(nvid / 2)
    Num_d = int(32 * nvid)

    sub_max = tf.ones_like(y_pred)
    sub_sum_labels = tf.ones_like(y_true)
    sub_sum_l1 = tf.ones_like(y_true)
    sub_l2 = tf.ones_like(y_true)

    for ii in range(0, nvid):
        mm = y_true[ii * n_seg:ii * n_seg + n_seg]
        sub_sum_labels = tf.concat([sub_sum_labels, [tf.math.reduce_sum(mm)]], 0)   

        Feat_Score = y_pred[ii * n_seg:ii * n_seg + n_seg]
        sub_max = tf.concat([sub_max, [tf.reduce_max(Feat_Score, 0)]], 0)           
        sub_sum_l1 = tf.concat([sub_sum_l1, [tf.math.reduce_sum(Feat_Score)]], 0)   

        z1 = tf.ones_like(Feat_Score)
        z2 = tf.concat([z1, Feat_Score], 0)
        z3 = tf.concat([Feat_Score, z1], 0)
        z_22 = z2[31:]
        z_44 = z3[:33]
        z = z_22 - z_44
        z = z[1:32]
        z = tf.math.reduce_sum(tf.math.square(z))                                   
        sub_l2 = tf.concat([sub_l2, [z]], 0)                                        

    sub_score = sub_max[Num_d:]
    F_labels = sub_sum_labels[Num_d:]

    sub_sum_l1 = sub_sum_l1[Num_d:]
    sub_sum_l1 = sub_sum_l1[:n_exp]
    sub_l2 = sub_l2[Num_d:]
    sub_l2 = sub_l2[:n_exp]

    indx_nor = tf.where(tf.equal(F_labels, 32))                                     
    indx_abn = tf.where(tf.equal(F_labels, 0))                                      

    n_Nor = n_exp

    Sub_Nor = tf.gather_nd(sub_score, indx_nor)                                     
    Sub_Abn = tf.gather_nd(sub_score, indx_abn)                                     
    z = tf.ones_like(y_true)

    for ii in range(0, n_Nor):
        sub_z = tf.reduce_max(1 - Sub_Abn + Sub_Nor[ii], 0)
        z = tf.concat([z, [tf.math.reduce_sum(sub_z)]], 0)                            

    z = z[Num_d:]
    z = tf.math.reduce_mean(z) + 0.00008 * tf.math.reduce_sum(sub_sum_l1) + 0.00008 * tf.math.reduce_sum(sub_l2)
    
    return z