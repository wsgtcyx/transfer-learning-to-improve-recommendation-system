
import pickle
import numpy as np
import pandas as pd
import cPickle
import time

def generate_both_matrix():

    data2 = pickle.load(open("data/0_app_profile", "r"))

    apps = set()
    for user in data2.keys():
        apps = apps.union(set(data2[user]['user_all_pkgs']))

    apps = list(apps)
    uids = data2.keys()
    uids = list(set(uids))
    
    # constructs a matrix, where the first layer is in binary, and the second is a real valued rating score
    result_dict = pickle.load(open('data/1_labels.pkl', 'r'))
    pairs = result_dict.index
    uids = uids
    aids = []
    for pair in pairs:
        # uids.append(pair[0])
        aids.append(pair[1])
    
    aids = list(set(aids))
    matrix1 = np.zeros((len(uids), len(aids)))
    matrix_B = np.zeros((len(uids), len(aids)))
    print len(uids), len(aids)




    i = 0
    for pair in pairs:
        i += 1
        if i % 1000 == 0:
            print i
        (uid, aid) = uids.index(pair[0]), aids.index(pair[1])
        matrix1[uid, aid] = result_dict.loc[pair]
        matrix_B[uid, aid] = 1

    matrix2 = np.zeros((len(uids), len(apps)))
    i = 0
    for uid_index in range(len(uids)):
        temp_apps = data2[uids[uid_index]]['user_all_pkgs']
        for app in temp_apps:
            i += 1
            if i % 3000 == 0:
                print i
            app_index = apps.index(app)
            matrix2[uid_index, app_index] = 1
    print "done"

    result={'scores': matrix1,'scores_B': matrix_B, 'aids': np.array(aids), 'uids': np.array(uids),'matrix_apps':matrix2,'apps':np.array(apps)}


    np.savez(open("data/5_rating_matrix_all.npz", "w"),u_art=matrix1,u_art_b=matrix_B,u_app=matrix2,uid=uids,aid=aids)
    print "DONE"

generate_both_matrix()











