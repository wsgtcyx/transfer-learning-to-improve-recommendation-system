import pickle
import numpy as np
import pandas as pd
import cPickle


def loadData(file_path):
    u_app = pd.read_csv(open(file_path, "r"))

    data = np.load(open('data/5_rating_matrix_all.npz','r'))
    uids = data['uid']
    aids = data['aid']
    uart = data['u_art']
    uapp = data['u_app']
    return uids, aids, uart, uapp

def load_prediction_matrix():
    pred_data=np.load(open('data/3_SVD_XYZ_tf.npz','r'))
    user_matrix=pred_data['X']
    article_matrix=pred_data['Y']
    app_matrix=pred_data['Z']

    pred_matrix=np.dot(user_matrix,article_matrix.T)
    return pred_matrix, user_matrix,article_matrix,app_matrix


def topN(pred_matrix, uids, aids, N=5):
    top_scores = pred_matrix[:,:N]
    top_ids = np.array([aids[:N]*len(uids)]).reshape((len(uids),N))

    print 'start'
    n_slides = pred_matrix.shape[1]

    for i in range(N,n_slides):
        if i%500==0:
            print i
        curr = pred_matrix[:, i]
        hist_min = np.min(top_scores, axis=1)
        hist_min_idx = np.argmin(top_scores, axis=1)
        replace = curr > hist_min
        for j in range(replace.shape[0]):
            if replace[j]:
                top_scores[j,hist_min_idx[j]] = curr[j]
                top_ids[j,hist_min_idx[j]] = aids[i]

    return top_scores, top_ids


def gen_pred_pairs(uids, top_ids):
    result = []
    _N = len(uids)
    _M = top_ids.shape[1]

    for u in range(_N):
        for a in range(_M):
            result.append((uids[u],top_ids[u,a]))

    return result

def gen_pred_pairs2(uids, top_ids):
    result = []
    _N = len(uids)
    _M = top_ids.shape[1]

    for u in range(_N):
        for a in range(_M):
            result.append((uids[u],top_ids[u,a]))

    return result

def gen_true_pairs():
    data=pickle.load(open('data/1_labels.pkl','r'))
    pairs=data.index
    pairs=list(pairs)
    return pairs

def performance_measure_test(true_pairs,pred_pairs):
    """
    :param true_pairs:  a list of tuples, (userid, articleID) ,each user we will generate 1~5 pairs
    :param pred_pairs:  a list tuples, (userid, articleID) each user we will generate 1-5 pairs
    :return:  recall and precision
    """
    true_pairs=set(true_pairs)
    pred_pairs=set(pred_pairs)

    num_true_all=len(true_pairs)
    num_pred_all=len(pred_pairs)

    TP=len(true_pairs.intersection(pred_pairs))*1.0
    precision=TP/num_pred_all
    recall = TP/num_true_all  # if you have read this article, it means that
    print "precision:",precision," recall:",recall
    return precision,recall
def transform_original_pair_2_l1pair(pairs):
    article_cat_dict=pickle.load(open('data/7_dict_article_l1cat.pkl','r'))
    result=[]
    for pair in pairs:
        try:
            temp_pair=(pair[0],article_cat_dict[pair[1]])
        except KeyError:
            pass
        else:
            result.append(temp_pair)
    "print transform finish"
    return result



if __name__ == '__main__':
    N = 3 # number of articles to be recommended to each user

    path = "data/2_user_with_article_all.csv"
    uids, aids, uart, uapp = loadData(path)
    uids, aids = list(uids),list(aids)

    pred_matrix, user_matrix, article_matrix, app_matrix=load_prediction_matrix()

    pred_matrix=np.load("data/3_rating_matrix.npz")['arr_0']
    top_scores, top_ids = topN(pred_matrix, uids, aids, N)

    print 'step1'
    pred_pairs = gen_pred_pairs(uids, top_ids)
    print 'step2'
    true_pairs = gen_true_pairs()
    print 'step3'
    #pred_pairs_l1=transform_original_pair_2_l1pair(pred_pairs)
    #true_pairs_l1=transform_original_pair_2_l1pair(true_pairs)
    print 'step4'

    precision, recall=performance_measure_test(true_pairs,pred_pairs)

    # CMF
    # top  3:precision: 0.00502220561989  recall: 0.001040896030
    # top  5:precision: 0.00373552484124  recall: 0.001290366980
    # top 10:precision: 0.00247789814469  recall: 0.001711886860
    # top 15:precision: 0.00190926825219  recall: 0.001978562703

 # top 5 cat:precision: 0.29917011031  recall: 0.385631273111










