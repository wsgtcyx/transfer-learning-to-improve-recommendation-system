import pickle
import numpy as np
import pandas as pd
from model_CMF import transform_original_pair_2_l1pair

def loadData(file_path):
    u_app = pd.read_csv(open(file_path, "r"))

    data = np.load(open('data/5_rating_matrix_all.npz','r'))
    uids = data['uid']
    aids = data['aid']
    uart = data['u_art']
    uapp = data['u_app']
    return uids, aids, uart, uapp

def gen_true_pairs():
    data=pickle.load(open('data/1_labels.pkl','r'))
    pairs=data.index
    pairs=list(pairs)
    return pairs

def train_model():
    print "begin train"
    result_pd=pd.read_csv(open('data/2_user_with_article_all.csv','r'))

    labels = result_pd['scores'].as_matrix()
    result_drop = result_pd.drop(['scores','Unnamed: 0','Unnamed: 1'],axis=1)
    X = result_drop.as_matrix()
    print result_drop.columns
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X, labels.ravel())
    print reg.score(X, labels.ravel())
    return reg
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
def gen_pred_pairs(N,uids):
    # generate pred_pairs
    user_profile=pickle.load(open('data/1_user_with_app.pkl','r'))
    article_profile=pickle.load(open('data/1_article_df_finish.pkl','r'))

    num_user_feat=len(user_profile.columns)
    num_art_feat=len(article_profile.columns)
    num_art=len(article_profile.index)
    num_user=len(uids)
    article_list = article_profile.index
    pred_pairs=[]
    for uid_index in range(num_user):
        if uid_index%100==0:
            print uid_index
        user_all_data=np.zeros((num_art,num_art_feat+num_user_feat))
        temp_user=user_profile.loc[uids[uid_index]].as_matrix()
        user_all_data[:,:num_user_feat]=temp_user

        temp_article_profile=article_profile.as_matrix()
        user_all_data[:,num_user_feat:]=temp_article_profile

        pred_scores=RM_model.predict(user_all_data)
        topN_indexs = pred_scores.argsort()[-N:]
        for topN_index in topN_indexs:
            temp_pair=(uids[uid_index],article_list[topN_index])
            pred_pairs.append(temp_pair)
    return pred_pairs
if __name__ == '__main__':

    path = "data/2_user_with_article_all.csv"
    uids, aids, uart, uapp = loadData(path)
    uids, aids = list(uids), list(aids)
    N=3
    RM_model=train_model()
    true_pairs=gen_true_pairs()

    pred_pairs=gen_pred_pairs(N,uids)
    #performance
    pred_pairs_l1 = transform_original_pair_2_l1pair(pred_pairs)
    true_pairs_l1 = transform_original_pair_2_l1pair(true_pairs)
    print 'step4'

    precision, recall = performance_measure_test(true_pairs_l1, pred_pairs_l1)
    # Random Forest
    #top  3:precision: 0.0020337857469  recall: 0.000421519880254
    #top  5:precision: 0.00166853442909  recall: 0.000576363917898
    #top 10:precision: 0.00125762669655  recall: 0.000868847100115
    #top 15:precision: 0.00111235628606  recall: 0.0011527278358


    #top  3:precision: 0.212439647827  recall: 0.0987849973587

    #pred_pairs=gen_pred_pairs(RM_model,)











