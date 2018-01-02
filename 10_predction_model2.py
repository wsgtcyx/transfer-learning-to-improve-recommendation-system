import pickle
import numpy as np
import pandas as pd

def load_prediction_matrix():
    pred_data=np.load(open('data/3_SVD_XYZ_tf.npz','r'))
    user_matrix=pred_data['X']
    article_matrix=pred_data['Y']
    app_matrix=pred_data['Z']

    pred_matrix=np.dot(user_matrix,article_matrix.T)
    return pred_matrix, user_matrix,article_matrix,app_matrix
def load_data():
    path_user = 'data/7_KNN_user_data'
    path_article = 'data/7_KNN_art_data_update.pkl'
    user_data = pickle.load(open(path_user,'r'))
    art_data = pickle.load(open(path_article,'r'))
    tr_u, tst_u = user_data['train_user'], user_data['test_user']
    tr_a, tst_a = art_data['train_art'], art_data['test_art']
    return tr_u, tr_a, tst_u, tst_a

def loadData(file_path):
    u_app = pd.read_csv(open(file_path, "r"))

    data = np.load(open('data/5_rating_matrix_all.npz','r'))
    uids = data['uid']
    aids = data['aid']
    uart = data['u_art']
    uapp = data['u_app']
    return uids, aids, uart, uapp

def generate_old_user_to_new_article_rating_matrix():

    print len(tst_aids)
    for new_aid_index in range(len(tst_aids)):
        if new_aid_index%200==0:
            print new_aid_index
        old_article_aid_index=knn_a[new_aid_index][0]
        old_article_aid=tr_aids[old_article_aid_index]
        #print old_article_aid
        rating_col=pred_matrix[:,list(aids).index(old_article_aid)]
        rating_col=rating_col.reshape((-1,1))
        if new_aid_index==0:
            result_matrix=rating_col.copy()
            print old_article_aid,rating_col
            #break
        else:
            result_matrix=np.hstack((result_matrix,rating_col))

    print result_matrix.shape
    return result_matrix

test_user_profile=pickle.load(open('data/5_user_profile','r'))
test_user_ids=test_user_profile.keys()


knn_user_result=np.load(open('data/8_knn_uids_result.npz','r'))
tr_uids=knn_user_result['tr_uids']
tst_uids=knn_user_result['tst_uids']
knn_u=knn_user_result['knn_u']

knn_article_result=np.load(open('data/8_knn_aids_result.npz','r'))
tr_aids=knn_article_result['tr_aids']
tst_aids=knn_article_result['tst_aids']
knn_a=knn_article_result['knn_a']

path = "data/2_user_with_article_all.csv"
uids, aids, uart, uapp = loadData(path)


pred_matrix, user_matrix, article_matrix, app_matrix=load_prediction_matrix()



new_rating_matrix=generate_old_user_to_new_article_rating_matrix()

result = {}
for index in range(len(test_user_ids)):
    test_user_id = test_user_ids[index]
    if index % 100 == 0:
        print index
    result[test_user_id] = []  # null list
    if test_user_id in uids:
        uid_index = list(uids).index(test_user_id)
    else:
        new_user_index = list(tst_uids).index(test_user_id)
        old_user_index = knn_u[new_user_index][0]
        old_user_id = list(tr_uids)[old_user_index]
        uid_index = list(uids).index(old_user_id)

    sorted_new_article = np.argsort(new_rating_matrix[uid_index, :])[::-1]
    top_new_article_index = sorted_new_article[:5]
    top_tst_aids = tst_aids[list(top_new_article_index)]
    result[test_user_id] = top_tst_aids


pickle.dump(new_rating_matrix,open("data/11_new_rating_matrix.pkl",'w'))
pickle.dump(result,open("data/11_submission_dict.pkl",'w'))


test_uids=result.keys()
import csv

#write file
with open('12_test_result.csv','w') as f:
    f_csv = csv.writer(f)
    for test_uid in test_uids:
        x = [test_uid]
        x.extend(result[test_uid])
        f_csv.writerow(x)

with open('12_test_result.txt','w') as f:
    for test_uid in test_uids:
        x = str(test_uid)+":"+",".join(result[test_uid])+'\n'
        f.write(x)

#check
for test_uid in test_uids:
    articles=result[test_uid]
    for article in articles:
        if article not in tst_aids:
            print "error"
