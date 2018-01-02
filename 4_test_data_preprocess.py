# import json
# import time
# import pandas as pd
# !/usr/bin/python
# -*- coding: UTF-8 -*-
# path_sample = "Sample Data"
path_tr = "data/tr_data_20170128"  # train
path_tstu = "data/tst_user_20170128"  # test user
path_tsta = "data/tst_article_20170128"  # test article

import ast
import pickle


def file_process(data_filepath):
    user_profile = {}
    app_profile = {}
    article_info = {}
    users = set()
    articles = set()
    count = 0
    with open(data_filepath) as f:
        data = f.readlines()
        article_count = 1
        for line in data:

            if count % 1000 == 0:
                print(count, len(users))
            count += 1

            record = ast.literal_eval(line.strip())

            user_id = record['user_id']
            #article_id = record['article_contentid']
            users.add(user_id)
            #articles.add(article_id)

            user_profile_keys = ['user_age', 'user_gender', 'user_gender_weight', 'user_model', 'user_brand',
                                 'user_country', 'user_maxmind_country_iso_code', 'user_maxmind_state_iso_code',
                                 'user_maxmind_city']
            app_profile_keys = ['user_all_pkgs', 'user_gp_frequency', 'user_gp_keywords']

            if user_id in user_profile.keys():
                article_count += 1
                article_info[user_id][article_count] = {}

            else:
                article_count = 1
                user_profile[user_id] = {}
                app_profile[user_id] = {}
                article_info[user_id] = {}
                article_info[user_id][article_count] = {}
                for key in user_profile_keys:
                    if key in record.keys():
                        user_profile[user_id][key] = record[key]
                for key in app_profile_keys:
                    if key in record.keys():
                        app_profile[user_id][key] = record[key]


    pickle.dump(user_profile, open('data/5_user_profile_test', 'w'))
    pickle.dump(app_profile, open('data/5_app_profile_test', 'w'))

    print(count, len(users),  '=========')
    return user_profile, app_profile,


if __name__ == '__main__':
    user_profile, app_profile,  = file_process(path_tstu)

