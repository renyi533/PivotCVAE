# preproccess data: movielens-100

# 首先把每个user的slate划分为size为5
# rating: 4-5 label 1
# rating: 0-3 label 0
# format: 
# DATA_ROOT + "movielens/test_slate.csv":  5个int类型： item1,item2,item3,item4,item5
# DATA_ROOT + "movielens/test_user.csv: 1个int类型：uid
# DATA_ROOT + "movielens/test_resp.csv: 5个int类型： [1,0,0,0,0]
import pandas as pb
import csv
from tqdm import tqdm
import os
import pdb

# 一共五份数据
# 从./mk_100k里的u1.base/u1.test生成../data/movielens/train1_slate.csv等
for i in range(1,2):
    for data_type in ['base', 'val', 'test']:
        base_dir = "./ml_1m/u{}.".format(i)+data_type
        if data_type == 'base':
            date_tyepe0 = 'train'
        elif data_type == 'test':
            date_tyepe0 = 'test'
        else:
            date_tyepe0 = 'val'
        slate_dir = "../data/movielens/"+date_tyepe0+"{}_slate.csv".format(i)
        user_dir = "../data/movielens/"+date_tyepe0+"{}_user.csv".format(i)
        resp_dir = "../data/movielens/"+date_tyepe0+"{}_resp.csv".format(i)
#         print("base_dir: ", base_dir)
#         print("slate_dir: ", slate_dir)
#         print("user_dir: ", user_dir)
#         print("resp_dir: ", resp_dir)
        user_dict = {}
        with open(base_dir, 'rt') as u1test:
            data = csv.reader(u1test)
            for row in tqdm(data):
                user_id, item_id, rating, time = row[0].split('\t')
                if user_id not in user_dict:
                    user_dict[user_id] = [[time, item_id, rating]]
                else:
                    user_dict[user_id].append([time, item_id, rating])
            print('num_user:', len(user_dict.keys()))
        slate_set = []
        resp_set = []
        user_set = []
        for key in user_dict.keys():
            # 按照时间戳排序
            user_dict[key] = sorted(user_dict[key])
            # 遍历所有的uid, 在当前的uid下，每5个item作为一个slate
              # 提取出所有的
            item = [item for _, item, rating in user_dict[key]]
            rating = [1 if int(rating) > 3 else 0 for _, item, rating in user_dict[key]]
            for k in range(0, len(user_dict[key])//5*5, 5):
                resp_set.append(rating[k:k+5])
                user_set.append(key)
                slate_set.append(item[k:k+5])        
        print('number of samples:', len(user_set))
        if not os.path.exists("../data/movielens/"):
            os.makedirs("../data/movielens/")
        if os.path.exists(slate_dir):
            print(slate_dir ,"already exists!")
            break
        if os.path.exists(user_dir):
            print(user_dir ,"already exists!")
            break
        if os.path.exists(resp_dir):
            print(resp_dir ,"already exists!")
            break 
        with open(slate_dir, 'wt') as u1test_slate:
            cw1 = csv.writer(u1test_slate)
            for slate in slate_set:
                cw1.writerow(slate)
        with open(user_dir, 'wt') as u1test_user:
            cw2 = csv.writer(u1test_user)
            for user in user_set:
                cw2.writerow(user)
        with open(resp_dir, 'wt') as u1test_resp:
            cw3 = csv.writer(u1test_resp)
            for resp in resp_set:
                cw3.writerow(resp)