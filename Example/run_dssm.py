import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from DLFeatureFactory.featureColumes import DenseFeat, SparseFeat, VarLenSparseFeat
from Model.Deep_matching.DSSM import  DSSM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
import faiss



def process_data(data_path, neg_sample=0):
    """读取数据
    """
    data_df = pd.read_csv(data_path, sep=',')

    use_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip']

    # 特征转换, 类别编码
    feature_max_index_dict = {}
    for feat in use_features:
        lbe = LabelEncoder()
        data_df[feat] = lbe.fit_transform(data_df[feat]) + 1  # 让id从1开始，0可能会被做掩码
        feature_max_index_dict[feat] = data_df[feat].max() + 1

    norm = MinMaxScaler()
    data_df['occupation'] = norm.fit_transform(data_df[['occupation']])

    user_profile = data_df[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates('user_id')
    item_profile = data_df[["movie_id"]].drop_duplicates('movie_id')
    # 构建数据标签
    # 将数据按照时间进行排序, 默认是升序
    data_df.sort_values("timestamp", inplace=True)

    unique_item_ids = data_df['movie_id'].unique()

    train_data,test_data=train_test_split(data_df,train_size=0.8)

    train_data_list = []
    test_data_list = []
    # 设置最短历史序列长度
    min_seq_len = 1
    # 遍历每个用户构建正负样本
    for user_id, hist_df in train_data.groupby('user_id'):
        pos_list = hist_df['movie_id'].to_list()
        rating_list = hist_df['rating'].to_list()

        if neg_sample > 0:
            candidate_list = list(set(unique_item_ids) - set(pos_list))
        for i in range(len(pos_list)):
            train_data_list.append((user_id, pos_list[i], 1, rating_list[i]))
        for negi in range(neg_sample):
            train_data_list.append((user_id,random.choice(candidate_list)[0], 0, 0))


    random.shuffle(train_data_list)
    random.shuffle(test_data_list)

    # 将输入的特征转换成字典的形式
    train_data_dict = {}
    test_data_dict = {}

    # 构建训练集数据
    train_data_dict['user_id'] = np.array([line[0] for line in train_data_list])
    train_data_dict['movie_id'] = np.array([line[1] for line in train_data_list])
    train_data_dict['label'] = np.array([line[2] for line in train_data_list])
    train_data_dict['rating'] = np.array([line[3] for line in train_data_list])
    for key in ["gender", "age", "occupation", "zip"]:
        # 将样本中所有user_id对应的其他的特征都索引到
        tmp_list = []
        for i in range(len(train_data_dict['user_id'])):
            tmp_list.append((user_profile[user_profile['user_id'] ==train_data_dict['user_id'][i]][key].values[0]))
        train_data_dict[key]=np.array(tmp_list)


    # 构建测试集数据
    test_data_dict['user_id'] = np.array([line[0] for line in test_data_list])
    test_data_dict['movie_id'] = np.array([line[1] for line in test_data_list])
    test_data_dict['label'] = np.array([line[2] for line in test_data_list])
    test_data_dict['rating'] = np.array([line[3] for line in train_data_list])
    #
    for key in ["gender", "age", "occupation", "zip"]:
        # 将样本中所有user_id对应的其他的特征都索引到
        tmp_list = []
        for i in range(len(train_data_dict['user_id'])):
            tmp_list.append((user_profile[user_profile['user_id'] == train_data_dict['user_id'][i]][key].values[0]))
        test_data_dict[key] = np.array(tmp_list)

    return feature_max_index_dict, train_data_dict, test_data_dict


if __name__ == "__main__":
    data_path=r'../data/movielens(ml-1m).txt'

    feature_max_index_dict, train_data_dict, test_data_dict = process_data(data_path)

    train_label = train_data_dict['label']
    print(train_data_dict.keys())
    train_data_dict.pop("label")
    embedding_dim = 4
    user_feature_columns = [SparseFeat('user_id', feature_max_index_dict['user_id'], embedding_dim),
                            SparseFeat("gender", feature_max_index_dict['gender'], embedding_dim),
                            SparseFeat("age", feature_max_index_dict['age'], embedding_dim),
                            DenseFeat("occupation",1),
                            SparseFeat("zip", feature_max_index_dict['zip'], embedding_dim)
                            ]

    item_feature_columns = [SparseFeat('movie_id', feature_max_index_dict['movie_id'], embedding_dim)]
    print(user_feature_columns + item_feature_columns)

    model = DSSM(user_feature_columns, item_feature_columns)

    model.compile(optimizer="adam", loss='binary_crossentropy')

    model.fit(train_data_dict, train_label, batch_size=20, epochs=50, verbose=1, validation_split=0.1, )
    #
    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    train_user_model_input = [train_data_dict[fc.name] for fc in user_feature_columns]
    train_item_model_input = [train_data_dict[fc.name] for fc in item_feature_columns]

    user_embs = user_embedding_model.predict(train_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(train_item_model_input, batch_size=2 ** 12)

    index = faiss.IndexFlatL2(len(user_embs[0]))
    index.add(item_embs)


    distance,index_list=index.search(user_embs,k=3)

    print(index_list[0])
    print(item_embs[index_list[0]])
    print(user_embs[0])
    print(distance[0])






