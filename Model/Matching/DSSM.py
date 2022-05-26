from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten
from DLFeatureFactory.Features import  FeatureEncoder
from Layer.cores import  DNN,CosinSimilarity,PredictLayer
import tensorflow as tf

class NoMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask):
        return None

def concat_func(inputs, axis=-1, mask=False):
    if not mask:
        inputs = list(map(NoMask(), inputs))
    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)


def combined_dnn_input(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_func(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_func(dense_value_list))
        return concat_func([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_func(sparse_embedding_list))
    elif len(dense_value_list) > 0:
        return Flatten()(concat_func(dense_value_list))
    else:
        raise NotImplementedError("dnn_feature_columns can not be empty list")


def process_feature(feature_columns, feature_encode):

    group_embedding_dict = feature_encode.sparse_feature_dict
    dense_feature_dict = feature_encode.dense_feature_dict

    feature_names=[fc.name for fc in feature_columns]

    sparse_dnn_input = [v for k, v in group_embedding_dict['default_group'].items() if k in feature_names]
    dense_dnn_input = [v for k, v in dense_feature_dict.items() if k in feature_names]

    return sparse_dnn_input, dense_dnn_input


def DSSM(user_feature_columns, item_feature_columns, dnn_units=[64, 32],temp=10, task='binary'):
    # 构建所有特征的Input层和Embedding层
    feature_encode = FeatureEncoder(user_feature_columns + item_feature_columns)
    feature_input_layers_list = list(feature_encode.feature_input_layer_dict.values())

    # 特征处理
    user_sparse_dnn_input, user_dense_dnn_input = process_feature(user_feature_columns, feature_encode)
    user_dnn_input=combined_dnn_input(user_sparse_dnn_input,user_dense_dnn_input)

    item_sparse_dnn_input, item_dense_dnn_input = process_feature(item_feature_columns, feature_encode)
    item_dnn_input = combined_dnn_input(item_sparse_dnn_input, item_dense_dnn_input)



    user_dnn_input = Flatten()(user_dnn_input)
    item_dnn_input = Flatten()(item_dnn_input)

    user_dnn_out = DNN(dnn_units)(user_dnn_input)
    item_dnn_out = DNN(dnn_units)(item_dnn_input)

    # 计算相似度
    scores = CosinSimilarity(temp)([user_dnn_out, item_dnn_out]) # (B,1)

    # 确定拟合目标
    output = PredictLayer()(scores)

    # 根据输入输出构建模型
    model = Model(feature_input_layers_list, output)
    # model.summary()

    return model