# coding: utf8
"""
文本分类网络
"""
import paddle
import paddle.fluid as fluid

def cnn_net(data,
            label,
            dict_dim,
            emb_dim = 128,
            hid_dim = 128,
            hid_dim2 = 96,
            class_dim = 2,
            win_size = 3):
    emb = fluid.layers.embedding(input = data,
                                 size = [dict_dim, emb_dim])

    conv1 = fluid.nets.sequence_conv_pool(input = emb,
                                          num_filters = hid_dim,
                                          filter_size = win_size,
                                          act = "tanh",
                                          pool_type = "max")

    fc1 = fluid.layers.fc(input = [conv1],
                          size = hid_dim2)

    # 预测输出
    prediction = fluid.layers.fc(input = [fc1], size = class_dim, act = "softmax")
    # loss
    cost = fluid.layers.cross_entropy(input = prediction, label = label)
    # avg_loss
    avg_cost = fluid.layers.mean(x = cost)
    # accuracy
    acc = fluid.layers.accuracy(input = prediction, label = label)

    return avg_cost, acc, prediction
