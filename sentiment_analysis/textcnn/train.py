# coding: utf8

import sys
reload(sys)
sys.setdefaultencoding("utf8")
import paddle.fluid as fluid

from sentiment_analysis.dataset.kaggle import KaggleDataset
from sentiment_analysis.textcnn.network import cnn_net
from sentiment_analysis.textcnn.preprocess import prepare_data

def train(reader, word_dim, network, pass_num):
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)
    label = fluid.layers.data(
        name="label", shape=[1], dtype="int64")

    cost, acc, prediction = network(data, label, word_dim)

    optimizer = fluid.optimizer.Adagrad()
    optimizer.minimize(cost)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list = [data, label], place = place)

    exe.run(fluid.default_startup_program())

    for pass_id in xrange(pass_num):
        data_size, data_count, total_acc, total_cost = 0, 0, 0., 0.

        for data in reader():
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                              feed = feeder.feed(data),
                                              fetch_list=[cost, acc])

            data_size = len(data)

            total_acc += data_size * avg_acc_np
            total_cost += data_size * avg_cost_np
            data_count += data_size

        avg_cost = total_cost / data_count
        avg_acc = total_acc / data_count

        print "pass_id: %d, avg_acc: %f, avg_cost: %f" % (pass_id, avg_acc, avg_cost)

if __name__ == "__main__":
    # 加载数据集
    print >> sys.stderr, "load dataset"
    dataset = KaggleDataset("../../kaggle_data_test")
    print >> sys.stderr, "prepare data"
    train_reader = prepare_data(dataset, 128)

    pass_num = 10
    print >> sys.stderr, "start train"
    train(train_reader, len(dataset["word_dict"]), cnn_net, pass_num)