# coding: utf8
import paddle

def prepare_data(dataset,
                 batch_size):
    def train_reader_fn():
        for i in range(len(dataset["train"]["reviews"])):
            review = dataset["train"]["review"][i]
            label = dataset["train"]["label"][i]

            doc = []
            for word in review:
                doc.append(dataset["word_dict"][word])
            yield doc, label

    train_reader = paddle.batch(train_reader_fn, batch_size=batch_size)

    return train_reader









