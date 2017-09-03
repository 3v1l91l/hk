import pandas as pd
from scipy import misc
import numpy as np
import deepneuralnet as net
from tflearn.data_utils import load_csv
from tflearn.data_utils import to_categorical


TEST_IMG_DIR = "test_img"
TRAIN_IMG_DIR = "train_img"

def load_data():
    # train, train_labels = load_csv('train.csv', target_column=1,
    #                         categorical_labels=True, n_classes=26)
    # train, _ = load_csv('train.csv', target_column=1,
    #                          categorical_labels=True, n_classes=1)
    number_categories = 26
    train = pd.read_csv("train.csv", sep=',', index_col=0)[0:20]
    test = pd.read_csv("test.csv", sep=',', index_col=0)[0:20]
    train_images_list= []
    to_categorical(train['label'], nb_classes=26)
    for row in train.itertuples():
        img = misc.imread(TRAIN_IMG_DIR + '/' + row.Index + '.png', flatten=True)
        train_images_list.append(img)

    test_images_list= []
    for row in test.itertuples():
        img = misc.imread(TEST_IMG_DIR + '/' + row.Index + '.png', flatten=True)
        test_images_list.append(img)

    return train, train_images_list, test, test_images_list

    #     df = pd.read_csv(file_, index_col=None, header=0)
    #     list_.append(df)
    # frame = pd.concat(list_)

    # misc.imread
    # return Cov


def main():
    train, train_images_list, test, test_images_list = load_data()
    # model = net.model
    # train_images_list = np.array(train_images_list).reshape(-1, 256, 256, 1)



    # model.fit(train_images_list, np.array(train['label']), n_epoch=15, validation_set=(train_images_list, np.array(train['label'])), show_metric=True, run_id="deep_nn")
    # model.save('final-model.tflearn')

    print('done')

if __name__ == "__main__":
    main()