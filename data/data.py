from csv import DictReader

import numpy as np
from keras.utils import np_utils


def load_data(path):

    with open(path, 'r') as face_emotion_data:

        filed_names = ['emotion', 'pixels', 'Usage']
        reader = DictReader(face_emotion_data, fieldnames=filed_names)
        reader.next()

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        for row in reader:
            row['pixels'] = [float(pixel) / 255 for pixel in row['pixels'].split()]
            image = np.array(row['pixels'], dtype='float32').reshape((1, 48, 48))
            if row['Usage'] == 'Training':
                train_data.append(image)
                train_labels.append(int(row['emotion']))
            else:
                test_data.append(image)
                test_labels.append(int(row['emotion']))

    train_data = np.array(train_data)
    train_labels = np_utils.to_categorical(train_labels, 7)
    test_data = np.array(test_data)
    test_labels = np_utils.to_categorical(test_labels, 7)

    return train_data, test_data, train_labels, test_labels
