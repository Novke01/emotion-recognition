from keras.models import load_model
import numpy as np

from data.data import load_data


if __name__ == '__main__':

    train_data, test_data, train_labels, test_labels = load_data('./data/fer2013.csv')

    print 'Data loaded.'

    model = load_model('./cnn_model/trained_deep_model.h5')

    print 'Model loaded.'

    # score = model.evaluate(test_data, test_labels, verbose=0)
    #
    # print 'Model evaluated.'
    #
    # print score

    size = 32

    result = model.predict_classes(test_data, verbose=0)

    print result

    real_labels = []
    for l in test_labels:
        real_labels.append(np.where(l == 1.)[0][0])

    real_labels = np.array(real_labels)

    counts = [0, 0, 0, 0, 0, 0, 0]

    sizes = [0, 0, 0, 0, 0, 0, 0]

    n = len(real_labels)

    for index in xrange(n):
        if result[index] == real_labels[index]:
            counts[real_labels[index]] += 1
        sizes[real_labels[index]] += 1

    for i in xrange(7):
        print str(counts[i] / float(sizes[i])) + '%'
