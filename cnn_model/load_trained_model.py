from keras.models import load_model

from data.data import load_data

if __name__ == '__main__':

    train_data, test_data, train_labels, test_labels = load_data('./data/fer2013.csv')

    print 'Data loaded.'

    model = load_model('./cnn_model/trained_model.h5')

    print 'Model loaded.'

    score = model.evaluate(test_data, test_labels, verbose=0)

    print 'Model evaluated.'

    print score
