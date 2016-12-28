from keras.callbacks import ModelCheckpoint
from keras.layers import Convolution2D, Activation, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential

from data.data import load_data

if __name__ == '__main__':

    train_data, test_data, train_labels, test_labels = load_data('./data/fer2013.csv')

    model = Sequential()

    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(1, 48, 48)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(512, 3, 3))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))
    model.add(Dense(7))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    model.fit(train_data, train_labels, batch_size=64, nb_epoch=50, verbose=1,
              validation_data=(test_data, test_labels),
              callbacks=[ModelCheckpoint('./cnn_model/trained_deep_model.h5', monitor='val_loss', verbose=1,
                                         save_best_only=True, save_weights_only=False, mode='auto')])
