import os
import pandas as pd
import numpy as np
from mobilenet import MobileNet
from mobilenet_dih import MobileNetDih4
from mobilenet_dih_r import MobileNetDR
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras import metrics
from keras import losses
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import imageio
from skimage.transform import resize as imgresize

TRAIN_PATH = 'input/train.json'
BATCH_SIZE = 32

def json2img_and_labels(df:pd.DataFrame):
    imgs = []
    y = []

    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75,75)
        band_2 = np.array(row['band_2']).reshape(75,75)

        d1 = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        d2 = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())

        imgs.append(np.dstack([d1, d2]))
        y.append(row['is_iceberg'])

    return imgs, y

def get_data():
    print('Read Data')
    df = pd.read_json(TRAIN_PATH)
    imgs, y = json2img_and_labels(df)
    train_img, valid_img, train_y, valid_y = train_test_split(imgs,
                                                              y,
                                                              random_state=131,
                                                              shuffle=True,
                                                              stratify=y,
                                                              train_size=0.75)
    return train_img, valid_img, train_y, valid_y


def get_callbacks(filepath, patience=1):
    mcp = ModelCheckpoint(filepath,
                          monitor='val_loss',
                          verbose=2,
                          save_best_only=True,
                          save_weights_only=False,
                          mode='min',
                          period=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=patience, min_lr=1e-16, verbose=1)
    csv_log = CSVLogger(filename=filepath+'.csv')
    return [mcp, rlr, csv_log]

def training_model(model_name='mobilenet'):
    train_img, valid_img, train_y, valid_y = get_data()
    callbacks = get_callbacks('mobilenet_10fulld01_b16', patience=2)
    if model_name == 'mobilenet':
        print('MobileNet')
        model = MobileNet(alpha=1.)
        model.summary()
    elif model_name =='mobilenet_dih':
        print('MobileNetDih')
        model = MobileNetDih4(alpha=1.)
        model.summary()
    elif model_name =='mobilenet_dih_r':
        print('MobileNetDihR')
        model = MobileNetDR(alpha=1.)
        model.summary()

    opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    #opt = Adadelta(lr=1e-1, rho=0.95, decay=0.1)
    #opt = SGD(lr=1e-7, momentum=0.9, decay=0., nesterov=True)

    model.compile(optimizer=opt,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    #model.load_weights('mobilenet_05shortd01_catcros_resize_b16.hdf5')
    gen = ImageDataGenerator(rotation_range=359,
                             zoom_range=[0.5, 2],
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             vertical_flip=True,
                             horizontal_flip=True)

    model.fit_generator(gen.flow(np.array(train_img),
                            np.array(train_y),
                            batch_size=BATCH_SIZE),
                        steps_per_epoch=16*len(train_y)//BATCH_SIZE,
                        epochs=40,
                        validation_data=[np.array(valid_img), np.array(valid_y)],
                        verbose=1,
                        callbacks=callbacks)
#    """
    #opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    #opt = Adadelta(lr=1e-1, rho=0.95, decay=0.1)
    opt = SGD(lr=0.05, momentum=0.9, decay=0., nesterov=True)
    model.load_weights('mobilenet_10shortd01_b16_sgd')
    model.fit_generator(gen.flow(np.array(train_img),
                            np.array(train_y),
                            batch_size=BATCH_SIZE),
                        steps_per_epoch=16*len(train_y)//BATCH_SIZE,
                        epochs=10,
                        validation_data=[np.array(valid_img), np.array(valid_y)],
                        verbose=1,
                        callbacks=callbacks)
#"""
