import matplotlib.pyplot as plt
import numpy as np
from trains import Task, Logger

import keras
from keras import metrics
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from allegro_board import *

import tempfile

img_height = 150
img_width = 150

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def CreateDataSet(foldername, augmentation, batchSize):
    image_gen = ImageDataGenerator(rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True) if augmentation else ImageDataGenerator(rescale=1. / 255)

    ds = image_gen.flow_from_directory(
        foldername,
        target_size=(img_height, img_width),
        batch_size=batchSize,
        class_mode='binary',
        shuffle=False)
    return ds

dataset_folder_path = 'MRI_CT_data'
train_ds = CreateDataSet(dataset_folder_path + '/train', True, 60)
valid_ds = CreateDataSet(dataset_folder_path + '/test', False, 20)
test_ds = CreateDataSet(dataset_folder_path + '/valid', False, 20)

def TrainModel(model, base_model, model_name):

    task = Task.init(project_name="Ex3ModelTrains", task_name=model_name)
    reporter = TrainsReporter()
    # Show a summary of the model. Check the number of trainable parameters
    model.summary()

    # Compile the model
    model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[metrics.BinaryAccuracy()])

    # Train the model
    model.fit(
        train_ds,
        steps_per_epoch=train_ds.samples / train_ds.batch_size,
        epochs=20,
        validation_data=valid_ds,
        validation_steps=valid_ds.samples / valid_ds.batch_size,
        callbacks=[reporter],
        verbose=1)

    # Unfreeze the base_model. Note that it keeps running in inference mode
    # since we passed `training=False` when calling it. This means that
    # the batchnorm layers will not update their batch statistics.
    # This prevents the batchnorm layers from undoing all the training
    # we've done so far.
    base_model.trainable = True
    reporter.epoch_ref = 20

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    reporter.TrainsReporter

    model.fit(
        train_ds,
        steps_per_epoch=train_ds.samples / train_ds.batch_size,
        epochs=10,
        validation_data=valid_ds,
        validation_steps=valid_ds.samples / valid_ds.batch_size,
        callbacks=[reporter],
        verbose=1)

    score = model.evaluate(test_ds)
    print('Test evaluation Score:', model.evaluate(test_ds))
    print('validation evaluation Score:', model.evaluate(valid_ds))

def CreateXceptionModel():
    base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = False

    inputs = keras.Input(shape=(150, 150, 3))

    # Pre-trained Xception weights requires that input be normalized
    # from (0, 255) to a range (-1., +1.), the normalization layer
    # does the following, outputs = (inputs - mean) / sqrt(var)
    norm_layer = keras.layers.experimental.preprocessing.Normalization()
    mean = np.array([127.5] * 3)
    var = mean ** 2
    # Scale inputs to [-1, +1]
    x = norm_layer(inputs)
    norm_layer.set_weights([mean, var])

    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    return model, base_model

m1, m2 = CreateXceptionModel()
TrainModel(m1, m2, "Xception")

