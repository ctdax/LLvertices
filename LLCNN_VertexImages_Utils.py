import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from statistics import mean
from LLcommonFunctions import returnTestSamplesSplitIntoSignalAndBackground, compareManyHistograms, returnBestCutValue
from zipfile import ZipFile
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from keras.preprocessing.image import ImageDataGenerator
import re
import numpy as np
import time
import os, sys
import math as m
import random
import shutil
import splitfolders

def train_test_val_split(filepath, testing_fraction): #Creates training and testing folders for signal and background to the filepath
    print("Creating Training and Testing folders for the data")
    # Make sure we remove any existing folders and start from a clean slate
    shutil.rmtree(filepath+'/train/signal/', ignore_errors=True)
    shutil.rmtree(filepath+'/train/background/', ignore_errors=True)
    shutil.rmtree(filepath+'/test/signal/', ignore_errors=True)
    shutil.rmtree(filepath+'/test/background/', ignore_errors=True)
    shutil.rmtree(filepath + '/val/signal/', ignore_errors=True)
    shutil.rmtree(filepath + '/val/background/', ignore_errors=True)

    # Get the number of signal and background images
    _, _, sig_images = next(os.walk(filepath+'/signal/'))
    num_sig_images = len(sig_images)
    _, _, bkg_images = next(os.walk(filepath+'/background/'))
    num_bkg_images = len(bkg_images)
    if num_sig_images >= num_bkg_images:
        n = num_bkg_images
    else:
        n = num_sig_images

    trainN = int(n * (1 - testing_fraction) * (1 - testing_fraction))
    testN = int(n * testing_fraction)
    valN = int(n * (1 - testing_fraction) * testing_fraction)

    try:
        print("Working on splitting {} files".format(n*2))
        splitfolders.fixed(filepath, output=filepath, fixed=(trainN, valN, testN), move=False)
    except Exception:
        pass


def preprocess(imagefilepath, testing_fraction, dim, batch_size, splitfolders=True): #Preprocess the image data
    if splitfolders:
        train_test_val_split(imagefilepath, testing_fraction) #Creates training and testing folders for signal and background to the filepath
    trainfilepath = imagefilepath + "/train"
    testfilepath = imagefilepath + "/test"
    valfilepath = imagefilepath + "/val"

    image_generator = ImageDataGenerator()
    train = image_generator.flow_from_directory(trainfilepath, target_size=(dim, dim), batch_size=batch_size, class_mode='binary', shuffle=True)
    test = image_generator.flow_from_directory(testfilepath, target_size=(dim, dim), batch_size=batch_size, class_mode='binary', shuffle=True)
    val = image_generator.flow_from_directory(valfilepath, target_size=(dim, dim), batch_size=batch_size, class_mode='binary', shuffle=True)
    return train, test, val


def SplitPredictions(model, test_data, label_test): #Splits prediction data into signal and background for significance calculations, assumes np.ndarray
    signal = []
    background = []
    pred = model.predict(test_data).flatten()
    for i in range(len(pred)):
        if label_test[i] == 1:
            signal.append(pred[i])
        else:
            background.append(pred[i])

    return signal, background


def significance(model, test_data, label_test, testing_fraction, sig_nEvents, bkg_nEvents, minBackground=500,
                 logarithmic=False):
    pred_signal, pred_background = SplitPredictions(model, test_data, label_test)

    # Plot significance histograms
    _nBins = 40
    predictionResults = {'signal_pred': pred_signal, 'background_pred': pred_background}
    compareManyHistograms(predictionResults, ['signal_pred', 'background_pred'], 2, 'Signal Prediction', 'CNN Score',
                          0, 1,
                          _nBins, _normed=True, _testingFraction=testing_fraction, logarithmic=logarithmic)
    # Show significance
    returnBestCutValue('CNN', pred_signal.copy(), pred_background.copy(), _minBackground=minBackground,
                       _testingFraction=testing_fraction, ll_nEventsGen=int(10e4 * (sig_nEvents / 62642)),
                       qcd_nEventsGen=int(660740 * (bkg_nEvents / 138509)))


def accuracy(model, data_train, label_train, data_test, label_test):
    trainscores_raw = []
    testscores_raw = []
    x = 0
    while x <= 4:
        trainscores_raw.append(model.evaluate(data_train, label_train)[1] * 100)
        testscores_raw.append(model.evaluate(data_test, label_test)[1] * 100)
        x += 1
    trainscores = ("Training Accuracy: %.2f%%\n" % (mean(trainscores_raw)))
    testscores = ("Testing Accuracy: %.2f%%\n" % (mean(testscores_raw)))

    return trainscores, testscores


def epoch_history(history):
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('CNN Accuracy')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN Model Loss')
    plt.ylabel('Loss [A.U.]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def roc_plot(model, data_test, label_test):  # Plot roc curve with auc score
    predictions = model.predict(data_test)
    fpr, tpr, threshold = roc_curve(label_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()