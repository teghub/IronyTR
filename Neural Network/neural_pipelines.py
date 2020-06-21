from keras.layers.embeddings import Embedding
from sklearn.model_selection import KFold
from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.models import load_model
from keras.models import Sequential
from keras import backend as K
from keras import regularizers
from keras.layers import Dense
from keras.layers import LSTM

import keras.metrics as metrics
import keras.layers as layers
import tensorflow as tf
import fasttext.util
import pandas as pd
import numpy as np
import fasttext
import keras
import os

np.random.seed(7)
fasttext.util.download_model('tr', if_exists='ignore')

def FindMean(values):
    return sum(values) / len(values)

def CreateSequentialModel(vocab_size, embedding_matrix, max_length, embed_size):
    model = Sequential()
    embeddingLayer = Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=max_length, trainable=True)
    model.add(embeddingLayer)
    model.add(LSTM(128))
    model.add(layers.Dense(100, activation='relu'))
    model.add(Dense(1,  activation='relu'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def CalculateMetrics(testLabels, predictions):
    falsePositives = 0.0
    falseNegatives = 0.0
    truePositives = 0.0
    trueNegatives = 0.0
    accuratePredictions = 0.0

    for i in range(testLabels.size):
        if testLabels[i] == 1 and predictions[i] == 0:
            falseNegatives += 1
        elif testLabels[i] == 0 and predictions[i] == 1:
            falsePositives += 1
        elif testLabels[i] == 1 and predictions[i] == 1:
            truePositives += 1
            accuratePredictions += 1
        elif testLabels[i] == 0 and predictions[i] == 0:
            trueNegatives += 1
            accuratePredictions += 1

    accuracy = accuratePredictions/predictions.size

    if truePositives == 0 and falsePositives == 0 and falseNegatives == 0:
        precision = 1.0
        recall = 1.0
        f1Score = 1.0
    elif truePositives == 0 and (falsePositives > 0 or falseNegatives > 0):
        precision = 0.0
        recall = 0.0
        f1Score = 0.0
    else:
        precision = truePositives/(truePositives + falsePositives)
        recall = truePositives/(truePositives + falseNegatives)
        f1Score = 2*(precision*recall)/(precision+recall)

    print("accurate: ", accuratePredictions, " true_pos: ", truePositives, " false_pos: ", falsePositives, 
          " true_neg: ", trueNegatives, " false_neg: ", falseNegatives)

    return [accuracy, precision, recall, f1Score]

def SetUpLSTM():
    numberOfEpochs = 5
    ironicSentences = np.load("ironic_600.npy")
    nonIronicSentences = np.load("non_ironic_600.npy")
    trainingSentences = np.concatenate((ironicSentences, nonIronicSentences))
    trainingLabels = np.concatenate((np.ones(ironicSentences.shape, dtype=int), np.zeros(nonIronicSentences.shape, dtype=int)))


    additionalFeaturesIronic = pd.read_csv("data-600/basic-features-ironic.txt", sep=',',header=None)
    additionalFeaturesNonIronic = pd.read_csv("data-600/basic-features-nonironic.txt", sep=',',header=None)
    additionalFeatures = pd.concat([additionalFeaturesIronic, additionalFeaturesNonIronic])

    sizeOfVocabulary = 200
    encodedSentences = [text.one_hot(d, sizeOfVocabulary) for d in trainingSentences]

    train = text.Tokenizer()
    train.fit_on_texts(trainingSentences)
    sizeOfVocabulary = len(train.word_index) + 1
    encodedSentences = train.texts_to_sequences(trainingSentences)

    maximumLengthOfSentence = 20
    paddedSentences = sequence.pad_sequences(encodedSentences, maxlen=maximumLengthOfSentence, padding='post')

    embeddingsIndex = dict()
    matrixSize = 0
    filePointer = open('cc.tr.300.vec')
    next(filePointer)

    for line in filePointer:
      values = line.split()
      word = values[0]
      coefficients = np.asarray(values[1:], dtype='float32')
      embeddingsIndex[word] = coefficients
      matrixSize = len(coefficients)

    filePointer.close()
    print('Loaded %s word vectors.' % len(embeddingsIndex))

    nonZeroEmbeddings = 0
    zeroEmbeddings = 0
    embeddingMatrix = np.zeros((sizeOfVocabulary, matrixSize))
    for word, i in train.word_index.items():
      embeddingVector = embeddingsIndex.get(word)
      if embeddingVector is not None:
        nonZeroEmbeddings += 1
        embeddingMatrix[i] = embeddingVector
      else:
        zeroEmbeddings += 1

    print("Non-zero embeddings: ", nonZeroEmbeddings, " Zero embeddings: ", zeroEmbeddings)

    numberOfSplits=10

    np.random.shuffle(trainingLabels)
    np.random.shuffle(paddedSentences)
    np.random.seed(7)

    folds = []

    for trainIndex,validationIndex in KFold(numberOfSplits).split(paddedSentences):

        trainData, validationData = paddedSentences[trainIndex],paddedSentences[validationIndex]
        additionalFeaturesTrain, additionalFeaturesValidation = additionalFeatures.values[trainIndex], additionalFeatures.values[validationIndex]
        trainLabels, validationLabels = trainingLabels[trainIndex],trainingLabels[validationIndex]

        embeddingLayerInput = keras.Input(shape = (maximumLengthOfSentence, ), name = 'embeddingInput')
        additionalFeaturesInput = keras.Input(shape=(21, ), name='additionalFeatures')
        embeddingLayer = Embedding(sizeOfVocabulary, matrixSize, weights=[embeddingMatrix], input_length=maximumLengthOfSentence, trainable=True)
        LSTMLayer = LSTM(128)
        denseLayer1 = Dense(100,  activation='relu')
        denseLayer2 = Dense(1,  activation='relu')
        
        temporaryOutput = embeddingLayer(embeddingLayerInput)
        lstmOutput = LSTMLayer(temporaryOutput)
        concatenatedOutput = layers.Concatenate()([lstmOutput, additionalFeaturesInput])
        denseOutput = denseLayer1(concatenatedOutput)
        output = denseLayer2(denseOutput)

        model = keras.Model(inputs = [embeddingLayerInput, additionalFeaturesInput], outputs = output)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        outputOfModel = model.fit({"embeddingInput": trainData, "additionalFeatures": additionalFeaturesTrain}, trainLabels, epochs = numberOfEpochs)
        predictions = model.predict({"embeddingInput": validationData, "additionalFeatures": additionalFeaturesValidation}).flatten()
        predictions = np.asarray([1 if elem >= 0.5 else 0 for elem in predictions])
        print(predictions, validationLabels)

        foldResult = CalculateMetrics(validationLabels, predictions)
        folds.append(foldResult)
        
    print("10-Fold Averages: ", *map(FindMean, zip(*folds)))

def SetUpBiLSTM():
    numberOfEpochs = 3
    ironicSentences = np.load("ironic_600.npy")
    nonIronicSentences = np.load("non_ironic_600.npy")
    trainingSentences = np.concatenate((ironicSentences, nonIronicSentences))
    trainingLabels = np.concatenate((np.ones(ironicSentences.shape, dtype=int), np.zeros(nonIronicSentences.shape, dtype=int)))


    additionalFeaturesIronic = pd.read_csv("data-600/basic-features-ironic.txt", sep=',',header=None)
    additionalFeaturesNonIronic = pd.read_csv("data-600/basic-features-nonironic.txt", sep=',',header=None)
    additionalFeatures = pd.concat([additionalFeaturesIronic, additionalFeaturesNonIronic])

    sizeOfVocabulary = 200
    encodedSentences = [text.one_hot(d, sizeOfVocabulary) for d in trainingSentences]

    train = text.Tokenizer()
    train.fit_on_texts(trainingSentences)
    sizeOfVocabulary = len(train.word_index) + 1
    encodedSentences = train.texts_to_sequences(trainingSentences)

    maximumLengthOfSentence = 20
    paddedSentences = sequence.pad_sequences(encodedSentences, maxlen=maximumLengthOfSentence, padding='post')

    embeddingsIndex = dict()
    matrixSize = 0
    filePointer = open('cc.tr.300.vec')
    next(filePointer)

    for line in filePointer:
      values = line.split()
      word = values[0]
      coefficients = np.asarray(values[1:], dtype='float32')
      embeddingsIndex[word] = coefficients
      matrixSize = len(coefficients)

    filePointer.close()
    print('Loaded %s word vectors.' % len(embeddingsIndex))

    nonZeroEmbeddings = 0
    zeroEmbeddings = 0
    embeddingMatrix = np.zeros((sizeOfVocabulary, matrixSize))
    for word, i in train.word_index.items():
      embeddingVector = embeddingsIndex.get(word)
      if embeddingVector is not None:
        nonZeroEmbeddings += 1
        embeddingMatrix[i] = embeddingVector
      else:
        zeroEmbeddings += 1

    print("Non-zero embeddings: ", nonZeroEmbeddings, " Zero embeddings: ", zeroEmbeddings)

    numberOfSplits=10

    np.random.shuffle(trainingLabels)
    np.random.shuffle(paddedSentences)
    np.random.seed(7)

    folds = []

    for trainIndex,validationIndex in KFold(numberOfSplits).split(paddedSentences):

        trainData, validationData = paddedSentences[trainIndex],paddedSentences[validationIndex]
        additionalFeaturesTrain, additionalFeaturesValidation = additionalFeatures.values[trainIndex], additionalFeatures.values[validationIndex]
        trainLabels, validationLabels = trainingLabels[trainIndex],trainingLabels[validationIndex]

        embeddingLayerInput = keras.Input(shape = (maximumLengthOfSentence, ), name = 'embeddingInput')
        additionalFeaturesInput = keras.Input(shape=(21, ), name='additionalFeatures')
        embeddingLayer = Embedding(sizeOfVocabulary, matrixSize, weights=[embeddingMatrix], input_length=maximumLengthOfSentence, trainable=True)
        LSTMLayer = layers.Bidirectional(LSTM(128))
        denseLayer1 = Dense(100,  activation='relu')
        denseLayer2 = Dense(1,  activation='relu')
        
        temporaryOutput = embeddingLayer(embeddingLayerInput)
        lstmOutput = LSTMLayer(temporaryOutput)
        concatenatedOutput = layers.Concatenate()([lstmOutput, additionalFeaturesInput])
        denseOutput = denseLayer1(concatenatedOutput)
        output = denseLayer2(denseOutput)

        model = keras.Model(inputs = [embeddingLayerInput, additionalFeaturesInput], outputs = output)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        outputOfModel = model.fit({"embeddingInput": trainData, "additionalFeatures": additionalFeaturesTrain}, trainLabels, epochs = numberOfEpochs)
        predictions = model.predict({"embeddingInput": validationData, "additionalFeatures": additionalFeaturesValidation}).flatten()
        predictions = np.asarray([1 if elem >= 0.5 else 0 for elem in predictions])
        print(predictions, validationLabels)

        foldResult = CalculateMetrics(validationLabels, predictions)
        folds.append(foldResult)
        
    print("10-Fold Averages: ", *map(FindMean, zip(*folds)))

def SetUpCnnLSTM():
    numberOfEpochs = 5
    ironicSentences = np.load("ironic_600.npy")
    nonIronicSentences = np.load("non_ironic_600.npy")
    trainingSentences = np.concatenate((ironicSentences, nonIronicSentences))
    trainingLabels = np.concatenate((np.ones(ironicSentences.shape, dtype=int), np.zeros(nonIronicSentences.shape, dtype=int)))


    additionalFeaturesIronic = pd.read_csv("data-600/basic-features-ironic.txt", sep=',',header=None)
    additionalFeaturesNonIronic = pd.read_csv("data-600/basic-features-nonironic.txt", sep=',',header=None)
    additionalFeatures = pd.concat([additionalFeaturesIronic, additionalFeaturesNonIronic])

    sizeOfVocabulary = 200
    encodedSentences = [text.one_hot(d, sizeOfVocabulary) for d in trainingSentences]

    train = text.Tokenizer()
    train.fit_on_texts(trainingSentences)
    sizeOfVocabulary = len(train.word_index) + 1
    encodedSentences = train.texts_to_sequences(trainingSentences)

    maximumLengthOfSentence = 20
    paddedSentences = sequence.pad_sequences(encodedSentences, maxlen=maximumLengthOfSentence, padding='post')

    embeddingsIndex = dict()
    matrixSize = 0
    filePointer = open('cc.tr.300.vec')
    next(filePointer)

    for line in filePointer:
      values = line.split()
      word = values[0]
      coefficients = np.asarray(values[1:], dtype='float32')
      embeddingsIndex[word] = coefficients
      matrixSize = len(coefficients)

    filePointer.close()
    print('Loaded %s word vectors.' % len(embeddingsIndex))

    nonZeroEmbeddings = 0
    zeroEmbeddings = 0
    embeddingMatrix = np.zeros((sizeOfVocabulary, matrixSize))
    for word, i in train.word_index.items():
      embeddingVector = embeddingsIndex.get(word)
      if embeddingVector is not None:
        nonZeroEmbeddings += 1
        embeddingMatrix[i] = embeddingVector
      else:
        zeroEmbeddings += 1

    print("Non-zero embeddings: ", nonZeroEmbeddings, " Zero embeddings: ", zeroEmbeddings)

    numberOfSplits=10

    np.random.shuffle(trainingLabels)
    np.random.shuffle(paddedSentences)
    np.random.seed(7)

    folds = []

    for trainIndex,validationIndex in KFold(numberOfSplits).split(paddedSentences):

        trainData, validationData = paddedSentences[trainIndex],paddedSentences[validationIndex]
        additionalFeaturesTrain, additionalFeaturesValidation = additionalFeatures.values[trainIndex], additionalFeatures.values[validationIndex]
        trainLabels, validationLabels = trainingLabels[trainIndex],trainingLabels[validationIndex]

        embeddingLayerInput = keras.Input(shape = (maximumLengthOfSentence, ), name = 'embeddingInput')
        additionalFeaturesInput = keras.Input(shape=(21, ), name='additionalFeatures')
        embeddingLayer = Embedding(sizeOfVocabulary, matrixSize, weights=[embeddingMatrix], input_length=maximumLengthOfSentence, trainable=True)
        convolutionalLayer = layers.Conv1D(64, 5, strides=1, padding='valid')
        LSTMLayer = LSTM(1*6*1*64)
        denseLayer1 = Dense(100,  activation='relu')
        denseLayer2 = Dense(1,  activation='relu')
        
        temporaryOutput = embeddingLayer(embeddingLayerInput)
        lstmOutput = LSTMLayer(temporaryOutput)
        concatenatedOutput = layers.Concatenate()([lstmOutput, additionalFeaturesInput])
        denseOutput = denseLayer1(concatenatedOutput)
        output = denseLayer2(denseOutput)

        model = keras.Model(inputs = [embeddingLayerInput, additionalFeaturesInput], outputs = output)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        outputOfModel = model.fit({"embeddingInput": trainData, "additionalFeatures": additionalFeaturesTrain}, trainLabels, epochs = numberOfEpochs)
        predictions = model.predict({"embeddingInput": validationData, "additionalFeatures": additionalFeaturesValidation}).flatten()
        predictions = np.asarray([1 if elem >= 0.5 else 0 for elem in predictions])
        print(predictions, validationLabels)

        foldResult = CalculateMetrics(validationLabels, predictions)
        folds.append(foldResult)
        
    print("10-Fold Averages: ", *map(FindMean, zip(*folds)))

if __name__ == "__main__":
    SetUpCnnLSTM()
    
