import librosa as lb
import soundfile as sf
import numpy as np
import pickle
import pandas as pd


def read_audio(filename):
    """This function converts the .wav file found in filename and converts it into a mono representation numpy array
    with 22050 sampling rate. The function returns the numpy array, the sampling rate of the array, and the original
    file's sampling rate in that order."""

    x, sr = sf.read(filename)
    x = x.T  # transpose to match format of librosa array representation

    # set sample rate at 22050 to reduce memory usage.
    x = lb.resample(x, sr, 22050)

    # change any stereo audio to mono.
    if x.shape[0] == 2:
        x = lb.to_mono(x)

    return x, 22050, sr


def get_features(sound, sr=22050):
    """This function returns temporally averaged features for a given array. The features are mel-scaled spectrograms,
    MFCCs, chroma features, and tonnetz features"""

    mfccs = np.mean(lb.feature.mfcc(y=sound, sr=sr, n_mfcc=40).T, axis=0) #mel frequency cepstral coefficient
    mel = np.mean(lb.feature.melspectrogram(y=sound, sr=sr, n_mels=128).T, axis=0) #melspectrogram
    stft = np.abs(lb.stft(sound)) #short time fourier transform
    chroma = np.mean(lb.feature.chroma_stft(S=stft, sr=sr).T, axis=0) #chroma features
    tonnetz = np.mean(lb.feature.tonnetz(y=lb.effects.harmonic(sound), sr=sr).T, axis=0) #tonnetz features

    return mel, mfccs, chroma, tonnetz


def write_features():
    """ This function calculates the features using get_features for all the sounds in the UrbanNoises8k Dataset. If
    any of the features cannot be calculated, the function returns a np.nan vectors for the features of the sound."""
    key = pd.read_csv('Data/UrbanSound8K.csv')

    melSpectro = np.zeros((len(key),128))
    MFCC = np.zeros((len(key),40))
    ChromaFeatures = np.zeros((len(key), 12))
    tonnetzFeatures = np.zeros((len(key), 6))

    for i in range(len(key)):
        fold = key.iloc[i, 5]
        name = key.iloc[i, 0]
        filepath = '../Cap1/AllData/UrbanSound8K/audio/fold' + str(fold) + '/' + name
        x, sr, _ = read_audio(filepath)

        try:
            mel, mfccs, chroma, tonnetz = get_features(x)

        except:

            print('Error at sample {}'.format(i))
            mel = np.nan * np.ones((1, 128))
            mfccs = np.nan * np.ones((1, 40))
            chroma = np.nan * np.ones((1, 12))
            tonnetz = np.nan * np.ones((1, 6))

        melSpectro[i, :] = mel
        MFCC[i, :] = mfccs
        ChromaFeatures[i, :] = chroma
        tonnetzFeatures[i, :] = tonnetz

    with open('Data/pickles/MelFeatures.pickle', 'wb') as file:
        pickle.dump(melSpectro, file)

    with open('Data/pickles/MFCCFeatures.pickle', 'wb') as file:
        pickle.dump(MFCC, file)

    with open('Data/pickles/ChromaFeatures.pickle', 'wb') as file:
        pickle.dump(ChromaFeatures, file)

    with open('Data/pickles/tonnetzFeatures.pickle', 'wb') as file:
        pickle.dump(tonnetzFeatures, file)

    print('Feature Extraction Complete')

    return 1


def plot_confusion_matrix(y_test, y_pred,labels):

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    ConfMatrix = confusion_matrix(y_test, y_pred, labels=labels)

    import itertools

    def plot_confusion_matrix(cm, classes,
                              normalize=True,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

    fig = plt.figure(figsize=(10, 6))
    plot_confusion_matrix(ConfMatrix, classes=labels, title='Normalized Confusion matrix')
    plt.xticks(rotation='vertical')
    return fig
