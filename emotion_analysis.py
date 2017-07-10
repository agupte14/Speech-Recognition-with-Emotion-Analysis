#importing the required packages
import wave
import numpy as np
import speech_recognition as sr
import gensim
import os
import sklearn.preprocessing as preprocess
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
# import keras as kr
__authors__ = "Akshay Gupte, Leela Krishna Raavi, Shomron Jacob"


#constants
number_of_features= 2500
word_vector_size = 50

#word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
# # model.save("google")
# model = gensim.models.Word2Vec.load("google")

#-------------------------models--------------------------------
def LogReg(X_train,X_test,y_train,y_test):
    f1 = []
    P = ['l1', 'l2']
    C = [1, 0.01, 0.001, 0.0001]
    for p in P:
        for c in C:
            clf = LogisticRegression(penalty=p, C=c)
            score = cross_val_score(clf, X_train, y_train, scoring='accuracy',
                                    cv=10).mean()
            f1.append((p, c, score))
            print('Penality: %s\tC : %0.5f \tScore:%0.5f' % (p, c, score))
    name = 'Logistic Regression'
    best_case = max(f1,key=lambda y:y[2])
    return best_case[2], name, LogisticRegression(penalty=best_case[0],C=best_case[1])

def NaiveB(X_train,X_test,y_train,y_test):
    # f1 = []
    clf = GaussianNB()
    clf.fit(X_train,y_train)
    name = 'Naive Bayes'
    # accuracy = clf.score(X_test,y_test)
    f1 = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10)
    print(name,f1)
    return f1, name, clf

def ANN(X_train,X_test,y_train,y_test):
    f1 = []
    activation_functions = ['logistic', 'tanh', 'relu']
    learning_rate_initial = [0.2, 0.1, 0.01, 0.001]
    alpha = [0.01, 0.001, 0.0001, 0.00001]
    hidden_layers = [1, 2, 3]
    neurons_in_a_layer = [5, 10, 15, 20, 25]
    clf = MLPClassifier(solver='sgd',random_state=0)
    print("Artificial neural network")
    for a in alpha:
        for lr in learning_rate_initial:
            for fun in activation_functions:
                for layers in hidden_layers:
                    for neurons in neurons_in_a_layer:
                        size = []
                        for i in range(layers):
                            size.append(neurons)
                        size = tuple(size)
                        clf.set_params(activation=fun, learning_rate_init=lr, hidden_layer_sizes=size,alpha=a)
                        score = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10).mean()
                        f1.append((score, size, fun, lr, a))
                        print(score,size,fun,lr,a)
    name = 'Neural Network'
    best_case = max(f1,key=lambda y:y[0])
    return best_case[0], name, clf.set_params(activation=best_case[2], learning_rate_init=best_case[3], hidden_layer_sizes=best_case[1],alpha=best_case[4])

def SupoortVM(X_train,X_test,y_train,y_test):
    clf = svm.SVC()
    clf.fit(X_train,y_train)
    name = 'Support Vector Method'
    # accuracy = clf.score(X_test,y_test)
    f1 = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=10)
    print(f1,name)
    return f1, name, clf

def DecTree(X_train,X_test,y_train,y_test):
    Criterions = ['gini', 'entropy']
    Max_features = ['auto', 'sqrt', 'log2', 'None']
    Max_depths = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
    f1 = []
    print("Decision Tree")
    for crit in Criterions:
        for max_feat in Max_features:
            for depth in Max_depths:
                if max_feat is not "None":
                    clf = DecisionTreeClassifier(criterion=crit, max_features=max_feat, max_depth=depth,
                                                    random_state=0)
                else:
                    clf = DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=0)
                scoreDT = cross_val_score(clf, X_train, y_train, scoring='accuracy',
                                          cv=10).mean()
                f1.append([scoreDT, max_feat, depth, crit])
                print(scoreDT, max_feat, depth, crit)
    name = 'Decision Tree'
    bestcase = max(f1,key=lambda y:y[0])
    if bestcase[1] is not "None":
        return  bestcase[0], name, DecisionTreeClassifier(criterion=bestcase[3], max_features=bestcase[1], max_depth=bestcase[2],
                                                    random_state=0)
    else:
        return bestcase[0], name, DecisionTreeClassifier(criterion=bestcase[3], max_depth=bestcase[2], random_state=0)



# The raw data of the audio clip is processed by the class voice_analysis. The audio clip
# is provided in the form of .wav file. Audio clip can also be provided using microphone (planning).

#size of the chunck to be read at once
CHUNCK = 2**10

class audio_analysis():

    def __init__(self,filename,remaining_size):
        self.filename = filename
        self.waveData = np.array([])
        self.remaining_size = remaining_size

    # Returns the maximum amplitude of the wave, signifying the maximum engry of the
    # wave at a particular time.
    def extract_amplitude(self,waveData):
        return np.max(waveData)

    # Fast Fourier Transformation is used to get the frequency spectrum of the wave
    # number of the frequency
    def extract_frequency_spectrum(self,waveData):
        complex_array = np.fft.rfft(waveData,n=self.remaining_size-2)
        return np.absolute(complex_array)


    # The method extracts the voice features from the wave data by calling
    def extract_audio_features(self):
        wave_object = wave.open("Data/"+self.filename,'rb')
        count = 1
        data = np.abs(np.fromstring(wave_object.readframes(CHUNCK),dtype=np.int16))

        while True:
            if data.shape[0] == 0:
                break
            data = np.abs(np.fromstring(wave_object.readframes(CHUNCK),dtype=np.int16))

            self.waveData = np.concatenate([self.waveData,data])
            count +=1
        #calling the required function for each feature
        amplitude = self.extract_amplitude(self.waveData)
        frequency_spectrum = self.extract_frequency_spectrum(self.waveData)
        duration = count

        return np.concatenate((frequency_spectrum,np.array([amplitude,duration])))


# The class trains a model using the RNN sequential model and
# predicts the emotion of the speech
class emotion_analysis():

    # define anything that is neccessary while creating the object
    def __init__(self):
        pass


    # The method extract the features of from the raw that is from
    # the audio file
    #
    # filename: name of the .wav file
    #
    # returns: an np array consisting features
    def extract_features(self,filename):



        words = np.array([])

        # use the audio file as the audio source
        r = sr.Recognizer()
        with sr.AudioFile("Data/"+filename) as source:
            audio = r.record(source)  # read the entire audio file

        # recognize speech using Sphinx
        try:
            # print("Sphinx thinks you said " + r.recognize_sphinx(audio))
            words = r.recognize_sphinx(audio).split(" ")
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")
        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))

        audio_features = audio_analysis(filename,number_of_features-(len(words)*word_vector_size)).extract_audio_features()

        sentence_features = np.array([])
        features = np.array([])
        # word 2 vec
        for word in words:
            try:
                vect=model.wv[word]
            except:
                vect = np.zeros(50)
            features = np.concatenate([features,vect])
            sentence_features = np.concatenate([sentence_features, vect])

        for _ in range(len(words),51):
            sentence_features = np.concatenate([sentence_features, np.zeros(50)])

        remaining = number_of_features-len(audio_features)-len(features)
        # print(features.shape,audio_features.shape,remaining)
        temp =[]
        for _ in range(remaining):
            temp.append(0)

        features = np.concatenate([features,temp])
        # print(features.shape)

        return np.concatenate([np.array(features),audio_features]),sentence_features


    # The fit function fits the training data given in the form of wave files
    #
    # X: np array of the audio file names
    # Y: np array of true labels (ex. ['angry','sad','happy']) of the audio files
    #
    # returns nothing
    def fit(self,X):
        #for each file extract features using extract_features function defined above
        enc = preprocess.LabelEncoder()
        binarize = preprocess.Binarizer(threshold=0, copy=True)

        X_train_raw = []
        X_train_without_audio = []
        Y_train_raw = []
        for filename in X:
            features = self.extract_features(filename)
            X_train_raw.append(features[0])
            X_train_without_audio.append(features[1])
            Y_train_raw.append(filename.split("_")[0].upper())

        X_train_raw = np.array(X_train_raw)
        # print(len(X_train_raw),len(Y_train_raw))
        y_numeric = np.array(enc.fit_transform(Y_train_raw))
        X = X_train_raw
        # print(X.shape, y.shape)
        # y_binarized = np.array(binarize.fit_transform(Y_train_raw)).flatten()

        f1 = []
        name = []
        models = []
        performance_without_audio = []
        X_train, X_test, y_train_numeric, y_test_numeric = train_test_split(X, y_numeric, test_size=0.33, random_state=42)
        # X_train, X_test, y_train_binarized, y_test_binarized = train_test_split(X, y_binarized, test_size=0.33, random_state=42)
        X_train_WA, X_test_WA, y_train_numeric_WA, y_test_numeric_WA = train_test_split(np.array(X_train_without_audio), y_numeric, test_size=0.33, random_state=42)
        # X_train_WA, X_test_WA, y_train_binarized_WA, y_test_binarized_WA = train_test_split(np.array(X_train_without_audio), y_binarized, test_size=0.33, random_state=42)

        # f, nam, model = LogReg(X_train, X_test, y_train_binarized, y_test_binarized)
        # performance_without_audio.append(LogReg(X_train_WA, X_test_WA, y_train_binarized_WA, y_test_binarized_WA))
        # f1.append(f)
        # name.append(nam)
        # models.append(model)
        # Display(accuracy,f1,name)

        f, nam,model = NaiveB(X_train, X_test, y_train_numeric, y_test_numeric)
        performance_without_audio.append(NaiveB(X_train_WA, X_test_WA, y_train_numeric_WA, y_test_numeric_WA))
        f1.append(f)
        name.append(nam)
        models.append(model)
        # # Display(accuracy, f1, name)

        f, nam,model = ANN(X_train, X_test, y_train_numeric, y_test_numeric)
        performance_without_audio.append(ANN(X_train_WA, X_test_WA, y_train_numeric_WA, y_test_numeric_WA))
        f1.append(f)
        name.append(nam)
        models.append(model)
        # Display(accuracy, f1, name)

        f, nam,model = DecTree(X_train, X_test, y_train_numeric, y_test_numeric)
        performance_without_audio.append(DecTree(X_train_WA, X_test_WA, y_train_numeric_WA, y_test_numeric_WA))
        f1.append(f)
        name.append(nam)
        models.append(model)
        # # Display(accuracy, f1, name)

        f, nam, model = SupoortVM(X_train, X_test, y_train_numeric, y_test_numeric)
        performance_without_audio.append(SupoortVM(X_train_WA, X_test_WA, y_train_numeric_WA, y_test_numeric_WA))
        f1.append(f)
        name.append(nam)
        models.append(model)
        # Display(accuracy,f1,name)

        f1 =np.array(f1)
        Max_index = f1.argmax()
        Max_index_name = name[Max_index]

        Best_model = models[Max_index]
        Best_model_without_audio = max(performance_without_audio,key=lambda y:y[0])

        if Max_index_name is 'Logistic Regression':
            if Best_model_without_audio[1] is 'Logistic Regression':
                return (Best_model, Max_index_name,X_test,y_test_binarized),(Best_model_without_audio,X_test_WA,y_test_binarized_WA)
            else:
                return (Best_model, Max_index_name, X_test, y_test_binarized), (
                Best_model_without_audio, X_test_WA, y_test_numeric_WA)
        else:
            if Best_model_without_audio[1] is 'Logistic Regression':
                return (Best_model,Max_index_name,X_test,y_test_numeric),(Best_model_without_audio,X_test_WA,y_test_binarized_WA)
            else:
                return (Best_model, Max_index_name, X_test, y_test_numeric), (
                Best_model_without_audio, X_test_WA, y_test_numeric_WA)

    # Evalute the model fitted on the training data with inputted testing data and paramter
    #
    # X: Testing features data
    # Y: testing true data parameter:
    # Parameter to be used in evaluting the model ex. 'accuracy','recall'
    #
    # returns: the evalutation in percentage rounded upto 3 decimals
    def evalution(self,X,Y,model,parameter='f1'):
        f1 = model.score(X, Y)
        return f1

    # Predict the emotion of the audio files in X using th model trained before
    #
    # X: np array of filenames
    #
    # returns: an np array of emotion labels for each audio file
    def predict(self,X):
        pass


if __name__=='__main__':
        input_voice_files = os.listdir("Data")
        print(input_voice_files)
        #the best model on the data is returned
        emotion = emotion_analysis()

        models = emotion.fit(input_voice_files)
        model,name,X_test,Y_test= models[0]
        model_without_audio,X_test_WA,Y_test_WA = models[1]

        print("The evaluation of the "+name+" model on the test data :\n")
        print(emotion.evaluate(X_test,Y_test,model))

        print("\n\nThe evaluation of the " + model_without_audio[1] + " model on the test data with out audio features:\n")
        print(emotion.evaluate(X_test_WA, Y_test_WA, model_without_audio[2]))