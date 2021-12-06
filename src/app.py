#imports
import pandas
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns

from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, RocCurveDisplay

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopwords = stopwords.words('english') 

from sklearn.feature_extraction.text import CountVectorizer

class Sentiment:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path



    def load_dataset(self):
        #load dataset
        data = pandas.read_csv(
            self.dataset_path,
            encoding='latin-1',
            header=None,
            usecols=[0, 5],
            names=['target', 'text']
        )

        #random sample
        data = data.sample(n=50000, random_state=100).reset_index(drop=True)

        return data

    def clean_text(self, text):
        """
        Return clean text
        params
        ------------
            text: string
        """

        text = text.lower() #lowercase
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if not t in stopwords] #remove stopwords
        tokens = [t for t in tokens if t.isalnum()] #remove punctuation
        text_clean = " ".join(tokens)
        
        return text_clean

    def data_split(self, data_text):
        #train test split
        train = data_text[0:40000]
        test = data_text[40000:50000].reset_index(drop=True)

        return train, test

    def vectorize_text(self, train, test):
        # Create count vectoriser
        vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)

        # Transform training corpus into feature matrix
        train_csr_matrix = vectorizer.fit_transform(train['text'])
        feature_names = vectorizer.get_feature_names()

        x_train = pandas.DataFrame(data=train_csr_matrix.toarray(), columns=feature_names)
        y_train = train['target']

        # Transform testing corpus into feature matrix
        test_csr_matrix = vectorizer.transform(test['text'])

        x_test = pandas.DataFrame(data=test_csr_matrix.toarray(), columns=feature_names)
        y_test = test['target']

        return x_train, y_train, x_test, y_test

    def scale_feature_matrix(self, x_train, x_test):
        # Min-Max scalling
        x_train_maximum = x_train.max()
        x_train_minimum = x_train.min()

        x_train = (x_train - x_train_maximum)/x_train_maximum
        x_test = (x_test - x_train_minimum)/x_train_maximum

        return x_train, x_test

    def model(self, x_train, y_train, x_test, y_test): 
        #fit SVM algorithm 
        # do some fits on our x train data. Precisely known as training the model
        print("Ready to do some fits. Please wait...")
        model = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

        #get predictions on test set
        print("Ready to do some predictions on xtest data. Please wait...")
        y_predictions = model.predict(x_test)
        print("\n")
        print("\n")

        #accuracy on test set
        model_accuracy = accuracy_score(y_test, y_predictions)
        print("Accuracy: {}".format(model_accuracy))  

        return model, y_predictions

    def error_matrix(self, y_test, predictions):
        #create confusion matrix
        conf_matrix = confusion_matrix(y_test, predictions)
        conf_matrix = pandas.DataFrame(data = conf_matrix, columns=['negative','positive'], index=['negative','positive'])

        #plot confusion matrix
        plot.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
        sns.set(font_scale=1.5)
        sns.heatmap(conf_matrix,cmap='coolwarm',annot=True,fmt='.5g',cbar=False)
        plot.ylabel('Actual',size=20)
        plot.xlabel('Predicted',size=20)

        return plot