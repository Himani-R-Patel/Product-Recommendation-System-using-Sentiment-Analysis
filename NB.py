import string
import pandas as pd
import numpy as np
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix


train = pd.read_csv(r"train_dataset.csv")
X_train = train['Feature_0']
Y_train = train['Labels']

test = pd.read_csv(r"test_dataset.csv")
X_test = test['Feature_0']
Y_test = test['Labels']

# Building a vocabulary of words from the given documents
vocab = {}
for i in range(len(X_train)):
    word_list = []
    for word in X_train[i].split():
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1


features = []
for key in vocab:
    features.append(key)

#
# print("Xtrain is :",X_train)
# print("features are",features)

# To represent training data as word vector counts
X_train_dataset = np.zeros((len(X_train), len(features)))
for i in range(len(X_train)):
    word_list = [word.strip(string.punctuation).lower() for word in X_train[i].split()]
    for word in word_list:
        if word in features:
            X_train_dataset[i][features.index(word)] += 1


# To represent test data as word vector counts
X_test_dataset = np.zeros((len(X_test), len(features)))
for i in range(len(X_test)):
    word_list = [word.strip(string.punctuation).lower() for word in X_test[i].split()]
    for word in word_list:
        if word in features:
            X_test_dataset[i][features.index(word)] += 1

print(X_train_dataset)
print("column length:",len(X_train_dataset[0]))
print("row length :",len(X_train_dataset))
print("Y_train is:",Y_train)



# Implementing Multinomial Naive Bayes
class MultinomialNaiveBayes:

    def __init__(self):

        self.count = {}
        self.labels = None

    def fit(self, X_train, Y_train):
        self.labels = set(Y_train)

        for label in self.labels:
            self.count[label] = {}
            for i in range(len(X_train[0])):
                self.count[label][i] = 0
            self.count[label]['total'] = 0
        self.count['total_points'] = len(X_train)
        print("count is:", self.count)

        for i in range(len(X_train)):
            for j in range(len(X_train[0])):
                print("Ytrain i is : ",Y_train[i])
                self.count[Y_train[i]][j] += X_train[i][j]
            self.count[Y_train[i]]['total'] += 1
        print("count after cv: ", self.count)

    def probability(self, test_point, class_):

        log_prob = np.log(self.count[class_]['total']) - np.log(self.count['total_points'])
        total_words = len(test_point)
        for i in range(len(test_point)):
            current_word_prob = test_point[i] * (
                        np.log(self.count[class_][i] + 1) - np.log(self.count[class_]['total'] + total_words))
            log_prob += current_word_prob

        return log_prob

    def pred_one(self, test_point):

        best_class = None
        best_prob = None
        first_run = True

        for label_ in self.labels:
            log_probability_current_class = self.probability(test_point, label_)
            if (first_run) or (log_probability_current_class > best_prob):
                best_class = label_
                best_prob = log_probability_current_class
                first_run = False

        return best_class


    def predict(self, X_test):
        Y_pred = []
        for i in range(len(X_test)):
            Y_pred.append(self.pred_one(X_test[i]))

        return Y_pred

    def accuracy(self, Y_pred, Y_true):
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                count += 1
        return count / len(Y_pred)


mnb = MultinomialNaiveBayes()
mnb.fit(X_train_dataset, Y_train)

Y_test_pred = mnb.predict(X_test_dataset)
print("Ytest pred :" , Y_test_pred)

results_accuracy = mnb.accuracy(Y_test_pred, Y_test)

results = confusion_matrix(Y_test, Y_test_pred,labels=[1,-1])
print('Confusion Matrix:')
print(results)



print("Our accuracy on testing data :", results_accuracy)
print("Classification report for testing data :-")
print(classification_report(Y_test, Y_test_pred))



# # Using sklearn's Multinomial Naive Bayes
# NB = MultinomialNB()
# NB.fit(X_train_dataset, Y_train)
# Y_test_pred = NB.predict(X_test_dataset)
#
# sklearn_score_train = NB.accuracy(X_train_dataset, Y_train)
# print("Sklearn's accuracy on training data :", sklearn_score_train)
#
# sklearn_score_test = NB.accuracy(X_test_dataset, Y_test)
# print("Sklearn's accuracy on testing data :", sklearn_score_test)
#
# print("Classification report for testing data :-")
#
# # print("Score of our model on test data:", our_score_test)
# # print("Score of inbuilt sklearn's MultinomialNB on the same data :", sklearn_score_test)
#