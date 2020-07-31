import pandas as pd
import numpy as np
import operator
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

def euclideanDistance(data1, data2, length):
     distance = 0
     for x in range(length):
         distance += np.square(data1[x] - data2[x])
     return np.sqrt(distance)


def knn(train_data, test_data, k):
    distances = {}
    length = test_data.shape[1]
    for x in range(len(train_data)):
        dist = euclideanDistance(test_data, train_data.iloc[x], length)
        distances[x] = dist[0]
    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []

    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classCnt = {}

    for x in range(len(neighbors)):
        response = train_data.iloc[neighbors[x]][-1]

    if response in classCnt:
        classCnt[response] += 1
    else:
        classCnt[response] = 1

    sorted_cnt = sorted(classCnt.items(), key=operator.itemgetter(1), reverse=True)
    return (sorted_cnt[0][0], neighbors)

data = pd.read_csv("train_dataset.csv")
train = data[['Feature_2', 'Feature_1', 'Labels']]


testdata = pd.read_csv("test_dataset.csv")

df= testdata['Labels']
prod = testdata['Product_Name']

labels_list=df.values.tolist()

dftestdata = testdata[['Feature_2','Feature_1']]
testdata_list=dftestdata.values.tolist()


k = 8
predicted_data=[]

for x in testdata_list:
    testSet=x
    predict = pd.DataFrame(testSet)
    result, neigh = knn(train, predict, k)
    predicted_data.append(result)

results = confusion_matrix(labels_list, predicted_data)
print('Confusion Matrix:')
print(results)
print('Accuracy Score :', accuracy_score(labels_list, predicted_data))
print('Classification Report')
print(classification_report(labels_list, predicted_data))

products = []
for x in prod:
        products.append(x)


complete_list = []
def perf_measure(products,y_actual, y_pred):

    for i in range(len(y_pred)):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==-1:
           TN += 1
        if y_pred[i]==-1 and y_actual[i]!=y_pred[i]:
           FN += 1

        complete_list.append([products[i],y_actual[i],y_pred[i],TP, FP, TN, FN])

perf_measure(products, labels_list, predicted_data)
header = ['Products_Name','Actual','Predicted','TP','FP','TN','FN']
dfObj = pd.DataFrame(complete_list,columns=header)
dfObj.to_csv("Output_KNN.csv",index=False)
