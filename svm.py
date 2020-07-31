import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score

style.use('ggplot')

class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    def fit(self, training_data):
        self.train_data = training_data
        # { ||w||: [w,b] }
        optimum_values = {}

        transforms = [[1, 1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]

        feature_set = []
        for yi in self.train_data:

            for featureset in self.train_data[yi]:

                for feature in featureset:
                    feature_set.append(feature)

        self.max_feature_value = max(feature_set)
        self.min_feature_value = min(feature_set)
        feature_set = None

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      ]

        # extremely expensive
        b_range_multiple = 0.4
        # we dont need to take as small of steps
        # with b as we do w
        b_multiple = 3
        w_optimum = self.max_feature_value * 10

        for step in step_sizes:
            w = np.array([w_optimum, w_optimum])

            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple),
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):

                    #print (b)
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True

                        for i in self.train_data:
                            # print(i)
                            for xi in self.train_data[i]:
                                yi = i

                                if yi * (np.dot(w_t, xi) + b) < 1:
                                    found_option = False
                                break

                        if found_option:
                            optimum_values[np.linalg.norm(w_t)] = [w_t, b]
                        break

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in optimum_values])
            # ||w|| : [w,b]
            optimum = optimum_values[norms[0]]
            self.w = optimum[0]
            self.b = optimum[1]
            w_optimum = optimum[0][0] + step * 2


    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    def visualize(self):
        [[self.ax.scatter(x[-1], x[1], s=100, color=self.colors[i]) for x in my_dict[i]] for i in my_dict]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # (w.x+b) = 0
        # positive support vector hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


header = ['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']
data_read = pd.read_csv(r"train_dataset.csv",sep=',', usecols=header)

data_read1=data_read[['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']]

my_dict={}
list=[]
list2=[]
for index,row in data_read1.iterrows():
    if row['Labels']==1:
        list.append(row[3:].tolist())
    else:
        list2.append(row[3:].tolist())

my_dict[1]=np.asarray(list)
my_dict[-1]=np.asarray(list2)

svm = SVM()
svm.fit(training_data=my_dict)


header = ['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']
data_read2 = pd.read_csv(r"test_dataset.csv",sep=',', usecols=header)
data_read3=data_read2[['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']]

list3 = []

label = data_read3['Labels']
list4 = []
for index,row in data_read3.iterrows():
        list3.append(row[3:].tolist())

for x in label:
        list4.append(x)

predict_us = list3
pred = []

for p in predict_us:
    pred.append(svm.predict(p))

svm.visualize()

results = confusion_matrix(list4, pred,labels=[-1,1])
print('Confusion Matrix:')
print(results)

print(classification_report(list4, pred))
print('Accuracy Score :',accuracy_score(list4, pred))

products_list = data_read3['Product_Name']
products = []
for x in products_list:
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

perf_measure(products,list4,pred)
header = ['Products_Name','Actual','Predicted','TP','FP','TN','FN']
dfObj = pd.DataFrame(complete_list,columns=header)
dfObj.to_csv("Output_SVM.csv",index=False)