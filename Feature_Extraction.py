import pandas as pd
from sklearn.model_selection import train_test_split

data_read = pd.read_csv(r"Dataset_preprocessed.csv")

df = data_read[['Labels','Feature_0','Product_Name','Feature_1', 'Feature_2']]


train, test = train_test_split(df,test_size=0.2,random_state=1)

X_train = train[['Product_Name','Feature_0','Feature_1', 'Feature_2']]
Y_train = train['Labels']

X_test = test[['Product_Name','Feature_0','Feature_1', 'Feature_2']]
Y_test = test['Labels']


df1 = pd.DataFrame(train)
df1.to_csv("train_dataset.csv",index=False)


df2 = pd.DataFrame(test)
df2.to_csv("test_dataset.csv",index=False)