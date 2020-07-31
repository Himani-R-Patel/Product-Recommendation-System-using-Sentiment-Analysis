import pandas as pd
from prettytable import PrettyTable

header = ['Products_Name','Actual','Predicted','TP','FP','TN','FN']
data_read = pd.read_csv(r"Output_KNN.csv",sep=',')
data_frame1=data_read[['Products_Name','Actual','Predicted','TP','FP','TN','FN']]

group = data_frame1.groupby('Products_Name')

# table = PrettyTable(['Product','Positive Reviews','Negative Reviews'])
# for Product_Name,Product_df in group:
#     table.add_row([Product_Name,Product_df['TP'].sum(),Product_df['TN'].sum()])
#
# print(table)

# for Product_Name,Product_df in group:
#     print('This ', Product_Name, ' Product has been recommended by ', Product_df['TP'].sum(), 'customers ',
#           'and not recommended by ', Product_df['TN'].sum(), 'customers.')

Product_values = {}
for Product_Name,Product_df in group:
    Product_values.update({Product_Name:[Product_df['TP'].sum(),Product_df['TN'].sum()]})

Product = input("Please enter the Product Name?\n")
if Product in Product_values.keys():
    print('This Product', Product, 'has been recommended by', Product_values.get(Product)[0], 'customers',
              'and not recommended by', Product_values.get(Product)[1], 'customers.')
else:
    print("This is not among the list of products")
