import csv
import pandas as pd
import numpy as np
from collections import OrderedDict, defaultdict
with open("C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/cifar10_alpha/federated_train_alpha_0.00.csv" ) as filecsv:
    lettore = csv.reader(filecsv,delimiter=",")

header_name=[0,1,2]

dict_from_csv= pd.read_csv("C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/cifar10_alpha/federated_train_alpha_0.00.csv", header=0)
new=dict_from_csv.groupby('user_id')['image_id'].apply(set).to_dict()
#print(new[1])
labels= dict_from_csv['class']
