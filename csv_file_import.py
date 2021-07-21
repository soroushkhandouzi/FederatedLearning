import csv
import pandas as pd
import numpy as np
with open("C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/cifar10_alpha/federated_train_alpha_0.00.csv" ) as filecsv:
    lettore = csv.reader(filecsv,delimiter=",")

header_name=[0,1,2]

dict_from_csv = pd.read_csv("C:/Users/Oana Madalina Breban/Downloads/FederatedLearning-main/FederatedLearning-main/cifar10_alpha/federated_train_alpha_0.00.csv", header=0)

print(dict_from_csv)



dict_users= {}
lists_users=[]
list_users=set
users = np.arange(0,100)
user=0
lists_users1=[]
list_users1=set
for index in range(len(dict_from_csv)):
       if dict_from_csv['user_id'][index]==65:
           #print(dict_from_csv['user_id'][index])
           #print(dict_from_csv['image_id'][user])
            lists_users.append(dict_from_csv['image_id'][index])
            list_users=set(lists_users)
            dict_users[0]=list_users

       if dict_from_csv['user_id'][index]==1:
           #print(dict_from_csv['user_id'][index])
           #print(dict_from_csv['image_id'][user])
            lists_users1.append(dict_from_csv['image_id'][index])
            list_users1=set(lists_users)
            dict_users[1]=list_users1



print(list_users)
print(dict_users)




'''
d = {'key': 'value'}
print(d)
# {'key': 'value'}
d['mynewkey'] = 'mynewvalue'
print(d)
# {'key': 'value', 'mynewkey': 'mynewvalue'}

print(dict_from_csv.values)
dict_users = collections.defaultdict(dict)

values=dict_from_csv.values
dict={}

for i in range(len(values)):

    if values[i][0] not in dict:
        dict[values[i][0]] = values[i][1]
    else:
        list_images.append(dict[values[i][0]])
        list_images.append(values[i][1])
        dict[values[i][0]] = list_images

#print(list_images)
print(dict)

dict = {}
dict_users = collections.defaultdict(dict)
for row in dict_from_csv.rows():

    user = dict_from_csv['user_id']
    image = dict_from_csv[index]['image_id']
    if user not in dict:
            dict[user] = image

print(dict)
'''