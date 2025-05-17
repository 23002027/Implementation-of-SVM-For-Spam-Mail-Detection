# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
Program to implement the SVM For Spam Mail Detection..

Developed by: KAMESH RR
RegisterNumber: 212223230095
```
```
import chardet
file = "/content/spam.csv"
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
print(result)

import pandas as pd
data = pd.read_csv( "/content/spam.csv", encoding='windows-1252')
print(data.head())
print(data.info())
print(data.isnull().sum())

x = data["v2"].values 
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
```

## Output:

![image](https://github.com/user-attachments/assets/977e8d76-67f1-4324-bd98-49d94f7ee863)

![image](https://github.com/user-attachments/assets/f7c3f12a-9d24-488e-8947-c6c2d7a1d27d)

![image](https://github.com/user-attachments/assets/f79218b2-c460-4ea4-ae1e-72730e3e28a2)

![image](https://github.com/user-attachments/assets/0c94616b-3cbc-4c19-8215-f115e6846a0d)

![image](https://github.com/user-attachments/assets/1f04634e-1493-4ba0-90fc-a051dc4f4934)

![image](https://github.com/user-attachments/assets/3f86c342-44f5-429f-85fb-d505426c81f9)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
