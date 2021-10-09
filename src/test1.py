# https://www.analyticsvidhya.com/blog/2021/01/a-guide-to-the-naive-bayes-algorithm/
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pca import PCA

# Importing the dataset
dataset = pd.read_csv('./Data/Social_Network_Ads.csv')
AgeColumnIndex = 2
EstimatedSalaryColumnIndex = 3
PurchasedColumnIndex = -1
#X = dataset.iloc[startRow:endRow],StartColumn:EndColumns]
X = dataset.iloc[:, [AgeColumnIndex, EstimatedSalaryColumnIndex]].values
y = dataset.iloc[:, PurchasedColumnIndex].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Naive Bayes model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test, y_pred)
print(ac)



#Applying it to PCA function
mat_reduced = PCA(X_train, 2)
 
#Creating a Pandas DataFrame of reduced Dataset
principal_df = pd.DataFrame(mat_reduced , columns = ['PC1','PC2'])
 
#Concat it with target variable to create a complete Dataset
principal_df = pd.concat([principal_df , pd.DataFrame(y_train)] , axis = 1)

colors = list()
palette = {0: "red", 1:'green'}

for c in y_train: 
    colors.append(palette[c])

import matplotlib.pyplot as plt
plt.scatter(principal_df['PC1'], principal_df['PC2'], c = colors)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# show the full number
np.set_printoptions(suppress=True)

#Queries:
result = pd.DataFrame(X_train)
for index, row in result.iterrows():
	sample = pd.DataFrame({0: [row[0]], 1: [row[1]]})
	result = classifier.predict(sample)
	sample_predict = sc.inverse_transform(sample)
	print("Query 1:- {} ---> {}".format(sample_predict, result))
