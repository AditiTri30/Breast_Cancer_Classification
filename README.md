# Breast_Cancer_Classification
 Developed a model where analysis of all the data, their standard error and the worst case prediction was done. Based on the learning can classify the cells into benign or malignant. 

#Importing the Dependencies
import numpy as npy
import pandas as pds
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection & Processing

#Loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

#Loading the data to a data frame
data_frame = pds.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

#Adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

#Separating the features and target

X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

#Splitting the data into training data & Testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#Model Training

#Logistic Regression

model = LogisticRegression()

#Training the Logistic Regression model using Training data

model.fit(X_train, Y_train)

#Model Evaluation

#Accuracy Score

#Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data = ', training_data_accuracy)
#Output = (Accuracy on training data =  0.9494505494505494)

#Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)
#Output= (Accuracy on test data =  0.9298245614035088)

#Building a Predictive System

input_data = (#select any random row from dataset)

#Change the input data to a numpy array
input_data_as_numpy_array = npy.asarray(input_data)

#Reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Malignant')

else:
  print('The Breast Cancer is Benign')

#Output = ([1]
The Breast Cancer is Benign)



