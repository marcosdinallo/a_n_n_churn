# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
# library to encode
from sklearn.preprocessing import LabelEncoder
# library fot hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# library for split model between train and test
from sklearn.model_selection import train_test_split
# library for scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score



# Check tensorflow version
print('TensorFlow version: ', tf.__version__)

# Part 1 - Data Preprocessing

# Importing the dataset
# Read file
dataset = pd.read_csv('Churn_Modelling.csv')
# Data from column index 3 to penultimate (excluding official result)
x = dataset.iloc[:, 3:-1].values
# Only data from last column (official result)
y = dataset.iloc[:, -1].values
# Check data to use as input
print('Dataset used to calculate result: ', x)
# Check data to use as official result
print('Dataset with official result: ', y)


# Encoding categorical data

# Label Encoding the "Gender" column
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
print('Encoded dataset used to calculate result: ', x)
# One Hot Encoding the "Geography" column
# hot encoding obj, column 1 (Geography)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# this hot encoding set encoded geography column as the first column
x = np.array(ct.fit_transform(x))
print('Hot encoded dataset: ', x)
# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print("Split X train result: ", x_train)
print("Split X test result: ", x_test)
print("Split Y train result: ", y_train)
print("Split Y test result: ", y_test)

# Feature Scaling
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
print('Scaling x train: ', x_train)
print('Scaling x test: ', x_test)

# Part 2 - Building the ANN

# Initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
hidden_neuron_unit_layers = 6
output_neuron_unit_layers = 1
activation_function = 'relu'
activation_function_probability = 'sigmoid'

ann.add(tf.keras.layers.Dense(units=hidden_neuron_unit_layers, activation=activation_function))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=hidden_neuron_unit_layers, activation=activation_function))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=output_neuron_unit_layers, activation=activation_function_probability))
# Part 3 - Training the ANN

# Compiling the ANN
optimizer_function = 'adam'
loss_function = 'binary_crossentropy'
neuron_metrics = ['accuracy']
ann.compile(optimizer=optimizer_function, loss=loss_function, metrics=neuron_metrics)
# Training the ANN on the Training set
ann.fit(x_train, y_train, batch_size=32, epochs=100)
# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France ===== 1,0,0  ---- because HOT encoding
Credit Score: 600
Gender: Male ===== 1 because encoding
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""
prediction_result = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
print('Should we say goodbye to that customer?: ', 'Yes' if prediction_result > 0.5 else 'No')
"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""

# Predicting the Test set results
y_pred = ann.predict(x_test)
y_pred = y_pred > 0.5
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))
# Making the Confusion Matrix

print('Confusion Matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy Score: ', accuracy_score(y_test, y_pred))
