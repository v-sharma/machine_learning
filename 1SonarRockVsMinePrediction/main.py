import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


sonar_dataset = pd.read_csv('data/sonar.all-data', header=None)
print(sonar_dataset.shape)
print(sonar_dataset.head())
# print(sonar_dataset.info())
print(sonar_dataset.describe())
print(sonar_dataset.isnull().sum())

# Count number of values for Rock and Mine values
print(sonar_dataset[60].value_counts())


print(sonar_dataset.groupby(60).mean())

# Extract data and labels in separate dataframes
X = sonar_dataset.drop(columns=60, axis=1)
Y = sonar_dataset[60]

# Split training and test datasets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=0)

# Create a model
# Train model
model = LogisticRegression()
model.fit(X_Train, Y_Train)

predict = model.predict(X_Train)
score = accuracy_score(Y_Train, predict)

print('Accuracy on train data: ', score)

# Test model
predict = model.predict(X_Test)
score = accuracy_score(Y_Test, predict)
print('Accuracy score of test data: ', score)


# Predicitive system
input_data = (0.0079,0.0086,0.0055,0.0250,0.0344,0.0546,0.0528,0.0958,0.1009,0.1240,0.1097,0.1215,0.1874,0.3383,0.3227,0.2723,0.3943,0.6432,0.7271,0.8673,0.9674,0.9847,0.9480,0.8036,0.6833,0.5136,0.3090,0.0832,0.4019,0.2344,0.1905,0.1235,0.1717,0.2351,0.2489,0.3649,0.3382,0.1589,0.0989,0.1089,0.1043,0.0839,0.1391,0.0819,0.0678,0.0663,0.1202,0.0692,0.0152,0.0266,0.0174,0.0176,0.0127,0.0088,0.0098,0.0019,0.0059,0.0058,0.0059,0.0032)

# Change data input to numpy array
data = np.asarray(input_data)

# Reshape the data as we are only providing one data point
data = data.reshape(1, -1)

# Output should be R
prediction = model.predict(data)
print('Prediction: ', prediction)