import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
data = pd.read_csv('heart.csv')

# Separate features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Initialize and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions on test data
y_pred = knn.predict(X_test)

# Save the model
filename = 'knn.pkl'
pickle.dump(knn, open(filename,'wb'))

try:
    # Load the saved model
    model = pickle.load(open(filename, 'rb'))

    # Check model attributes
    print("Model parameters:", model.get_params())

    # New data for prediction
    new_data = np.array([[56,1,2,130,256,1,0,142,1,0.6,1,1,1]])
    print("New data:", new_data)

    # Reshape new_data array
    new_data = new_data.reshape(1, -1)

    # Make prediction on new data
    prediction = model.predict(new_data)
    print("Prediction:", prediction)
except Exception as e:
    print("Error loading the model:", e)
