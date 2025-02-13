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

# Save the model
filename = 'knn.pkl'
pickle.dump(knn, open(filename,'wb'))

try:
    # Load the saved model
    model = pickle.load(open(filename, 'rb'))

    # Check model attributes
    print("Model parameters:", model.get_params())

    # New data for prediction
    new_data = np.array([[47,1,2,108,243,0,1,152,0,0,2,0,2]])

    # Convert to DataFrame with column names
    feature_names = X.columns  # Get feature names from training data
    new_data_df = pd.DataFrame(new_data, columns=feature_names)

    # Make prediction on new data
    prediction = model.predict(new_data_df)
    print("Prediction:", prediction)

    # Get the predicted class probabilities
    class_probabilities = model.predict_proba(new_data_df)
    print("Class probabilities:", class_probabilities)

    # Calculate the percentage based on the probabilities
    percentage = class_probabilities[0][1] * 100
    print("Percentage prediction:", percentage)

except Exception as e:
    print("Error loading the model:", e)
