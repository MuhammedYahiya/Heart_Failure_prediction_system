import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# loading and reading the dataset
heart_df = pd.read_csv("heart_cleveland_upload.csv")

# Renaming the target column
heart_df = heart_df.rename(columns={'condition': 'target'})

# Separate features and target variable
X = heart_df.drop(columns='target')
y = heart_df['target']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating K-Nearest-Neighbor classifier (KNN)
# We'll use GridSearchCV to find the best hyperparameters
param_grid = {'n_neighbors': range(1, 21)}  # trying neighbors from 1 to 20
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
best_k = best_params['n_neighbors']

# Train the KNN model with the best hyperparameters
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: {}%\n'.format(round((accuracy_score(y_test, y_pred)*100),2)))

# Print classification report
print('Classification Report\n', classification_report(y_test, y_pred))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save the trained model
filename = 'heart-disease-prediction-knn-model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)
