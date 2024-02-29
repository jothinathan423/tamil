import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

# Load dataset
dataset = pd.read_csv('cropdata.csv')

# Separate features and target variable
X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6]].values
y = dataset.iloc[:, 7].values

# Scale features
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=dataset['label'])

# Train Naive Bayes classifier
naive_bayes = GaussianNB()
naive_bayes.fit(X_train, y_train)

# Score of the model on test data
score = naive_bayes.score(X_test, y_test)
print("Model Accuracy:", score)

# Predict using the trained model
y_pred = naive_bayes.predict(X_test)

# Save the model
pickle.dump(naive_bayes, open('model.pkl', 'wb'))

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# Sample input data for prediction
sample_data = np.array([[106, 18, 70, 23.603016, 60.3, 6.7, 140.91]])

# Transform the input data using the scaler
scaled_sample_data = scaler.transform(sample_data)

# Predict the label for the sample data
predicted_label = model.predict(scaled_sample_data)[0]
print("Predicted Label:", predicted_label)

# Filter the dataset based on the predicted label
remaining_columns = dataset.columns[dataset.columns != 'label'] if predicted_label == 'rice' else dataset[dataset['label'] == predicted_label]

# Show remaining columns or remaining data based on the condition
print("Remaining Columns:")
print(remaining_columns)
