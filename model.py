import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("sleep_data.csv")

# Convert text values into numbers
label = LabelEncoder()
data['Stress_Level'] = label.fit_transform(data['Stress_Level'])
data['Exercise_Frequency'] = label.fit_transform(data['Exercise_Frequency'])
data['Sleep_Quality'] = label.fit_transform(data['Sleep_Quality'])

# Input (X) and Output (y)
X = data.drop('Sleep_Quality', axis=1)
y = data['Sleep_Quality']

# Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test accuracy
prediction = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, prediction))
