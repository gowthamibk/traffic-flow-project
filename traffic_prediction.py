import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("traffic.csv")

print(data.head())

# Remove rows with 'Total'
data = data[~data['State/UT/City'].str.contains('Total')]

# Drop text column
data = data.drop(['State/UT/City'], axis=1)

# Fill missing values
data = data.fillna(0)

# Convert all data to numeric (IMPORTANT)
data = data.apply(pd.to_numeric)

# Features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Error
print("Error:", mean_squared_error(y_test, y_pred))

import matplotlib.pyplot as plt

# Graph 1: Actual vs Predicted (Line Graph)
plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Actual vs Predicted Traffic Accident Deaths")
plt.xlabel("Data Points")
plt.ylabel("Number of Deaths")
plt.legend()
plt.show()


# Graph 2: Scatter Plot
plt.figure()
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Scatter Plot")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()


# Graph 3: Error Distribution
errors = y_test - y_pred

plt.figure()
plt.hist(errors)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

print("Project Completed")