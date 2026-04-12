import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = pd.read_csv("traffic.csv")

print(data.head())

data = data[~data['State/UT/City'].str.contains('Total')]

data = data.drop(['State/UT/City'], axis=1)

data = data.fillna(0)

data = data.apply(pd.to_numeric)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Error:", mean_squared_error(y_test, y_pred))

import matplotlib.pyplot as plt

plt.figure()
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.title("Actual vs Predicted Traffic Accident Deaths")
plt.xlabel("Data Points")
plt.ylabel("Number of Deaths")
plt.legend()
plt.show()

plt.figure()
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Scatter Plot")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.show()

errors = y_test - y_pred

plt.figure()
plt.hist(errors)
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

print("Project Completed")
