from preprocessing import load_data, preprocess
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt

# Load data
df = load_data('traffic.csv')

# Clean data
df = preprocess(df)

# Target column
target = 'Total Traffic Accidents - Cases'

X = df.drop(target, axis=1)
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = build_model()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Show error
print("MSE:", mean_squared_error(y_test, pred))

# Save model
joblib.dump(model, '../model.pkl')

# Graph
plt.plot(y_test.values, label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.title("Traffic Prediction")
plt.show()
