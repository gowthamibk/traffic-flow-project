from preprocessing import load_data, preprocess
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# Correct file path (guaranteed)
file_path = os.path.join(os.path.dirname(__file__), 'traffic.csv')

df = load_data(file_path)
df = preprocess(df)

print("Columns:", df.columns)

target = 'Total Traffic Accidents - Cases'

X = df.drop(target, axis=1)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = build_model()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

joblib.dump(model, 'model.pkl')

plt.plot(y_test.values, label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.title("Traffic Prediction")
plt.show()
