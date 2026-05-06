from preprocessing import load_data, preprocess
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# -----------------------------
# FIXED FILE PATH
# -----------------------------
file_path = os.path.join(os.path.dirname(__file__), 'traffic.csv')

# Load data
df = load_data('src/traffic.csv)

# Preprocess
df = preprocess(df)

# Show columns (for checking)
print("Columns:", df.columns)

# Target column (make sure matches CSV)
target = 'Total Traffic Accidents - Cases'

# Features & target
X = df.drop(target, axis=1)
y = df[target]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = build_model()
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# -----------------------------
# METRICS
# -----------------------------
print("MSE:", mean_squared_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, 'model.pkl')

# -----------------------------
# GRAPH
# -----------------------------
plt.plot(y_test.values, label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.title("Traffic Prediction")

# Save graph as image
plt.savefig('output_graph.png')

# Show graph
plt.show()
