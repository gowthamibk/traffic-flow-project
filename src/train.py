from preprocessing import load_data, preprocess
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

# ----------------------------
# FIXED FILE PATH (IMPORTANT)
# ----------------------------
base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, 'src', 'traffic.csv')

# Load data
df = load_data(file_path)

# Clean data
df = preprocess(df)

# Check columns (for debugging)
print("Columns:", df.columns)

# Target column (make sure this matches your CSV exactly)
target = 'Total Traffic Accidents - Cases'

# Split features & target
X = df.drop(target, axis=1)
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = build_model()

# Train
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Metrics
print("MSE:", mean_squared_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

# Save model
model_path = os.path.join(base_dir, 'model.pkl')
joblib.dump(model, model_path)

# Plot graph
plt.plot(y_test.values, label='Actual')
plt.plot(pred, label='Predicted')
plt.legend()
plt.title("Traffic Prediction")

# Save graph
graph_path = os.path.join(base_dir, 'output_graph.png')
plt.savefig(graph_path)

plt.show()
