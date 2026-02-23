import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("pose_dataset.csv")

# Features and labels
X = data.drop("risk_label", axis=1)
y = data["risk_label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Test accuracy
print("Accuracy on test set:", model.score(X_test_scaled, y_test))

# Save model and scaler
joblib.dump(model, "risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")