import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
# Load dataset
df = pd.read_csv("data/wildlife_data.csv")

# Features
X = df[["Population", "Threat_Level"]]
y = df["Migration_Distance"]

# Create RandomForest model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

# Train model
model.fit(X, y)
os.makedirs("models", exist_ok=True)
# Save model
joblib.dump(model, "models/migration_model.pkl")

print("✅ Migration model trained and saved successfully!")