import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load your nutrient dataset
df = pd.read_csv("nutrient_dataset.csv")

X = df[['protein','fat','carbs','fiber']]
y = df['calories']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "calorie_lr_model.pkl")

print("Model trained and saved successfully!")
