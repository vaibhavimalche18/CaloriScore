import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

def train_ml_model():
    # Step 1: Load the dataset
    df = pd.read_csv("dataset/food_nutrition.csv")  # or your correct path
    print("📊 Columns found:", list(df.columns))

    # ✅ Step 2: Clean the dataset (add this right after reading CSV)
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with missing or invalid data
    df = df.dropna()

    # Step 3: Separate features and target
    X = df.drop(columns=["calories"])  # your features
    y = df["calories"]  # your label

    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 5: Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Step 6: Train the model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Step 7: Save model and scaler
    joblib.dump(model, "model/food_calorie_model.pkl")
    joblib.dump(scaler, "model/scaler.pkl")

    print("✅ Model training complete and saved!")

if __name__ == "__main__":
    train_ml_model()
