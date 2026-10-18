import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("loan_data.csv")

features = ["principal", "rate", "tenure", "num_missed", "rate_change_month",
            "new_rate", "base_emi", "base_interest"]
X = df[features]
y_extra = df["total_interest"] - df["base_interest"]
y_months = df["total_months"]

X_train, X_test, ye_train, ye_test, ym_train, ym_test = train_test_split(
    X, y_extra, y_months, test_size=0.2, random_state=42
)

interest_model = RandomForestRegressor(n_estimators=100, random_state=42)
months_model = RandomForestRegressor(n_estimators=100, random_state=42)
interest_model.fit(X_train, ye_train)
months_model.fit(X_train, ym_train)

base_test = X_test["base_interest"]
pred_total = interest_model.predict(X_test) + base_test
true_total = ye_test + base_test
pred_months = months_model.predict(X_test)

print("Interest MAE:", mean_absolute_error(true_total, pred_total))
print("Interest R2 :", r2_score(true_total, pred_total))
print("Months MAE  :", mean_absolute_error(ym_test, pred_months))
print("Months R2   :", r2_score(ym_test, pred_months))
print("Avg interest in data:", true_total.mean())

joblib.dump(interest_model, "interest_model.pkl")
joblib.dump(months_model, "months_model.pkl")