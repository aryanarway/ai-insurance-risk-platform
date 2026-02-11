import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# FIXED CSV LOADING (your dataset needs this)
df = pd.read_csv("data/claims.csv", encoding="utf-8-sig")

# Your file currently loads as ONE column, so we split it manually
df = df.iloc[:,0].str.split(",", expand=True)

# Recreate proper column names
df.columns = [
"months_as_customer","age","policy_number","policy_bind_date","policy_state",
"policy_csl","policy_deductable","policy_annual_premium","umbrella_limit",
"insured_zip","insured_sex","insured_education_level","insured_occupation",
"insured_hobbies","insured_relationship","capital-gains","capital-loss",
"incident_date","incident_type","collision_type","incident_severity",
"authorities_contacted","incident_state","incident_city","incident_location",
"incident_hour_of_the_day","number_of_vehicles_involved","property_damage",
"bodily_injuries","witnesses","police_report_available","total_claim_amount",
"injury_claim","property_claim","vehicle_claim","auto_make","auto_model",
"auto_year","fraud_reported","_c39"
]

# remove last empty column
df = df.drop(columns=["_c39"])

# target
y = df["fraud_reported"]
X = df.drop("fraud_reported", axis=1)

# convert text to numbers
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=400, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "models/claims_model.pkl")
print("✅ Claims model trained and saved")

