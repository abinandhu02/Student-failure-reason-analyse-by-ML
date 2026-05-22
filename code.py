
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
input_path = r"C:\Users\ELCOT\Downloads\proj_data.csv"
df = pd.read_csv(input_path)

# Clean column names
df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully")
print("Total Students:", len(df))

np.random.seed(42)

df['Previous Backlogs'] = np.random.choice(
    [0,1,2,3,4,5],
    size=len(df),
    p=[0.7,0.15,0.05,0.05,0.03,0.02]
)

df['Library Marks'] = np.random.randint(0,6,size=len(df))
df['Participation Score'] = np.random.randint(0,6,size=len(df))

df["Failure_Type"] = np.where(
    df["Attendance %"] < 50, "Low Attendance",
    np.where(
        df["Average Internal"] < 50, "Low Internal Marks",
        np.where(
            df["Final Marks"] < 20, "Poor ESE Performance",
            "Other Academic Issues"
        )
    )
)

# Convert to numeric labels for ML
df["Reason_Label"] = df["Failure_Type"].astype("category").cat.codes

features = [
    "Attendance %",
    "Average Internal",
    "Final Marks",
    "Previous Backlogs",
    "Library Marks",
    "Participation Score"
]

X = df[features]
y = df["Reason_Label"]

X_noisy = X + np.random.normal(0,0.5,X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X_noisy,
    y,
    test_size=0.3,
    random_state=42
)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


df["Predicted_Label"] = model.predict(X)

# Convert numeric labels back to text
label_map = dict(enumerate(df["Failure_Type"].astype("category").cat.categories))
df["Predicted_Failure_Reason"] = df["Predicted_Label"].map(label_map)

if "Result" in df.columns:
    failed_df = df[df["Result"].str.strip().str.lower() == "fail"].copy()
else:
    failed_df = df[
        (df["Attendance %"] < 50) |
        (df["Average Internal"] < 50) |
        (df["Final Marks"] < 20)
    ].copy()

print("Failed Students:", len(failed_df))

export_df = failed_df.drop(columns=["Reason_Label","Predicted_Label"])

output_path = r"C:\Users\ELCOT\Desktop\student_failure_ml_report.xlsx"

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    export_df.to_excel(writer,index=False)

print("\nReport exported successfully to:")
print(output_path)