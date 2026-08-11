import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("disease.csv")

# Remove spaces from column names
df.columns = df.columns.str.strip()

print("Original Data Loaded:", df.shape)

# =========================
# DATA CLEANING
# =========================

df = df.drop_duplicates()
df = df.fillna(0)

print("After Cleaning:", df.shape)

# =========================
# FEATURES AND TARGET
# =========================

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Save feature names
features = X.columns.tolist()

# =========================
# ENCODE TARGET
# =========================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Number of diseases:", len(label_encoder.classes_))
print("Diseases:", label_encoder.classes_)

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# =========================
# DECISION TREE
# =========================

model = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================

y_pred = model.predict(X_test)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\nDecision Tree Accuracy: {:.2f}%".format(accuracy * 100))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "disease_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(features, "features.pkl")

print("\nFiles created successfully!")
print("disease_model.pkl")
print("label_encoder.pkl")
print("features.pkl")
