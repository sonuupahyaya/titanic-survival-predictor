import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load cleaned Titanic dataset
df = pd.read_csv("Titanic-Dataset-Cleaned.csv")

# ✅ Drop irrelevant columns including PassengerId
df = df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

# Encode categorical columns
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])         # male=1, female=0
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])  # S=2, C=0, Q=1

# Split features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# ---------------- Streamlit UI ----------------
st.title("🚢 Titanic Survival Prediction App")

st.sidebar.header("Enter Passenger Details")

def get_user_input():
    pclass = st.sidebar.selectbox("Passenger Class (Pclass)", [1, 2, 3])
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    age = st.sidebar.slider("Age", 0, 80, 25)
    sibsp = st.sidebar.slider("Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
    parch = st.sidebar.slider("Parents/Children Aboard (Parch)", 0, 6, 0)
    fare = st.sidebar.slider("Fare ($)", 0, 500, 50)
    embarked = st.sidebar.selectbox("Port of Embarkation", ["S", "C", "Q"])

    # Encode input values
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = le_embarked.transform([embarked])[0]

    data = {
        "Pclass": pclass,
        "Sex": sex_encoded,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked_encoded
    }
    return pd.DataFrame([data])

# Get user input
user_input = get_user_input()

# Show user input
st.subheader("Passenger Details")
st.write(user_input)

# Make prediction
prediction = model.predict(user_input)[0]
prediction_proba = model.predict_proba(user_input)

# Show prediction
st.subheader("Prediction Result")
if prediction == 1:
    st.success("🎯 The passenger **would have survived**.")
else:
    st.error("💀 The passenger **would not have survived**.")

# Show probabilities
st.subheader("Prediction Probability")
st.write(f"🟥 Not Survived: {prediction_proba[0][0] * 100:.2f}%")
st.write(f"🟩 Survived: {prediction_proba[0][1] * 100:.2f}%")
