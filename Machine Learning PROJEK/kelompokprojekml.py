import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# Judul aplikasi
st.title("Prediksi Diabetes dengan Decision Tree")
st.title("Kelompok Proyek Machine Learning")
st.write("AHMAT SAPI'I HARAHAP - RIFA ADRITIYA PAMUNGKAS - SIPRIANUS RADO SANGKEK")
# Load Dataset
df = pd.read_csv("diabetes.csv")

st.subheader("Dataset")
st.dataframe(df.head())

# Info Dataset
st.subheader("Informasi Dataset")
st.write(df.describe())

# Feature dan Target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Akurasi
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Akurasi Model")
st.success(f"{accuracy:.2%}")

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")

st.pyplot(fig)

# Classification Report
st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

st.dataframe(pd.DataFrame(report).transpose())

# Histogram Glucose
st.subheader("Distribusi Glucose")

fig, ax = plt.subplots(figsize=(8,5))

sns.histplot(
    df["Glucose"],
    kde=True,
    ax=ax
)

st.pyplot(fig)

# Korelasi
st.subheader("Heatmap Korelasi")

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)

# Decision Tree
st.subheader("Visualisasi Decision Tree")

fig, ax = plt.subplots(figsize=(20,10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Tidak Diabetes", "Diabetes"],
    filled=True,
    rounded=True,
    fontsize=10
)

st.pyplot(fig)