import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt   
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score

def create_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        age INTEGER
    )''')
    conn.commit()
    conn.close()

def add_user(name, email, age):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('INSERT INTO users(name, email, age) VALUES (?, ?, ?)', (name, email, age))
    conn.commit()
    conn.close()

def view_users():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users')
    data = c.fetchall()
    conn.close()
    return data

def delete_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('DELETE FROM users WHERE id=?', (user_id,))
    conn.commit()
    conn.close()


@st.cache_data
def load_data():
    data = pd.read_csv(
        "https://raw.githubusercontent.com/alltorie/streamlit_app_activity/refs/heads/main/mushroom_data_all.csv"
    )
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data


@st.cache_data
def split(df):
    y = df["type"]
    x = df.drop(columns=['type'])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0
    )
    return x_train, x_test, y_train, y_test



def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader('Confusion Matrix')
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
        st.pyplot(plt.gcf())

    if 'ROC Curve' in metrics_list:
        st.subheader('ROC Curve')
        RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(plt.gcf())

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader('Precision-Recall Curve')
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot(plt.gcf())



def main():
    st.title("Binary Classification Web App")
    st.caption("By: Allyssa Marie M. Martin")

    menu = st.sidebar.radio(
        "Choose a feature:",
        ["Mushroom Classifier 🧠", "User Database 👤"]
    )
    if menu == "User Database 👤":
        st.header("SQLite User Management System")
        create_table()
        choice = st.sidebar.selectbox("Menu", ["Add User", "View Users", "Delete User"])

        if choice == "Add User":
            st.subheader("Add New User")
            name = st.text_input("Name")
            email = st.text_input("Email")
            age = st.number_input("Age", 0, 120)
            if st.button("Submit"):
                add_user(name, email, age)
                st.success(f"{name} added successfully!")

        elif choice == "View Users":
            st.subheader("All Registered Users")
            users = view_users()
            df = pd.DataFrame(users, columns=["ID", "Name", "Email", "Age"])
            st.dataframe(df)

        elif choice == "Delete User":
            st.subheader("Delete a User")
            users = view_users()
            df = pd.DataFrame(users, columns=["ID", "Name", "Email", "Age"])
            st.dataframe(df)
            user_id = st.number_input("Enter ID to delete", 1)
            if st.button("Delete"):
                delete_user(user_id)
                st.warning(f"User {user_id} deleted!")

    # =====================================
    # SECTION 2: MUSHROOM CLASSIFIER
    # =====================================
    elif menu == "Mushroom Classifier 🧠":
        st.header("Binary Classification: Mushroom Edibility 🍄")
        df = load_data()
        x_train, x_test, y_train, y_test = split(df)
        class_names = ['edible', 'poisonous']
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier",
            ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest")
        )

        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01)
            kernel = st.sidebar.radio("kernel", ("rbf", "linear"))
            gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"))
            metrics = st.sidebar.multiselect("Metrics to plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify"):
                st.subheader("SVM Results")
                model = SVC(C=C, kernel=kernel, gamma=gamma)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, pos_label=1), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, pos_label=1), 2))
                plot_metrics(metrics, model, x_test, y_test, class_names)

        elif classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01)
            max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500)
            metrics = st.sidebar.multiselect("Metrics to plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify"):
                st.subheader("Logistic Regression Results")
                model = LogisticRegression(C=C, max_iter=max_iter)
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, pos_label=1), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, pos_label=1), 2))
                plot_metrics(metrics, model, x_test, y_test, class_names)

        elif classifier == 'Random Forest':
            st.sidebar.subheader("Model Hyperparameters")
            n_estimators = st.sidebar.slider("Number of trees", 100, 500, 100, step=10)
            max_depth = st.sidebar.slider("Max tree depth", 1, 20, 10, step=1)
            bootstrap = st.sidebar.radio("Bootstrap samples", ('True', 'False'))
            metrics = st.sidebar.multiselect("Metrics to plot", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

            if st.sidebar.button("Classify"):
                st.subheader("Random Forest Results")
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    bootstrap=True if bootstrap == 'True' else False,
                    random_state=0
                )
                model.fit(x_train, y_train)
                accuracy = model.score(x_test, y_test)
                y_pred = model.predict(x_test)
                st.write("Accuracy:", round(accuracy, 2))
                st.write("Precision:", round(precision_score(y_test, y_pred, pos_label=1), 2))
                st.write("Recall:", round(recall_score(y_test, y_pred, pos_label=1), 2))
                plot_metrics(metrics, model, x_test, y_test, class_names)

        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Mushroom Data Set")
            st.write(df)

if __name__ == '__main__':
    main()
