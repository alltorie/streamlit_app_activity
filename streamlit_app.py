import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt   
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score


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
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or poisonous? 🍄")

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
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_svm')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel_svm')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma_svm')

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'),
            key="metrics_svm"
        )

        if st.sidebar.button("Classify", key='classify_svm'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, pos_label=1), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, pos_label=1), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_lr')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter_lr')

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'),
            key="metrics_lr"
        )

        if st.sidebar.button("Classify", key='classify_lr'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, pos_label=1), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, pos_label=1), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.slider("Number of trees in the forest", 100, 500, 100, step=10, key='n_estimators_rf')
        max_depth = st.sidebar.slider("The maximum depth of the tree", 1, 20, 10, step=1, key='max_depth_rf')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", (True, False), key='bootstrap_rf')

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'),
            key="metrics_rf"
        )

        if st.sidebar.button("Classify", key='classify_rf'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                bootstrap=bootstrap,
                n_jobs=-1,
                random_state=0
            )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, pos_label=1), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, pos_label=1), 2))
            plot_metrics(metrics, model, x_test, y_test, class_names)

    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set Classification")
        st.write(df)


if __name__ == '__main__':
    main()
