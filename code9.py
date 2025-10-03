# app.py — revised, improved Streamlit credit-card-fraud demo
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score, average_precision_score
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st

@st.cache_data(persist=True)
def load_data(file_path):
    """
    Load CSV data and encode object columns to numeric.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed dataframe.
    """
    data = pd.read_csv(file_path)
    
    # Encode categorical object columns if any
    labelencoder = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = labelencoder.fit_transform(data[col])
    
    return data

# -----------------------------
# Metrics plotting
# -----------------------------
# -----------------------------
# Metrics plotting helpers
# -----------------------------
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
import streamlit as st

def st_plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    st.pyplot(fig)
    plt.close(fig)

def st_plot_roc_from_estimator(estimator, x_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_estimator(estimator, x_test, y_test, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

def st_plot_pr_from_estimator(estimator, x_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_estimator(estimator, x_test, y_test, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

def st_plot_roc_from_predictions(y_test, y_score):
    fig, ax = plt.subplots(figsize=(6, 4))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

def st_plot_pr_from_predictions(y_test, y_score):
    fig, ax = plt.subplots(figsize=(6, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Convenience wrapper for mixed metrics
# -----------------------------
def plot_metrics(metrics_list, model, x_test, y_test, y_score=None, class_names=["Non-Fraud", "Fraud"], is_nn=False):
    """
    Display selected metrics in Streamlit.
    - For neural networks, provide y_score (probabilities) and set is_nn=True.
    """
    if "Confusion Matrix" in metrics_list:
        cm = confusion_matrix(y_test, model.predict(x_test) if not is_nn else (y_score > 0.5).astype(int))
        st.subheader("Confusion Matrix")
        st_plot_confusion(cm, class_names)

    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        if is_nn:
            st_plot_roc_from_predictions(y_test, y_score)
        else:
            st_plot_roc_from_estimator(model, x_test, y_test)

    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        if is_nn:
            st_plot_pr_from_predictions(y_test, y_score)
        else:
            st_plot_pr_from_estimator(model, x_test, y_test)
def fraud_app():
    st.title("Credit Card Fraud Detection — Demo")
    st.sidebar.title("Settings")

    # Sidebar inputs
    data_path = st.sidebar.text_input(
        "CSV Path",
        value=r"C:\Users\eduv4822223\Downloads\New folder\New folder\creditcard.csv"
    )

    test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, 0.2, 0.05)
    random_state = int(st.sidebar.number_input("Random seed", 0, 9999, value=42))
    stratify_split = st.sidebar.checkbox("Stratify split by class (recommended)", value=True)

    classifier = st.sidebar.selectbox(
        "Classifier",
        ("Random Forest", "Support Vector Machine (SVM)", "Logistic Regression", "Shallow Neural Network")
    )

    metrics = st.sidebar.multiselect(
        "Metrics to plot",
        ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve")
    )

    # Hyperparameters
    if classifier == "Random Forest":
        n_estimators = st.sidebar.number_input("n_estimators", 10, 1000, value=100, step=10)
        max_depth = st.sidebar.number_input("max_depth (0 for None)", 0, 200, value=10, step=1)
    elif classifier == "Logistic Regression":
        C = st.sidebar.number_input("C (inverse regularization)", 1e-4, 100.0, value=1.0, step=0.1, format="%.4f")
        max_iter = st.sidebar.number_input("max_iter", 50, 1000, value=200, step=10)
    elif classifier == "Support Vector Machine (SVM)":
        C_svm = st.sidebar.number_input("C (regularization)", 1e-4, 100.0, value=1.0, step=0.1, format="%.4f")
        kernel = st.sidebar.selectbox("kernel", ("rbf", "linear"))
        gamma = st.sidebar.selectbox("gamma", ("scale", "auto"))
        scale_before = st.sidebar.checkbox("Apply StandardScaler (recommended for SVM)", True)
    elif classifier == "Shallow Neural Network":
        hidden_units = st.sidebar.number_input("Hidden layer units", 2, 512, value=16, step=2)
        epochs = st.sidebar.slider("Epochs", 1, 200, value=10)

    train_button = st.sidebar.button("Train model")

    # Load data
    try:
        with st.spinner("Loading data..."):
            data = load_data(data_path)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return

    st.subheader("Data preview")
    st.write(data.head())
    st.write(data.describe())

    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Split
    stratify = y if stratify_split else None
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    st.sidebar.markdown("### Class distribution (train)")
    st.sidebar.write(y_train.value_counts())

    if train_button:
        st.info(f"Training {classifier}...")

        # Scaling for some models
        scaler = None
        if classifier in ("Support Vector Machine (SVM)", "Logistic Regression", "Shallow Neural Network"):
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            x_test_scaled = scaler.transform(x_test)
        else:
            x_train_scaled, x_test_scaled = x_train.values, x_test.values

        # Train models
        if classifier == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=(None if max_depth == 0 else int(max_depth)),
                random_state=random_state,
                n_jobs=-1
            )
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            y_score = model.predict_proba(x_test_scaled)[:, 1]

        elif classifier == "Logistic Regression":
            model = LogisticRegression(C=C, max_iter=int(max_iter), class_weight='balanced', n_jobs=-1)
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_test_scaled)
            y_score = model.predict_proba(x_test_scaled)[:, 1]

        elif classifier == "Support Vector Machine (SVM)":
            model = SVC(C=C_svm, kernel=kernel, gamma=gamma, probability=True, class_weight='balanced')
            x_train_input = x_train_scaled if scale_before else x_train.values
            x_test_input = x_test_scaled if scale_before else x_test.values
            model.fit(x_train_input, y_train)
            y_pred = model.predict(x_test_input)
            y_score = model.predict_proba(x_test_input)[:, 1]

        elif classifier == "Shallow Neural Network":
            model = Sequential([
                InputLayer(input_shape=(x_train_scaled.shape[1],)),
                Dense(int(hidden_units), activation='relu'),
                BatchNormalization(),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            checkpoint = ModelCheckpoint('shallow_nn.keras', save_best_only=True)
            with st.spinner("Training neural network..."):
                model.fit(x_train_scaled, y_train, validation_split=0.1, epochs=int(epochs), callbacks=[checkpoint], verbose=0)
            y_pred_prob = model.predict(x_test_scaled).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)
            y_score = y_pred_prob

        # Compute metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_score)
        avg_prec = average_precision_score(y_test, y_score)

        st.subheader(f"{classifier} results")
        st.write(f"Accuracy: **{acc:.4f}**")
        st.write(f"Precision: **{prec:.4f}**")
        st.write(f"Recall: **{rec:.4f}**")
        st.write(f"F1-score: **{f1:.4f}**")
        st.write(f"ROC AUC: **{roc_auc:.4f}**")
        st.write(f"Average Precision (PR AUC): **{avg_prec:.4f}**")

        # Plot metrics using helper
        is_nn = classifier == "Shallow Neural Network"
        plot_metrics(metrics, model, x_test_scaled if scaler is not None else x_test.values, y_test, y_score=y_score, is_nn=is_nn)

        # Feature distribution explainer
        st.subheader("Feature Distributions by Class (Explainer)")
        balanced_df = x_train.copy()
        balanced_df['Class'] = y_train
        balanced_df_sample = balanced_df.sample(min(10000, len(balanced_df)), random_state=42)

        for col in balanced_df_sample.columns:
            fig = px.histogram(
                balanced_df_sample,
                x=col,
                color='Class',
                barmode='overlay',
                title=f'Feature: {col}',
                width=640,
                height=400,
                labels={'Class': 'Fraud (1) / Non-Fraud (0)'}
            )
            st.plotly_chart(fig)
    
        st.subheader("Feature Boxplots by Class")
        
        for col in balanced_df_sample.columns[:-1]:  # exclude 'Class'
            fig = px.box(
                balanced_df_sample,
                x="Class",
                y=col,
                points="outliers",  # shows outliers explicitly
                title=f'Boxplot of {col} by Class',
                width=640,
                height=400,
                labels={'Class': 'Fraud (1) / Non-Fraud (0)'}
            )
            st.plotly_chart(fig)
        st.subheader("Scatterplots to Detect Extreme Values")
        
        if "Amount" in balanced_df_sample.columns:
            fig_amount = px.scatter(
                balanced_df_sample,
                x=balanced_df_sample.index,
                y="Amount",
                color="Class",
                title="Transaction Amount Scatterplot",
                labels={"Class": "Fraud (1) / Non-Fraud (0)", "index": "Transaction Index"},
                width=800,
                height=500
            )
            st.plotly_chart(fig_amount)
        
        if "Time" in balanced_df_sample.columns:
            fig_time = px.scatter(
                balanced_df_sample,
                x="Time",
                y="Amount",
                color="Class",
                title="Transaction Amount vs Time Scatterplot",
                labels={"Class": "Fraud (1) / Non-Fraud (0)", "Time": "Transaction Time", "Amount": "Transaction Amount"},
                width=800,
                height=500
            )
            st.plotly_chart(fig_time)    
import streamlit as st
import numpy as np
import joblib

# ----------------------
# Load model & scaler
# ----------------------
  # make sure the filename matches
import streamlit as st
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu
import joblib
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu
from PIL import Image
# ----------------------
# Transaction predictor
# ----------------------
def transaction_predictor():
    import streamlit as st
    import numpy as np
    import joblib
    from sklearn.ensemble import IsolationForest

    # Load trained model and scaler
    iso_forest = joblib.load("isolation_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")  # scaler trained on all 30 features

    st.title("💳 Transaction Fraud Prediction")
    st.markdown("Enter transaction details below to predict fraud:")

    st.divider()

    # ----------------------
    # PCA Features (V1 - V28)
    # ----------------------
    st.subheader("⚙️ PCA Features (V1 - V28)")
    pca_features = []
    for i in range(1, 29):
        value = st.slider(f"V{i}", min_value=-20.0, max_value=20.0, value=0.0, step=0.01)
        pca_features.append(value)

    # ----------------------
    # Transaction Details
    # ----------------------
    st.subheader("⏱ Transaction Details")
    time_input = st.number_input("Transaction Time", min_value=0.0, value=0.0, step=1.0)
    amount_input = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=1.0)

    # Combine all features
    features = np.array(pca_features + [time_input, amount_input]).reshape(1, -1)

    # Scale all features (must match scaler training)
    features_scaled = scaler.transform(features)

    # Predict button
    if st.button("🔮 Predict Fraud"):
        pred = iso_forest.predict(features_scaled)  # 1 = normal, -1 = anomaly
        label = "Fraud" if pred[0] == -1 else "Not Fraud"
        st.success(f"Prediction: **{label}**")
    else:
        st.info("Adjust features and click the button to check for fraud.")

def main():
    # Sidebar Menu with nice design
    with st.sidebar:
        choice = option_menu(
            menu_title="Navigation",
            options=["🏠 Welcome Page", "📊 Fraud Detection App", "🧪 Transaction Predictor", "ℹ️ About"],
            icons=["house", "bar-chart", "info-circle"],
            menu_icon="list",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#f0f2f6"},
                "icon": {"color": "#000000", "font-size": "20px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px"},
                "nav-link-selected": {"background-color": "#000000", "color": "white"},
            }
        )

    # Welcome Page
    if choice == "🏠 Welcome Page":
        st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚀 Welcome to the Fraud Detection System</h1>", unsafe_allow_html=True)
        
        # Display a larger central image
        st.image(
            "https://techie-buzz.com/wp-content/uploads/2023/05/AI-Fraud-Detection.jpg",
            use_column_width=True
        )
    
        st.markdown("---")
        
        # Features and How to use in two columns
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("✨ Features")
            st.markdown("""
            - Multiple ML models (Random Forest, Logistic Regression, SVM, Neural Network)  
            - Performance metrics (Confusion Matrix, ROC Curve, Precision-Recall Curve)  
            - User-friendly interface for data science experimentation  
            - Interactive feature visualizations
            """)
    
        with col2:
            st.subheader("🛠 How to use")
            st.markdown("""
            1. Go to the **Fraud Detection App** tab  
            2. Choose a classifier  
            3. Adjust hyperparameters if needed  
            4. Train and evaluate the model  
            5. Explore feature distributions and metrics
            """)
    
        st.markdown("---")
    
        # Interactive buttons for quick navigation
        st.markdown("<h3 style='text-align: center; color: #2E86C1;'>Quick Navigation</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
    
        with col1:
            if st.button("Go to Fraud Detection App"):
                st.session_state['page'] = "📊 Fraud Detection App"
    
        with col2:
            if st.button("View Dataset Info"):
                st.write("You can view your dataset and statistics in the **Fraud Detection App** tab after uploading.")
    
        with col3:
            if st.button("About this App"):
                st.session_state['page'] = "ℹ️ About"

    # Fraud Detection App
    elif choice == "📊 Fraud Detection App":
        fraud_app()

    elif choice == "🧪 Transaction Predictor":
        transaction_predictor()

    elif choice == "ℹ️ About":
        st.title("ℹ️ About This App")
        st.markdown("<h1 style='color: #2E86C1;'>ℹ️ About This App</h1>", unsafe_allow_html=True)
        st.write("""
        Created for a Data Science project on **Credit Card Fraud Detection**.
        
        **👨‍💻 Developer:** Kelvin Kgarudi, Dylan Badenhorst, Letlhogonolo Karabo Matlhoi, Moshoeshoe Mokhachane  
        **🖥 Technologies:** Python, Streamlit, scikit-learn  
        **📊 Dataset:** Kaggle Credit Card Fraud Dataset
        """)

if __name__ == '__main__':
    main()
