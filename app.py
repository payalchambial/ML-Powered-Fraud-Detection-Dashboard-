
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_model.joblib"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Fraud Detection System")
st.write("Machine Learning Based Transaction Risk Analysis")
st.warning(
    "Educational demo only. A fraud prediction is not proof of fraud. "
    "Do not upload real banking or personal data."
)

if not MODEL_PATH.exists():
    st.error("Model not found. Please run train_model.py first.")
    st.code("python train_model.py")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
features = list(model.feature_names_in_)

if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
else:
    metrics = {}

tab1, tab2, tab3 = st.tabs([
    "Single Prediction",
    "Bulk CSV Analysis",
    "Model Performance"
])

# TAB 1: Single transaction
with tab1:
    st.subheader("Analyze One Transaction")
    st.caption(
        "Enter feature values. For meaningful testing, use a row "
        "from the downloaded dataset rather than arbitrary values."
    )

    with st.form("transaction_form"):
        values = {}
        columns = st.columns(3)

        for i, feature in enumerate(features):
            with columns[i % 3]:
                values[feature] = st.number_input(
                    feature,
                    value=0.0,
                    format="%.6f"
                )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            type="primary"
        )

    if submitted:
        row = pd.DataFrame([values], columns=features)
        probability = float(model.predict_proba(row)[0][1])
        prediction = int(model.predict(row)[0])

        col1, col2 = st.columns(2)
        col1.metric(
            "Estimated Fraud Probability",
            f"{probability * 100:.2f}%"
        )
        col2.metric(
            "Model Prediction",
            "Potential Fraud" if prediction == 1 else "Normal"
        )

        if prediction == 1:
            st.error(
                "⚠️ Flagged for further review. "
                "This does not prove that fraud occurred."
            )
        else:
            st.success(
                "No fraud flag from this model. "
                "This does not guarantee the transaction is safe."
            )

# TAB 2: CSV analysis
with tab2:
    st.subheader("Analyze Multiple Transactions")
    st.write(
        "Upload a CSV containing the same feature columns as the "
        "training dataset. The Class column is optional."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            # Remove labels if present; don't use them as model inputs.
            if "Class" in data.columns:
                data = data.drop(columns=["Class"])

            missing = [col for col in features if col not in data.columns]

            if missing:
                st.error(
                    "Missing required columns: " + ", ".join(missing)
                )
            else:
                X = data[features].apply(
                    pd.to_numeric, errors="coerce"
                )

                probabilities = model.predict_proba(X)[:, 1]
                predictions = model.predict(X)

                result = data.copy()
                result["fraud_probability_percent"] = (
                    probabilities * 100
                ).round(2)
                result["prediction"] = [
                    "Potential Fraud" if p == 1 else "Normal"
                    for p in predictions
                ]

                c1, c2, c3 = st.columns(3)
                c1.metric("Transactions", f"{len(result):,}")
                c2.metric(
                    "Flagged Transactions",
                    f"{int((predictions == 1).sum()):,}"
                )
                c3.metric(
                    "Flag Rate",
                    f"{(predictions.mean() * 100):.2f}%"
                    if len(predictions) else "0.00%"
                )

                st.dataframe(
                    result.head(500),
                    use_container_width=True
                )

                if len(result):
                    chart_data = result["prediction"].value_counts(
                    ).reset_index()
                    chart_data.columns = ["Prediction", "Count"]

                    fig = px.pie(
                        chart_data,
                        names="Prediction",
                        values="Count",
                        title="Transaction Prediction Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.download_button(
                    "Download Prediction Report",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="fraud_prediction_report.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Could not process the file: {e}")

# TAB 3: Evaluation
with tab3:
    st.subheader("Model Performance")

    if not metrics:
        st.info("Run train_model.py to generate evaluation metrics.")
    else:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Precision",
            f"{metrics.get('precision', 0):.3f}"
        )
        c2.metric(
            "Recall",
            f"{metrics.get('recall', 0):.3f}"
        )
        c3.metric(
            "F1 Score",
            f"{metrics.get('f1_score', 0):.3f}"
        )
        c4.metric(
            "Average Precision",
            f"{metrics.get('average_precision', 0):.3f}"
        )

        st.caption(
            f"Dataset transactions: "
            f"{metrics.get('total_transactions', 0):,} | "
            f"Fraud-labelled cases: {metrics.get('fraud_cases', 0):,}"
        )

        cm = metrics.get("confusion_matrix")

        if cm:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Normal", "Actual Fraud"],
                columns=["Predicted Normal", "Predicted Fraud"]
            )

            fig = px.imshow(
                cm_df,
                text_auto=True,
                aspect="auto",
                title="Confusion Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)

        report = metrics.get("classification_report")

        if report:
            st.write("Classification Report")
            st.dataframe(
                pd.DataFrame(report).T,
                use_container_width=True
            )

        st.caption(
            "Metrics are calculated on a held-out test set. "
            "Fraud datasets are imbalanced, so accuracy alone "
            "is not enough to evaluate the model."
        )
#& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m streamlit run app.py
#& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -c "import pandas as pd; df=pd.read_csv('Data/creditcard.csv'); print('Total Rows:',df.shape[0]); print('Total Columns:',df.shape[1]); print('Column Names:',df.columns.tolist())"
#& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -c "import pandas as pd; df=pd.read_csv('Data/creditcard.csv'); print(df['Class'].value_counts())"