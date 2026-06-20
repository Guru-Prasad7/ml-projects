import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_score, recall_score, f1_score
)
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def generate_dataset(n_samples=1500):
    np.random.seed(42)

    study_hours = np.random.uniform(1, 10, n_samples)
    attendance = np.random.uniform(40, 100, n_samples)
    prev_grade = np.random.uniform(40, 100, n_samples)
    sleep_hours = np.random.uniform(4, 9, n_samples)
    assignments_done = np.random.uniform(50, 100, n_samples)
    extra_activities = np.random.choice([0, 1], n_samples)
    parent_education = np.random.choice(["None", "High School", "Bachelor", "Masters"], n_samples)
    internet_access = np.random.choice([0, 1], n_samples)
    gender = np.random.choice(["Male", "Female"], n_samples)
    school_type = np.random.choice(["Public", "Private"], n_samples)

    pe_score = {"None": 0, "High School": 1, "Bachelor": 2, "Masters": 3}
    pe_arr = np.array([pe_score[p] for p in parent_education])

    score = (
        0.28 * study_hours / 10
        + 0.22 * (attendance - 40) / 60
        + 0.30 * (prev_grade - 40) / 60
        + 0.06 * (sleep_hours - 4) / 5
        + 0.08 * (assignments_done - 50) / 50
        + 0.03 * pe_arr / 3
        + 0.02 * internet_access
        + 0.01 * extra_activities
        + np.random.normal(0, 0.04, n_samples)
    )

    pass_threshold = 0.48
    result = (score >= pass_threshold).astype(int)

    df = pd.DataFrame({
        "Study_Hours_Day": study_hours.round(1),
        "Attendance_Pct": attendance.round(1),
        "Previous_Grade": prev_grade.round(1),
        "Sleep_Hours": sleep_hours.round(1),
        "Assignments_Done_Pct": assignments_done.round(1),
        "Extra_Activities": extra_activities,
        "Parent_Education": parent_education,
        "Internet_Access": internet_access,
        "Gender": gender,
        "School_Type": school_type,
        "Result": result,
    })
    return df


@st.cache_resource
def train_models(df):
    le_pe = LabelEncoder()
    le_gender = LabelEncoder()
    le_school = LabelEncoder()

    df_enc = df.copy()
    df_enc["Parent_Education_enc"] = le_pe.fit_transform(df["Parent_Education"])
    df_enc["Gender_enc"] = le_gender.fit_transform(df["Gender"])
    df_enc["School_Type_enc"] = le_school.fit_transform(df["School_Type"])

    feature_cols = [
        "Study_Hours_Day", "Attendance_Pct", "Previous_Grade",
        "Sleep_Hours", "Assignments_Done_Pct", "Extra_Activities",
        "Parent_Education_enc", "Internet_Access", "Gender_enc", "School_Type_enc",
    ]

    X = df_enc[feature_cols]
    y = df_enc["Result"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=4,
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_sc, y_train)

    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train_sc, y_train)

    rf_pred = rf_model.predict(X_test_sc)
    lr_pred = lr_model.predict(X_test_sc)
    rf_prob = rf_model.predict_proba(X_test_sc)[:, 1]
    lr_prob = lr_model.predict_proba(X_test_sc)[:, 1]

    def get_metrics(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "confusion_matrix": cm,
            "fpr": fpr, "tpr": tpr, "roc_auc": roc_auc,
            "report": classification_report(y_true, y_pred, target_names=["Fail", "Pass"], output_dict=True),
        }

    rf_metrics = get_metrics(y_test, rf_pred, rf_prob)
    lr_metrics = get_metrics(y_test, lr_pred, lr_prob)

    rf_cv = cross_val_score(rf_model, scaler.transform(X), y, cv=5, scoring="accuracy")
    lr_cv = cross_val_score(lr_model, scaler.transform(X), y, cv=5, scoring="accuracy")

    feature_display = [
        "Study Hours/Day", "Attendance %", "Previous Grade",
        "Sleep Hours", "Assignments Done %", "Extra Activities",
        "Parent Education", "Internet Access", "Gender", "School Type",
    ]
    importances = dict(zip(feature_display, rf_model.feature_importances_))

    encoders = {"pe": le_pe, "gender": le_gender, "school": le_school}
    return (rf_model, lr_model, scaler, encoders, feature_cols,
            rf_metrics, lr_metrics, rf_cv, lr_cv, importances, y_test, rf_pred, lr_pred)


def predict_student(model, scaler, encoders, feature_cols,
                    study, attendance, prev_grade, sleep,
                    assignments, extra, parent_edu, internet, gender, school):
    pe_enc = encoders["pe"].transform([parent_edu])[0]
    gender_enc = encoders["gender"].transform([gender])[0]
    school_enc = encoders["school"].transform([school])[0]

    row = pd.DataFrame([[study, attendance, prev_grade, sleep,
                          assignments, extra, pe_enc, internet, gender_enc, school_enc]],
                        columns=feature_cols)
    row_sc = scaler.transform(row)
    label = model.predict(row_sc)[0]
    proba = model.predict_proba(row_sc)[0]
    return label, proba


def plot_confusion_matrix(cm, title):
    labels = ["Fail", "Pass"]
    fig = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title=title,
    )
    fig.update_layout(height=320, coloraxis_showscale=False,
                      xaxis_title="Predicted", yaxis_title="Actual")
    return fig


def main():
    st.title("🎓 Student Performance Prediction")
    st.markdown("*Random Forest & Logistic Regression — Predict if a student will Pass or Fail*")
    st.divider()

    df = generate_dataset()
    (rf_model, lr_model, scaler, encoders, feature_cols,
     rf_metrics, lr_metrics, rf_cv, lr_cv,
     importances, y_test, rf_pred, lr_pred) = train_models(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔮 Predict Student", "📊 Model Comparison", "📈 Visualizations", "📂 Dataset"
    ])

    with tab1:
        st.header("Predict Student Outcome")
        st.write("Fill in the student details below to predict **Pass** or **Fail**.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Academic Info")
            study = st.slider("Study Hours / Day", 1.0, 10.0, 5.0, 0.5)
            attendance = st.slider("Attendance (%)", 40.0, 100.0, 75.0, 1.0)
            prev_grade = st.slider("Previous Grade (%)", 40.0, 100.0, 65.0, 1.0)
            assignments = st.slider("Assignments Done (%)", 50.0, 100.0, 80.0, 1.0)

        with col2:
            st.subheader("Personal Info")
            sleep = st.slider("Sleep Hours / Day", 4.0, 9.0, 7.0, 0.5)
            extra = st.selectbox("Extra-Curricular Activities", [0, 1], format_func=lambda x: "Yes" if x else "No")
            gender = st.selectbox("Gender", ["Male", "Female"])
            school = st.selectbox("School Type", ["Public", "Private"])

        with col3:
            st.subheader("Background")
            parent_edu = st.selectbox("Parent Education", ["None", "High School", "Bachelor", "Masters"])
            internet = st.selectbox("Internet Access at Home", [0, 1], format_func=lambda x: "Yes" if x else "No")
            model_choice = st.radio("ML Model", ["Random Forest", "Logistic Regression"])
            st.write("")
            predict_btn = st.button("🎯 Predict Now", use_container_width=True, type="primary")

        st.divider()

        if predict_btn:
            model = rf_model if model_choice == "Random Forest" else lr_model
            label, proba = predict_student(
                model, scaler, encoders, feature_cols,
                study, attendance, prev_grade, sleep,
                assignments, extra, parent_edu, internet, gender, school,
            )

            res_col1, res_col2, res_col3, res_col4 = st.columns(4)
            outcome = "✅ PASS" if label == 1 else "❌ FAIL"

            with res_col1:
                st.metric("Prediction", outcome)
            with res_col2:
                st.metric("Pass Probability", f"{proba[1]*100:.1f}%")
            with res_col3:
                st.metric("Fail Probability", f"{proba[0]*100:.1f}%")
            with res_col4:
                model_acc = rf_metrics["accuracy"] if model_choice == "Random Forest" else lr_metrics["accuracy"]
                st.metric("Model Accuracy", f"{model_acc*100:.2f}%")

            if label == 1:
                st.success(f"🎉 This student is predicted to **PASS** with {proba[1]*100:.1f}% confidence using {model_choice}.")
            else:
                st.error(f"⚠️ This student is predicted to **FAIL** with {proba[0]*100:.1f}% confidence using {model_choice}. Consider increasing study hours and attendance.")

            st.subheader("Confidence Breakdown")
            fig_conf = go.Figure(go.Bar(
                x=["Fail", "Pass"],
                y=[proba[0] * 100, proba[1] * 100],
                marker_color=["#e74c3c", "#2ecc71"],
                text=[f"{proba[0]*100:.1f}%", f"{proba[1]*100:.1f}%"],
                textposition="auto",
            ))
            fig_conf.update_layout(height=300, yaxis_title="Probability (%)",
                                   xaxis_title="Outcome", showlegend=False)
            st.plotly_chart(fig_conf, use_container_width=True)

            st.subheader("Input Summary")
            summary = pd.DataFrame({
                "Parameter": ["Study Hours/Day", "Attendance (%)", "Previous Grade (%)",
                               "Sleep Hours", "Assignments Done (%)", "Extra Activities",
                               "Gender", "School Type", "Parent Education", "Internet Access"],
                "Value": [study, attendance, prev_grade, sleep, assignments,
                          "Yes" if extra else "No", gender, school,
                          parent_edu, "Yes" if internet else "No"],
            })
            st.table(summary.set_index("Parameter"))

    with tab2:
        st.header("Model Comparison — Random Forest vs Logistic Regression")

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.subheader("🌲 Random Forest")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{rf_metrics['accuracy']*100:.2f}%")
            m2.metric("Precision", f"{rf_metrics['precision']*100:.2f}%")
            m3.metric("Recall", f"{rf_metrics['recall']*100:.2f}%")
            m4.metric("F1 Score", f"{rf_metrics['f1']*100:.2f}%")

        with col_m2:
            st.subheader("📈 Logistic Regression")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{lr_metrics['accuracy']*100:.2f}%")
            m2.metric("Precision", f"{lr_metrics['precision']*100:.2f}%")
            m3.metric("Recall", f"{lr_metrics['recall']*100:.2f}%")
            m4.metric("F1 Score", f"{lr_metrics['f1']*100:.2f}%")

        st.divider()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.plotly_chart(plot_confusion_matrix(rf_metrics["confusion_matrix"], "Random Forest — Confusion Matrix"), use_container_width=True)
        with col_c2:
            st.plotly_chart(plot_confusion_matrix(lr_metrics["confusion_matrix"], "Logistic Regression — Confusion Matrix"), use_container_width=True)

        st.subheader("ROC Curve Comparison")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=rf_metrics["fpr"], y=rf_metrics["tpr"],
            name=f"Random Forest (AUC = {rf_metrics['roc_auc']:.3f})",
            line=dict(color="#2ecc71", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=lr_metrics["fpr"], y=lr_metrics["tpr"],
            name=f"Logistic Regression (AUC = {lr_metrics['roc_auc']:.3f})",
            line=dict(color="#3498db", width=2),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random Chance",
            line=dict(color="gray", dash="dash"),
        ))
        fig_roc.update_layout(height=400, xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate",
                              legend=dict(x=0.5, y=0.1))
        st.plotly_chart(fig_roc, use_container_width=True)

        col_cv1, col_cv2 = st.columns(2)
        with col_cv1:
            st.subheader("Random Forest — 5-Fold CV Accuracy")
            fig_rfcv = px.bar(
                x=[f"Fold {i+1}" for i in range(5)], y=rf_cv * 100,
                color=rf_cv * 100, color_continuous_scale="Greens",
                labels={"x": "Fold", "y": "Accuracy (%)"},
            )
            fig_rfcv.add_hline(y=rf_cv.mean()*100, line_dash="dash", line_color="red",
                               annotation_text=f"Mean: {rf_cv.mean()*100:.2f}%")
            fig_rfcv.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_rfcv, use_container_width=True)

        with col_cv2:
            st.subheader("Logistic Regression — 5-Fold CV Accuracy")
            fig_lrcv = px.bar(
                x=[f"Fold {i+1}" for i in range(5)], y=lr_cv * 100,
                color=lr_cv * 100, color_continuous_scale="Blues",
                labels={"x": "Fold", "y": "Accuracy (%)"},
            )
            fig_lrcv.add_hline(y=lr_cv.mean()*100, line_dash="dash", line_color="red",
                               annotation_text=f"Mean: {lr_cv.mean()*100:.2f}%")
            fig_lrcv.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_lrcv, use_container_width=True)

        st.subheader("Metric Comparison Summary")
        comp_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "CV Mean Accuracy"],
            "Random Forest": [
                f"{rf_metrics['accuracy']*100:.2f}%",
                f"{rf_metrics['precision']*100:.2f}%",
                f"{rf_metrics['recall']*100:.2f}%",
                f"{rf_metrics['f1']*100:.2f}%",
                f"{rf_metrics['roc_auc']:.4f}",
                f"{rf_cv.mean()*100:.2f}%",
            ],
            "Logistic Regression": [
                f"{lr_metrics['accuracy']*100:.2f}%",
                f"{lr_metrics['precision']*100:.2f}%",
                f"{lr_metrics['recall']*100:.2f}%",
                f"{lr_metrics['f1']*100:.2f}%",
                f"{lr_metrics['roc_auc']:.4f}",
                f"{lr_cv.mean()*100:.2f}%",
            ],
        })
        st.dataframe(comp_df.set_index("Metric"), use_container_width=True)

        st.subheader("Feature Importance (Random Forest)")
        imp_df = pd.DataFrame(
            {"Feature": list(importances.keys()), "Importance": list(importances.values())}
        ).sort_values("Importance", ascending=True)
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Greens",
            labels={"Importance": "Importance Score"},
        )
        fig_imp.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with tab3:
        st.header("Data Visualizations")

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.subheader("Pass / Fail Distribution")
            dist = df["Result"].value_counts().reset_index()
            dist["Label"] = dist["Result"].map({0: "Fail", 1: "Pass"})
            fig_pie = px.pie(dist, values="count", names="Label",
                             color="Label",
                             color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"})
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_v2:
            st.subheader("Study Hours vs Result")
            df_plot = df.copy()
            df_plot["Outcome"] = df_plot["Result"].map({0: "Fail", 1: "Pass"})
            fig_study = px.histogram(
                df_plot, x="Study_Hours_Day", color="Outcome",
                barmode="overlay", opacity=0.7,
                color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"},
                labels={"Study_Hours_Day": "Study Hours / Day"},
            )
            fig_study.update_layout(height=350)
            st.plotly_chart(fig_study, use_container_width=True)

        col_v3, col_v4 = st.columns(2)

        with col_v3:
            st.subheader("Attendance vs Previous Grade")
            fig_scatter = px.scatter(
                df_plot, x="Attendance_Pct", y="Previous_Grade",
                color="Outcome", opacity=0.5,
                color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"},
                labels={"Attendance_Pct": "Attendance (%)", "Previous_Grade": "Previous Grade (%)"},
            )
            fig_scatter.update_layout(height=350)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_v4:
            st.subheader("Pass Rate by Parent Education")
            pe_pass = df.groupby("Parent_Education")["Result"].mean().reset_index()
            pe_pass.columns = ["Education", "Pass Rate"]
            pe_pass["Pass Rate (%)"] = pe_pass["Pass Rate"] * 100
            edu_order = ["None", "High School", "Bachelor", "Masters"]
            pe_pass["Education"] = pd.Categorical(pe_pass["Education"], categories=edu_order, ordered=True)
            pe_pass = pe_pass.sort_values("Education")
            fig_pe = px.bar(
                pe_pass, x="Education", y="Pass Rate (%)",
                color="Pass Rate (%)", color_continuous_scale="Blues",
                labels={"Education": "Parent Education Level"},
            )
            fig_pe.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_pe, use_container_width=True)

        col_v5, col_v6 = st.columns(2)

        with col_v5:
            st.subheader("Average Study Hours by Outcome")
            avg_study = df_plot.groupby("Outcome")["Study_Hours_Day"].mean().reset_index()
            fig_avg = px.bar(
                avg_study, x="Outcome", y="Study_Hours_Day",
                color="Outcome",
                color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"},
                labels={"Study_Hours_Day": "Avg Study Hours / Day"},
            )
            fig_avg.update_layout(height=320, showlegend=False)
            st.plotly_chart(fig_avg, use_container_width=True)

        with col_v6:
            st.subheader("Correlation Heatmap")
            num_cols = ["Study_Hours_Day", "Attendance_Pct", "Previous_Grade",
                        "Sleep_Hours", "Assignments_Done_Pct", "Extra_Activities",
                        "Internet_Access", "Result"]
            corr = df[num_cols].corr()
            short_names = ["Study Hrs", "Attendance", "Prev Grade",
                           "Sleep Hrs", "Assignments", "Extra Act.",
                           "Internet", "Result"]
            corr.index = short_names
            corr.columns = short_names
            fig_heat = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
            )
            fig_heat.update_layout(height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Box Plot — Key Features by Outcome")
        box_feature = st.selectbox("Select Feature", [
            "Study_Hours_Day", "Attendance_Pct", "Previous_Grade",
            "Sleep_Hours", "Assignments_Done_Pct"
        ], format_func=lambda x: x.replace("_", " "))
        fig_box = px.box(
            df_plot, x="Outcome", y=box_feature,
            color="Outcome",
            color_discrete_map={"Pass": "#2ecc71", "Fail": "#e74c3c"},
            points="outliers",
            labels={"Outcome": "Result", box_feature: box_feature.replace("_", " ")},
        )
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with tab4:
        st.header("Dataset Explorer")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Students", len(df))
        c2.metric("Features", df.shape[1] - 1)
        c3.metric("Pass Rate", f"{df['Result'].mean()*100:.1f}%")
        c4.metric("Fail Rate", f"{(1-df['Result'].mean())*100:.1f}%")

        st.divider()

        f1, f2, f3 = st.columns(3)
        with f1:
            filter_gender = st.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
        with f2:
            filter_school = st.multiselect("School Type", df["School_Type"].unique(), default=list(df["School_Type"].unique()))
        with f3:
            filter_result = st.multiselect("Result", [0, 1], default=[0, 1], format_func=lambda x: "Pass" if x else "Fail")

        df_show = df.copy()
        df_show = df_show[
            df_show["Gender"].isin(filter_gender)
            & df_show["School_Type"].isin(filter_school)
            & df_show["Result"].isin(filter_result)
        ]
        df_show["Result"] = df_show["Result"].map({0: "Fail", 1: "Pass"})

        st.write(f"Showing **{len(df_show)}** records")
        st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=320)

        st.subheader("Statistical Summary")
        num_cols = ["Study_Hours_Day", "Attendance_Pct", "Previous_Grade",
                    "Sleep_Hours", "Assignments_Done_Pct"]
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)


if __name__ == "__main__":
    main()
