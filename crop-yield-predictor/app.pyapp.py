import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def generate_dataset(n_samples=1200):
    np.random.seed(42)
    crops = ["Wheat", "Rice", "Maize", "Barley", "Soybean"]
    seasons = ["Kharif", "Rabi", "Whole Year"]
    regions = ["North", "South", "East", "West", "Central"]
    soil_types = ["Sandy", "Clay", "Loam", "Silt", "Peaty"]

    crop_arr = np.random.choice(crops, n_samples)
    season_arr = np.random.choice(seasons, n_samples)
    region_arr = np.random.choice(regions, n_samples)
    soil_arr = np.random.choice(soil_types, n_samples)

    rainfall = np.random.uniform(200, 2000, n_samples)
    temperature = np.random.uniform(10, 45, n_samples)
    fertilizer = np.random.uniform(50, 500, n_samples)
    humidity = np.random.uniform(30, 95, n_samples)
    ph = np.random.uniform(4.5, 8.5, n_samples)

    crop_base = {"Wheat": 3.5, "Rice": 4.5, "Maize": 5.0, "Barley": 3.2, "Soybean": 2.8}
    season_mult = {"Kharif": 1.1, "Rabi": 1.0, "Whole Year": 1.05}
    soil_mult = {"Sandy": 0.85, "Clay": 0.90, "Loam": 1.15, "Silt": 1.10, "Peaty": 1.05}
    region_mult = {"North": 1.0, "South": 1.05, "East": 0.95, "West": 0.98, "Central": 1.02}

    base_yield = np.array([crop_base[c] for c in crop_arr])
    s_mult = np.array([season_mult[s] for s in season_arr])
    so_mult = np.array([soil_mult[s] for s in soil_arr])
    r_mult = np.array([region_mult[r] for r in region_arr])

    rainfall_norm = (rainfall - 200) / (2000 - 200)
    temp_factor = 1.0 - 0.015 * np.abs(temperature - 25)
    fert_factor = 0.7 + 0.6 * (fertilizer - 50) / (500 - 50)
    humidity_factor = 0.85 + 0.3 * (humidity - 30) / (95 - 30)
    ph_factor = 1.0 - 0.08 * np.abs(ph - 6.5)

    yield_val = (
        base_yield
        * s_mult
        * so_mult
        * r_mult
        * (0.5 + 0.5 * rainfall_norm)
        * temp_factor
        * fert_factor
        * humidity_factor
        * ph_factor
        + np.random.normal(0, 0.2, n_samples)
    )
    yield_val = np.clip(yield_val, 0.5, 9.0).round(2)

    df = pd.DataFrame(
        {
            "Crop_Type": crop_arr,
            "Season": season_arr,
            "Region": region_arr,
            "Soil_Type": soil_arr,
            "Rainfall_mm": rainfall.round(1),
            "Temperature_C": temperature.round(1),
            "Fertilizer_kg_ha": fertilizer.round(1),
            "Humidity_pct": humidity.round(1),
            "Soil_pH": ph.round(2),
            "Yield_ton_ha": yield_val,
        }
    )
    return df


@st.cache_resource
def train_model(df):
    le_crop = LabelEncoder()
    le_season = LabelEncoder()
    le_region = LabelEncoder()
    le_soil = LabelEncoder()

    df_enc = df.copy()
    df_enc["Crop_Type_enc"] = le_crop.fit_transform(df["Crop_Type"])
    df_enc["Season_enc"] = le_season.fit_transform(df["Season"])
    df_enc["Region_enc"] = le_region.fit_transform(df["Region"])
    df_enc["Soil_Type_enc"] = le_soil.fit_transform(df["Soil_Type"])

    feature_cols = [
        "Crop_Type_enc",
        "Season_enc",
        "Region_enc",
        "Soil_Type_enc",
        "Rainfall_mm",
        "Temperature_C",
        "Fertilizer_kg_ha",
        "Humidity_pct",
        "Soil_pH",
    ]
    X = df_enc[feature_cols]
    y = df_enc["Yield_ton_ha"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_sc, y_train)

    y_pred = model.predict(X_test_sc)

    metrics = {
        "r2": r2_score(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "mae": mean_absolute_error(y_test, y_pred),
        "accuracy_pct": r2_score(y_test, y_pred) * 100,
    }

    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="r2")

    feature_names = [
        "Crop Type",
        "Season",
        "Region",
        "Soil Type",
        "Rainfall",
        "Temperature",
        "Fertilizer",
        "Humidity",
        "Soil pH",
    ]
    importances = dict(zip(feature_names, model.feature_importances_))

    encoders = {
        "crop": le_crop,
        "season": le_season,
        "region": le_region,
        "soil": le_soil,
    }

    return model, scaler, encoders, metrics, cv_scores, importances, y_test, y_pred, feature_cols


def predict_yield(model, scaler, encoders, feature_cols, crop, season, region, soil, rainfall, temp, fert, humidity, ph):
    crop_enc = encoders["crop"].transform([crop])[0]
    season_enc = encoders["season"].transform([season])[0]
    region_enc = encoders["region"].transform([region])[0]
    soil_enc = encoders["soil"].transform([soil])[0]

    input_data = pd.DataFrame(
        [[crop_enc, season_enc, region_enc, soil_enc, rainfall, temp, fert, humidity, ph]],
        columns=feature_cols,
    )
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    return round(pred, 3)


def main():
    st.title("🌾 Crop Yield Prediction System")
    st.markdown("*Machine Learning with Random Forest — Predict crop yield based on environmental and agricultural inputs*")
    st.divider()

    df = generate_dataset()
    model, scaler, encoders, metrics, cv_scores, importances, y_test, y_pred, feature_cols = train_model(df)

    tab1, tab2, tab3 = st.tabs(["🔮 Predict Yield", "📊 Model Performance", "📂 Dataset Explorer"])

    with tab1:
        st.header("Predict Crop Yield")
        st.write("Enter the agricultural and environmental conditions below to get a yield prediction.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Crop Info")
            crop = st.selectbox("Crop Type", sorted(df["Crop_Type"].unique()))
            season = st.selectbox("Season", sorted(df["Season"].unique()))
            region = st.selectbox("Region", sorted(df["Region"].unique()))
            soil = st.selectbox("Soil Type", sorted(df["Soil_Type"].unique()))

        with col2:
            st.subheader("Climate Inputs")
            rainfall = st.slider("Rainfall (mm)", min_value=200, max_value=2000, value=900, step=10)
            temperature = st.slider("Temperature (°C)", min_value=10, max_value=45, value=25, step=1)
            humidity = st.slider("Humidity (%)", min_value=30, max_value=95, value=65, step=1)

        with col3:
            st.subheader("Soil & Fertilizer")
            fertilizer = st.slider("Fertilizer (kg/ha)", min_value=50, max_value=500, value=200, step=10)
            soil_ph = st.slider("Soil pH", min_value=4.5, max_value=8.5, value=6.5, step=0.1)

            st.subheader("")
            predict_btn = st.button("🚀 Predict Yield", use_container_width=True, type="primary")

        st.divider()

        if predict_btn:
            pred = predict_yield(
                model, scaler, encoders, feature_cols,
                crop, season, region, soil,
                rainfall, temperature, fertilizer, humidity, soil_ph,
            )

            col_res1, col_res2, col_res3, col_res4 = st.columns(4)
            with col_res1:
                st.metric("Predicted Yield", f"{pred} ton/ha", delta=None)
            with col_res2:
                avg_yield = df[df["Crop_Type"] == crop]["Yield_ton_ha"].mean()
                diff = round(pred - avg_yield, 3)
                st.metric(f"Avg {crop} Yield", f"{avg_yield:.2f} ton/ha", delta=f"{diff:+.3f} vs average")
            with col_res3:
                st.metric("Model R² Score", f"{metrics['r2']:.4f}")
            with col_res4:
                if pred >= avg_yield * 1.1:
                    rating = "Above Average 🟢"
                elif pred >= avg_yield * 0.9:
                    rating = "Average 🟡"
                else:
                    rating = "Below Average 🔴"
                st.metric("Yield Rating", rating)

            st.success(f"**Prediction Complete!** Expected yield for {crop} under the given conditions: **{pred} tons per hectare**")

            st.subheader("Input Summary")
            summary_df = pd.DataFrame({
                "Parameter": ["Crop Type", "Season", "Region", "Soil Type", "Rainfall (mm)", "Temperature (°C)", "Fertilizer (kg/ha)", "Humidity (%)", "Soil pH"],
                "Value": [crop, season, region, soil, rainfall, temperature, fertilizer, humidity, soil_ph],
            })
            st.table(summary_df.set_index("Parameter"))

    with tab2:
        st.header("Model Performance — Random Forest Regressor")

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("R² Score (Accuracy)", f"{metrics['r2']:.4f}", f"{metrics['accuracy_pct']:.2f}%")
        col_m2.metric("RMSE", f"{metrics['rmse']:.4f} ton/ha")
        col_m3.metric("MAE", f"{metrics['mae']:.4f} ton/ha")
        col_m4.metric("CV Mean R² (5-fold)", f"{cv_scores.mean():.4f}", f"±{cv_scores.std():.4f}")

        st.divider()

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.subheader("Actual vs Predicted Yield")
            fig_scatter = px.scatter(
                x=y_test, y=y_pred,
                labels={"x": "Actual Yield (ton/ha)", "y": "Predicted Yield (ton/ha)"},
                opacity=0.6,
                color_discrete_sequence=["#2ecc71"],
            )
            min_val = min(y_test.min(), min(y_pred))
            max_val = max(y_test.max(), max(y_pred))
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines", name="Perfect Fit",
                line=dict(color="red", dash="dash", width=2),
            ))
            fig_scatter.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_p2:
            st.subheader("Feature Importance")
            imp_df = pd.DataFrame(
                {"Feature": list(importances.keys()), "Importance": list(importances.values())}
            ).sort_values("Importance", ascending=True)
            fig_imp = px.bar(
                imp_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Greens",
                labels={"Importance": "Feature Importance Score"},
            )
            fig_imp.update_layout(height=400, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)

        col_p3, col_p4 = st.columns(2)

        with col_p3:
            st.subheader("Prediction Error Distribution")
            errors = np.array(y_pred) - np.array(y_test)
            fig_err = px.histogram(
                x=errors, nbins=40,
                labels={"x": "Prediction Error (ton/ha)", "y": "Count"},
                color_discrete_sequence=["#3498db"],
            )
            fig_err.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
            fig_err.update_layout(height=350)
            st.plotly_chart(fig_err, use_container_width=True)

        with col_p4:
            st.subheader("5-Fold Cross-Validation R² Scores")
            fig_cv = px.bar(
                x=[f"Fold {i+1}" for i in range(5)],
                y=cv_scores,
                labels={"x": "Fold", "y": "R² Score"},
                color=cv_scores,
                color_continuous_scale="Blues",
            )
            fig_cv.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {cv_scores.mean():.4f}")
            fig_cv.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_cv, use_container_width=True)

        st.subheader("Model Configuration")
        config_col1, config_col2 = st.columns(2)
        with config_col1:
            st.info("""
**Algorithm:** Random Forest Regressor  
**Estimators (Trees):** 150  
**Max Depth:** 12  
**Min Samples Split:** 4  
**Min Samples Leaf:** 2  
**Features:** 9 input features  
**Train / Test Split:** 80% / 20%
            """)
        with config_col2:
            st.info("""
**Preprocessing:**
- Label Encoding for categorical features (Crop, Season, Region, Soil)
- Standard Scaler (zero mean, unit variance) for numerical features

**Validation:**
- Hold-out test set (20%)
- 5-Fold Stratified Cross-Validation
            """)

    with tab3:
        st.header("Dataset Explorer")

        col_ds1, col_ds2, col_ds3 = st.columns(3)
        col_ds1.metric("Total Samples", len(df))
        col_ds2.metric("Features", df.shape[1] - 1)
        col_ds3.metric("Avg Yield", f"{df['Yield_ton_ha'].mean():.2f} ton/ha")

        st.divider()

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_crop = st.multiselect("Filter by Crop", df["Crop_Type"].unique(), default=list(df["Crop_Type"].unique()))
        with col_f2:
            filter_season = st.multiselect("Filter by Season", df["Season"].unique(), default=list(df["Season"].unique()))
        with col_f3:
            filter_region = st.multiselect("Filter by Region", df["Region"].unique(), default=list(df["Region"].unique()))

        mask = (
            df["Crop_Type"].isin(filter_crop)
            & df["Season"].isin(filter_season)
            & df["Region"].isin(filter_region)
        )
        df_filtered = df[mask]

        st.write(f"Showing **{len(df_filtered)}** records")
        st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True, height=300)

        st.divider()

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.subheader("Average Yield by Crop Type")
            crop_avg = df_filtered.groupby("Crop_Type")["Yield_ton_ha"].mean().reset_index()
            fig_crop = px.bar(
                crop_avg, x="Crop_Type", y="Yield_ton_ha",
                color="Yield_ton_ha", color_continuous_scale="Greens",
                labels={"Crop_Type": "Crop", "Yield_ton_ha": "Avg Yield (ton/ha)"},
            )
            fig_crop.update_layout(height=350, coloraxis_showscale=False)
            st.plotly_chart(fig_crop, use_container_width=True)

        with col_v2:
            st.subheader("Yield Distribution by Season")
            fig_box = px.box(
                df_filtered, x="Season", y="Yield_ton_ha",
                color="Season",
                labels={"Yield_ton_ha": "Yield (ton/ha)"},
            )
            fig_box.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        col_v3, col_v4 = st.columns(2)

        with col_v3:
            st.subheader("Rainfall vs Yield")
            fig_rain = px.scatter(
                df_filtered, x="Rainfall_mm", y="Yield_ton_ha",
                color="Crop_Type", opacity=0.5,
                labels={"Rainfall_mm": "Rainfall (mm)", "Yield_ton_ha": "Yield (ton/ha)"},
            )
            fig_rain.update_layout(height=350)
            st.plotly_chart(fig_rain, use_container_width=True)

        with col_v4:
            st.subheader("Temperature vs Yield")
            fig_temp = px.scatter(
                df_filtered, x="Temperature_C", y="Yield_ton_ha",
                color="Crop_Type", opacity=0.5,
                labels={"Temperature_C": "Temperature (°C)", "Yield_ton_ha": "Yield (ton/ha)"},
            )
            fig_temp.update_layout(height=350)
            st.plotly_chart(fig_temp, use_container_width=True)

        st.subheader("Correlation Heatmap")
        num_cols = ["Rainfall_mm", "Temperature_C", "Fertilizer_kg_ha", "Humidity_pct", "Soil_pH", "Yield_ton_ha"]
        corr = df_filtered[num_cols].corr()
        fig_heat = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            labels={"color": "Correlation"},
            zmin=-1, zmax=1,
        )
        fig_heat.update_layout(height=400)
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Statistical Summary")
        st.dataframe(df_filtered[num_cols].describe().round(3), use_container_width=True)


if __name__ == "__main__":
    main()
