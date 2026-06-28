import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
    precision_score, recall_score
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Population Growth Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Census Tract Population Growth Predictor")
st.caption("East North Central Region · INFO 648 Final Project")

# ── GitHub raw URLs ───────────────────────────────────────────────────────────
BASE = "https://raw.githubusercontent.com/meganjohnson0522/INFO648/53627df6036465f67c62420780b1148b37d3d3e2/Final/StreamLit"
URLS = {
    "student":  f"{BASE}/student_tracts_raw.csv",
    "forecast": f"{BASE}/forecast_tracts_2020.csv",
    "region":   f"{BASE}/region_tract_counts.csv",
    "dict":     f"{BASE}/data_dictionary.csv",
}

@st.cache_data
def load_data():
    return (
        pd.read_csv(URLS["student"]),
        pd.read_csv(URLS["forecast"]),
        pd.read_csv(URLS["region"]),
        pd.read_csv(URLS["dict"]),
    )

# ── Sidebar: model settings only ─────────────────────────────────────────────
with st.sidebar:
    st.subheader("Model Settings")
    n_clusters   = st.slider("K-Means Clusters", 2, 10, 4)
    test_size    = st.slider("Test Split %", 10, 40, 25) / 100
    n_estimators = st.slider("RF Trees", 50, 300, 100, step=50)

# ── Helpers ───────────────────────────────────────────────────────────────────
EAST_NORTH_CENTRAL = ["Illinois", "Indiana", "Michigan", "Ohio", "Wisconsin"]

def engineer_features(df, year_suffix):
    df = df.copy()
    occ_col    = f"housing_occupied_{year_suffix}"
    vac_col    = f"housing_vacant_{year_suffix}"
    own_col    = f"housing_owner_occ_{year_suffix}"
    rent_col   = f"housing_renter_occ_{year_suffix}"
    pop_col    = f"pop_total_{year_suffix}"

    df["housing_total"] = df[occ_col] + df[vac_col]
    df["pct_vacant"]    = df[vac_col]  / df["housing_total"]
    df["pct_owner_occ"] = df[own_col]  / df["housing_total"]
    df["pct_renter_occ"]= df[rent_col] / df["housing_total"]

    age_cols  = [c for c in df.columns if c.startswith("age_")  and c.endswith(f"_{year_suffix}")]
    race_cols = [c for c in df.columns if c.startswith("race_") and c.endswith(f"_{year_suffix}")]

    for col in age_cols:
        df[f"pct_{col[4:-(len(year_suffix)+1)]}"] = df[col] / df[pop_col]
    for col in race_cols:
        df[f"pct_{col[5:-(len(year_suffix)+1)]}"] = df[col] / df[pop_col]

    drop = ([occ_col, vac_col, own_col, rent_col, "housing_total"]
            + age_cols + race_cols)
    df = df.drop(columns=[c for c in drop if c in df.columns])
    return df

# ── Load data from GitHub ─────────────────────────────────────────────────────
with st.spinner("Loading data from GitHub …"):
    student_tracts, forecast_tracts, region_counts, data_dict = load_data()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📁 Raw Data",
    "🔍 EDA",
    "🔵 Clustering",
    "🤖 Models",
    "📈 Forecast",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · Raw Data
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Loaded Datasets")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**student_tracts_raw** — {student_tracts.shape[0]:,} rows × {student_tracts.shape[1]} cols")
        st.dataframe(student_tracts.head(), use_container_width=True)
        st.markdown(f"**region_tract_counts** — {region_counts.shape[0]:,} rows × {region_counts.shape[1]} cols")
        st.dataframe(region_counts.head(10), use_container_width=True)
    with col2:
        st.markdown(f"**forecast_tracts_2020** — {forecast_tracts.shape[0]:,} rows × {forecast_tracts.shape[1]} cols")
        st.dataframe(forecast_tracts.head(), use_container_width=True)
        st.markdown(f"**data_dictionary** — {data_dict.shape[0]:,} rows × {data_dict.shape[1]} cols")
        st.dataframe(data_dict.head(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA PREP (shared across tabs)
# ══════════════════════════════════════════════════════════════════════════════
raw_rows = len(student_tracts)
s_clean = student_tracts.copy()
s_clean = s_clean[s_clean["pop_total_2010"] >= 100]
s_clean = s_clean[s_clean["land_area_sqkm"]  > 0]
s_clean = s_clean[s_clean["STATE"].isin(EAST_NORTH_CENTRAL)]

s_clean["growth_pct"] = (
    (s_clean["pop_total_2020"] - s_clean["pop_total_2010"])
    / s_clean["pop_total_2010"]
) * 100
s_clean["grew"] = (s_clean["growth_pct"] > 0).astype(int)

features = engineer_features(s_clean, "2010")

X = features.drop(columns=[c for c in
    ["GISJOIN","TRACTA","COUNTYA","STATEA","COUNTY",
     "pop_total_2020","growth_pct","grew"] if c in features.columns])
y = features["grew"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
cat_cols_base = [c for c in ["settlement_type","STATE"] if c in X_train.columns]

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Exploratory Data Analysis")

    st.markdown(f"""
    | Step | Count |
    |------|-------|
    | Original rows | {raw_rows:,} |
    | After cleaning | {len(s_clean):,} |
    | Rows removed | {raw_rows - len(s_clean):,} |
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Average Growth % by Settlement Type**")
        avg_growth = s_clean.groupby("settlement_type")["growth_pct"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=avg_growth, x="settlement_type", y="growth_pct",
                    palette="viridis", ax=ax)
        ax.set_xlabel("Settlement Type"); ax.set_ylabel("Avg Growth %")
        ax.set_title("Average Growth % by Settlement Type")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Distribution of Settlement Types**")
        fig, ax = plt.subplots(figsize=(6, 4))
        order = s_clean["settlement_type"].value_counts().index
        sns.countplot(data=s_clean, x="settlement_type", hue="settlement_type",
                      palette="viridis", legend=False, order=order, ax=ax)
        ax.set_xlabel("Settlement Type"); ax.set_ylabel("# Tracts")
        ax.set_title("Settlement Type Distribution (Cleaned)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.markdown("**Class balance (grew = 1 → population increased)**")
    st.bar_chart(y_train.value_counts().rename({0: "Did not grow", 1: "Grew"}))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · Clustering
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader(f"K-Means Clustering  (k = {n_clusters})")

    cluster_preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  MinMaxScaler()),
        ]), numeric_cols)
    ])

    with st.spinner("Running elbow analysis …"):
        distortions = []
        for i in range(1, 11):
            pl = Pipeline([
                ("prep",    cluster_preprocessor),
                ("cluster", KMeans(n_clusters=i, random_state=42, n_init=10)),
            ])
            pl.fit(X_train[numeric_cols])
            distortions.append(pl.named_steps["cluster"].inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, 11), distortions, marker="o")
    ax.axvline(n_clusters, color="red", linestyle="--", alpha=0.6,
               label=f"Selected k={n_clusters}")
    ax.set_xlabel("Number of Clusters"); ax.set_ylabel("Distortion")
    ax.set_title("Elbow Curve — East North Central Tracts")
    ax.legend(); plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Fit final cluster pipeline
    cluster_pipeline = Pipeline([
        ("prep",    cluster_preprocessor),
        ("cluster", KMeans(n_clusters=n_clusters, random_state=42, n_init=20)),
    ])
    cluster_pipeline.fit(X_train[numeric_cols])

    X_train = X_train.copy()
    X_test  = X_test.copy()
    X_train["cluster"] = cluster_pipeline.named_steps["cluster"].labels_
    X_test["cluster"]  = cluster_pipeline.predict(X_test[numeric_cols])

    st.markdown("**Cluster sizes (train)**")
    st.bar_chart(X_train.groupby("cluster").size().rename("Tract count"))

    # Centroids table
    scaler     = cluster_pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scale"]
    centroids  = cluster_pipeline.named_steps["cluster"].cluster_centers_
    cent_orig  = scaler.inverse_transform(centroids)
    cent_df    = pd.DataFrame(cent_orig, columns=numeric_cols)

    with st.expander("View centroid values (original scale)"):
        st.dataframe(cent_df.T.rename(columns=lambda i: f"Cluster {i}").style.format("{:.3f}"),
                     use_container_width=True)

    st.markdown("""
**Cluster Interpretation**

| Cluster | Label | Key Characteristics |
|---------|-------|---------------------|
| 0 | Rural/Suburban Owners | Large land area, low density, high owner-occupancy |
| 1 | Dense Urban Renters | Small land area, high density, more renters |
| 2 | Low Pop / High Vacancy | Very small area, highest vacancy rates |
| 3 | Very Dense, Diverse Urban | Extreme density, high renter share, diverse race mix |
""")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · Models
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Classification Models")

    cat_cols_full = [c for c in ["settlement_type","STATE","cluster"] if c in X_train.columns]

    full_preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_full),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  MinMaxScaler()),
        ]), numeric_cols),
    ])

    with st.spinner("Training Logistic Regression …"):
        model_lr = Pipeline([
            ("preprocess", full_preprocessor),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)

    with st.spinner("Training Random Forest …"):
        model_rf = Pipeline([
            ("preprocess", full_preprocessor),
            ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=42)),
        ])
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)

    # ── Ablation (without cluster) ────────────────────────────────────────────
    X_train_nc = X_train.drop(columns=["cluster"], errors="ignore")
    X_test_nc  = X_test.drop(columns=["cluster"],  errors="ignore")
    cat_cols_nc = [c for c in ["settlement_type","STATE"] if c in X_train_nc.columns]
    full_pp_nc = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_nc),
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  MinMaxScaler()),
        ]), numeric_cols),
    ])

    model_lr_nc = Pipeline([("preprocess", full_pp_nc),
                             ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    model_lr_nc.fit(X_train_nc, y_train)
    y_pred_lr_nc = model_lr_nc.predict(X_test_nc)

    model_rf_nc = Pipeline([("preprocess", full_pp_nc),
                             ("clf", RandomForestClassifier(n_estimators=n_estimators, random_state=42))])
    model_rf_nc.fit(X_train_nc, y_train)
    y_pred_rf_nc = model_rf_nc.predict(X_test_nc)

    # ── Metric table ──────────────────────────────────────────────────────────
    def metrics(y_true, y_pred, model, X):
        return {
            "Accuracy":  round(accuracy_score(y_true, y_pred), 3),
            "Precision": round(precision_score(y_true, y_pred, zero_division=0), 3),
            "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 3),
            "ROC AUC":   round(roc_auc_score(y_true, model.predict_proba(X)[:,1]), 3),
        }

    summary = pd.DataFrame({
        "LR w/ cluster":    metrics(y_test, y_pred_lr,    model_lr,    X_test),
        "LR w/o cluster":   metrics(y_test, y_pred_lr_nc, model_lr_nc, X_test_nc),
        "RF w/ cluster":    metrics(y_test, y_pred_rf,    model_rf,    X_test),
        "RF w/o cluster":   metrics(y_test, y_pred_rf_nc, model_rf_nc, X_test_nc),
    }).T

    st.markdown("### Model Comparison")
    st.dataframe(summary.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

    # ── Confusion matrices ────────────────────────────────────────────────────
    st.markdown("### Confusion Matrices")
    c1, c2 = st.columns(2)
    for col, title, model, y_pred, X in [
        (c1, "Logistic Regression", model_lr, y_pred_lr, X_test),
        (c2, "Random Forest",       model_rf, y_pred_rf, X_test),
    ]:
        with col:
            st.markdown(f"**{title}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            ConfusionMatrixDisplay.from_estimator(model, X, y_test, ax=ax, colorbar=False)
            ax.set_title(title)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    # ── ROC curves ────────────────────────────────────────────────────────────
    st.markdown("### ROC Curves")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, title, model, X in [
        (axes[0], "Logistic Regression", model_lr, X_test),
        (axes[1], "Random Forest",       model_rf, X_test),
    ]:
        RocCurveDisplay.from_estimator(model, X, y_test, ax=ax)
        ax.set_title(f"ROC — {title}")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("### Top 15 Feature Importances — Random Forest")
    feat_names  = model_rf.named_steps["preprocess"].get_feature_names_out()
    importances = model_rf.named_steps["clf"].feature_importances_
    feat_df = pd.DataFrame({"feature": feat_names, "importance": importances.round(4)})
    feat_df = feat_df.sort_values("importance", ascending=False).head(15).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=feat_df, y="feature", x="importance", palette="viridis", ax=ax)
    ax.set_title("Top 15 Features — Random Forest")
    ax.set_xlabel("Importance"); ax.set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
> **Why recall matters most:** Missing a tract that *will* grow leads to
> under-allocation of resources. Random Forest's recall of **0.726** makes
> it our preferred model.
""")

    # Store models in session state for forecast tab
    st.session_state["model_rf"]         = model_rf
    st.session_state["cluster_pipeline"] = cluster_pipeline
    st.session_state["numeric_cols"]     = numeric_cols
    st.session_state["X_train_cols"]     = X_train.columns.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("2020–2030 Growth Forecast")

    if "model_rf" not in st.session_state:
        st.warning("Train models first (visit the Models tab).")
        st.stop()

    model_rf_f         = st.session_state["model_rf"]
    cluster_pipeline_f = st.session_state["cluster_pipeline"]
    numeric_cols_f     = st.session_state["numeric_cols"]
    X_train_cols_f     = st.session_state["X_train_cols"]

    with st.spinner("Preparing forecast data …"):
        fc = forecast_tracts.copy()
        fc = fc[fc["land_area_sqkm"] > 0]
        fc = fc[fc["pop_total_2020"] >= 100]
        fc = fc[fc["STATE"].isin(EAST_NORTH_CENTRAL)]

        fc = engineer_features(fc, "2020")
        if "pop_total_2020" in fc.columns:
            fc = fc.rename(columns={"pop_total_2020": "pop_total_2010"})

        fc["cluster"] = cluster_pipeline_f.predict(fc[numeric_cols_f])

        # Align columns
        for col in X_train_cols_f:
            if col not in fc.columns:
                fc[col] = 0
        X_fc = fc[X_train_cols_f]

        fc["predicted_grew"] = model_rf_f.predict(X_fc)
        fc["prob_grow"]      = model_rf_f.predict_proba(X_fc)[:, 1]

    st.success(f"Forecast complete — {len(fc):,} tracts processed.")

    st.markdown("**Predicted Class Balance**")
    st.bar_chart(fc["predicted_grew"].value_counts().rename({0: "No Growth", 1: "Growth"}))

    # State summary
    state_sum = fc.groupby("STATE").agg(
        total_tracts          = ("predicted_grew", "count"),
        tracts_predicted_grow = ("predicted_grew", "sum"),
        avg_prob_grow         = ("prob_grow",       "mean"),
    ).reset_index()
    state_sum["pct_tracts_growing"] = (
        state_sum["tracts_predicted_grow"] / state_sum["total_tracts"] * 100
    ).round(1)
    state_sum["avg_prob_grow"] = state_sum["avg_prob_grow"].round(3)
    state_sum = state_sum.sort_values("pct_tracts_growing", ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**State-Level Summary**")
        st.dataframe(state_sum, use_container_width=True, hide_index=True)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=state_sum, x="STATE", y="pct_tracts_growing",
                    palette="viridis", ax=ax)
        ax.set_ylabel("% Tracts Predicted to Grow")
        ax.set_title("Predicted 2020–2030 Growth by State")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Top tracts
    top_cols = [c for c in
        ["GISJOIN","STATE","COUNTY","settlement_type",
         "pop_total_2010","density_perkm2","cluster","prob_grow","predicted_grew"]
        if c in fc.columns]

    st.markdown("**Top 10 Tracts Most Likely to Grow**")
    st.dataframe(
        fc.sort_values("prob_grow", ascending=False)[top_cols]
          .head(10)
          .reset_index(drop=True)
          .style.format({"prob_grow": "{:.3f}"}),
        use_container_width=True,
    )

    st.info(
        "Many top tracts are in urban settlement types (Michigan, Ohio), "
        "cluster 1 or 3, with high density — consistent with established "
        "or rapidly developing urban cores."
    )
