import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


# ============================================================
# 1. CONFIGURATION DE L'APPLICATION
# ============================================================

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Prediction")

st.markdown("""
### Prédiction du risque de défaut de paiement

Cette application utilise un modèle **XGBoost** entraîné sur des données
de crédit et **SHAP** pour expliquer les prédictions.
""")


# ============================================================
# 2. CHARGEMENT DU MODÈLE
# ============================================================

@st.cache_resource
def load_model():

    # Préprocesseur utilisé pendant l'entraînement
    preprocessor = joblib.load("preprocessor.pkl")

    # Modèle XGBoost
    model = XGBClassifier()
    model.load_model("xgb_model.json")

    # Seuil optimal trouvé pendant l'entraînement
    threshold = joblib.load("threshold.pkl")

    return preprocessor, model, threshold


try:
    preprocessor, model, threshold = load_model()

except Exception as e:
    st.error("❌ Impossible de charger le modèle.")
    st.code(str(e))
    st.stop()


# ============================================================
# 3. FORMULAIRE CLIENT
# ============================================================

st.header("👤 Informations du client")

with st.form("credit_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Informations générales
    # --------------------------------------------------------

    with col1:

        limit_bal = st.number_input(
            "LIMIT_BAL - Montant du crédit",
            min_value=0.0,
            value=50000.0,
            step=5000.0
        )

        sex = st.selectbox(
            "SEX",
            options=[1, 2],
            format_func=lambda x:
                "Homme" if x == 1 else "Femme"
        )

        education = st.selectbox(
            "EDUCATION",
            options=[0, 1, 2, 3, 4, 5, 6]
        )

        marriage = st.selectbox(
            "MARRIAGE",
            options=[0, 1, 2, 3]
        )

        age = st.number_input(
            "AGE",
            min_value=18,
            max_value=100,
            value=30,
            step=1
        )

    # --------------------------------------------------------
    # Statuts de remboursement
    # --------------------------------------------------------

    with col2:

        pay_0 = st.number_input(
            "PAY_0",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_2 = st.number_input(
            "PAY_2",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_3 = st.number_input(
            "PAY_3",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_4 = st.number_input(
            "PAY_4",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_5 = st.number_input(
            "PAY_5",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

        pay_6 = st.number_input(
            "PAY_6",
            min_value=-2,
            max_value=8,
            value=0,
            step=1
        )

    # --------------------------------------------------------
    # Factures
    # --------------------------------------------------------

    with col3:

        bill_amt1 = st.number_input(
            "BILL_AMT1",
            value=5000.0,
            step=500.0
        )

        bill_amt2 = st.number_input(
            "BILL_AMT2",
            value=5000.0,
            step=500.0
        )

        bill_amt3 = st.number_input(
            "BILL_AMT3",
            value=5000.0,
            step=500.0
        )

        bill_amt4 = st.number_input(
            "BILL_AMT4",
            value=5000.0,
            step=500.0
        )

        bill_amt5 = st.number_input(
            "BILL_AMT5",
            value=5000.0,
            step=500.0
        )

        bill_amt6 = st.number_input(
            "BILL_AMT6",
            value=5000.0,
            step=500.0
        )

    st.subheader("💰 Montants des paiements")

    col4, col5 = st.columns(2)

    with col4:

        pay_amt1 = st.number_input(
            "PAY_AMT1",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

        pay_amt2 = st.number_input(
            "PAY_AMT2",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

        pay_amt3 = st.number_input(
            "PAY_AMT3",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

    with col5:

        pay_amt4 = st.number_input(
            "PAY_AMT4",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

        pay_amt5 = st.number_input(
            "PAY_AMT5",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

        pay_amt6 = st.number_input(
            "PAY_AMT6",
            min_value=0.0,
            value=2000.0,
            step=500.0
        )

    submitted = st.form_submit_button(
        "🔍 Évaluer le risque",
        use_container_width=True
    )


# ============================================================
# 4. PRÉDICTION
# ============================================================

if submitted:

    # Création du DataFrame avec exactement les 23 variables
    # utilisées pendant l'entraînement

    input_data = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6
    }

    input_df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Préprocessing
    # --------------------------------------------------------

    input_transformed = preprocessor.transform(input_df)

    # --------------------------------------------------------
    # Probabilité de défaut
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_transformed
    )[0, 1]

    # --------------------------------------------------------
    # Décision avec le seuil optimal
    # --------------------------------------------------------

    prediction = int(probability >= threshold)

    # ========================================================
    # 5. AFFICHAGE DU RÉSULTAT
    # ========================================================

    st.divider()

    st.header("📊 Résultat de la prédiction")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Probabilité de défaut",
            f"{probability * 100:.2f}%"
        )

    with result_col2:

        st.metric(
            "Seuil utilisé",
            f"{threshold * 100:.0f}%"
        )

    if prediction == 1:

        st.error(
            "⚠️ Risque élevé : le modèle prédit un défaut de paiement."
        )

    else:

        st.success(
            "✅ Risque plus faible : le modèle ne prédit pas de défaut de paiement."
        )


    # ========================================================
    # 6. EXPLICATION SHAP
    # ========================================================

    st.divider()

    st.header("🔎 Explication de la prédiction avec SHAP")

    try:

        # Création de l'explainer SHAP
        explainer = shap.TreeExplainer(model)

        # Calcul des valeurs SHAP
        shap_values = explainer.shap_values(input_transformed)

        # Récupération des noms des variables après preprocessing
        feature_names = preprocessor.get_feature_names_out()

        # Graphique SHAP
        fig, ax = plt.subplots(figsize=(10, 7))

        shap.summary_plot(
            shap_values,
            input_transformed,
            feature_names=feature_names,
            plot_type="bar",
            max_display=10,
            show=False
        )

        st.pyplot(fig)

        st.info("""
        **Interprétation :**

        Les variables avec les barres les plus longues sont celles qui
        ont le plus influencé la prédiction du modèle pour ce client.

        SHAP explique l'influence des variables sur la prédiction du modèle.
        Cela ne signifie pas qu'une variable est directement responsable
        du défaut.
        """)

    except Exception as e:

        st.warning(
            "La prédiction fonctionne, mais l'explication SHAP "
            "n'a pas pu être générée."
        )

        st.code(str(e))


# ============================================================
# 7. INFORMATIONS SUR LE MODÈLE
# ============================================================

st.sidebar.title("ℹ️ Modèle")

st.sidebar.write("**Algorithme :** XGBoost")
st.sidebar.write("**Explainability :** SHAP")
st.sidebar.write(
    f"**Seuil de décision :** {threshold:.2f}"
)

st.sidebar.info(
    "Le modèle a été entraîné sur le dataset "
    "UCI Default of Credit Card Clients."
)