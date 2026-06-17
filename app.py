import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import time

# =========================
# CONFIG & DESIGN (UCD PLATFORM)
# =========================
st.set_page_config(page_title="MindCare-UCD - Student Depression Platform", layout="wide")

# CSS Design integration (Style Old Gold, Noir & Jaune Premium)
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a150b 60%, #33260c 100%);
        padding: 35px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.7);
        margin-bottom: 30px;
        border-left: 10px solid #C5A059;
        border-right: 10px solid #C5A059;
    }
    .kpi-box {
        background-color: #141414;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        border-top: 4px solid #C5A059;
        text-align: center;
        color: #ffffff;
    }
    .kpi-box h5 { color: #aaaaaa !important; margin-bottom: 5px; }
    .kpi-box h2 { color: #FFCC00 !important; font-weight: 800; margin: 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1c1c1c;
        color: #bbbbbb;
        border-radius: 10px 10px 0px 0px;
        padding: 12px 25px;
        font-weight: 800;
        font-size: 1rem;
        border: 1px solid #2d2d2d;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #C5A059 0%, #947437 100%) !important;
        color: #000000 !important;
    }
    /* Bouton personnalisé Old Gold / Jaune */
    .stButton>button {
        background: linear-gradient(135deg, #FFCC00 0%, #C5A059 100%) !important;
        color: black !important;
        font-weight: bold !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Banner Université Chouaib Doukkali - MindCare-UCD
st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: white; margin:0; font-family: 'Segoe UI', sans-serif; font-weight: 900; text-transform: uppercase;">Université Chouaib Doukkali — Faculté des Sciences</h1>
        <h2 style="text-align: center; color: #C5A059; margin:10px 0 0 0; font-family: 'Segoe UI', sans-serif; font-weight: 700;">MindCare-UCD — PLATEFORME EXPERTE DE PREDICTION & COMPUTING BIG DATA</h2>
        <div style="text-align: center; margin-top: 15px; color: #e1e1e1; font-size: 1.1rem;">
            <span>Module : <b>Data Science & Machine Learning</b></span> | <span>Encadré par : <b style="color: #FFCC00;">Monsieur Aaroud</b></span>
        </div>
    </div>
""", unsafe_allow_html=True)

try:
    # =========================
    # LOAD DATA (Sécurité Chemin Relatif Docker)
    # =========================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "Student Depression Dataset.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df = df.dropna().drop_duplicates()
        if 'id' in df.columns: df = df.drop(columns=['id'])
    else:
        st.error(f"❌ Fichier introuvable dans le conteneur Docker à l'emplacement : {file_path}")
        st.stop()

    # --- SÉCURISATION ET RECHERCHE SMART DES COLONNES ---
    def find_column(possible_names, default_name):
        for col in df.columns:
            if col.lower().strip() in [p.lower().strip() for p in possible_names] or any(p.lower() in col.lower() for p in possible_names):
                return col
        return default_name

    col_target = find_column(['Depression', 'Target', 'Output'], df.columns[-1])
    col_age = find_column(['Age'], 'Age')
    col_pressure = find_column(['Academic Pressure', 'Pressure'], 'Academic Pressure')
    col_cgpa = find_column(['CGPA', 'Note'], 'CGPA')

    # --- SÉCURISATION ULTIME DE L'ERREUR DES LABELS ENVISAGEABLES (ex: 2.0) ---
    df[col_target] = pd.to_numeric(df[col_target], errors='coerce')
    df = df[df[col_target].isin([0, 1, 0.0, 1.0])].dropna()
    df[col_target] = df[col_target].astype(int)

    # =========================
    # PROCESSING & ENCODING (Sécurisé à 100% contre l'erreur String)
    # =========================
    df_encoded = df.copy()
    label_encoders = {}

    # Katched kolchi li fih text kima bgha ykoune format dyalou (object, string, category)
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category', 'string']).columns

    for col in categorical_cols:
        if col != col_target:
            le = LabelEncoder()
            df_encoded[col] = df_encoded[col].astype(str).str.strip()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

    # Features / Target split
    X = df_encoded.drop(col_target, axis=1)
    y = df_encoded[col_target]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # =========================
    # MULTI-MODELS TRAINING
