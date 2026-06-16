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

# ==========================================
# CONFIGURATION & STYLE VISUEL (THÈME BLEU)
# ==========================================
st.set_page_config(page_title="Tech Workshop - AI Project", layout="wide")

# Intégration du CSS pour un thème 100% Bleu Tech
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0A192F 0%, #172A45 50%, #30475E 100%);
        padding: 35px;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        margin-bottom: 30px;
        border-left: 10px solid #00BCD4;
        border-right: 10px solid #00BCD4;
    }
    .kpi-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        border-top: 4px solid #1F6FEB;
        text-align: center;
        color: #333333;
    }
    .kpi-box h5 { color: #555555 !important; margin-bottom: 5px; }
    .kpi-box h2 { color: #1F6FEB !important; font-weight: 800; margin: 0; }
    .stTabs [data-baseweb="tab-list"] { gap: 15px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f4f8;
        border-radius: 10px 10px 0px 0px;
        padding: 12px 25px;
        font-weight: 800;
        font-size: 1rem;
        color: #172A45;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1F6FEB 0%, #00BCD4 100%) !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# Bannière personnalisée avec vos informations
st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: white; margin:0; font-family: 'Segoe UI', sans-serif; font-weight: 900; text-transform: uppercase; letter-spacing: 2px;">Tech Workshop</h1>
        <h2 style="text-align: center; color: #00BCD4; margin:10px 0 0 0; font-family: 'Segoe UI', sans-serif; font-weight: 700;">PROJET DE PRÉDICTION DE LA SANTÉ MENTALE (SMHP)</h2>
        <div style="text-align: center; margin-top: 20px; color: #e1e1e1; font-size: 1.1rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px;">
            <span>Filière : <b>IA S6</b></span> | 
            <span>Fait par : <b style="color: #00BCD4;">Houda Ajeouali & Ismail Bahi</b></span> | 
            <span>Encadré par : <b>Mr. Aaroud</b></span>
        </div>
    </div>
""", unsafe_allow_html=True)

try:
    # ==========================================
    # CHARGEMENT DES DONNÉES (Cloud & Local)
    # ==========================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "Student Depression Dataset.csv")

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        df = df.dropna().drop_duplicates()
        if 'id' in df.columns: df = df.drop(columns=['id'])
    else:
        st.error(f"❌ Fichier introuvable à l'emplacement : {file_path}")
        st.stop()

    # --- RECHERCHE INTELLIGENTE DES COLONNES ---
    def find_column(possible_names, default_name):
        for col in df.columns:
            if col.lower().strip() in [p.lower().strip() for p in possible_names] or any(p.lower() in col.lower() for p in possible_names):
                return col
        return default_name

    col_target = find_column(['Depression', 'Target', 'Output'], df.columns[-1])
    col_age = find_column(['Age'], 'Age')
    col_pressure = find_column(['Academic Pressure', 'Pressure'], 'Academic Pressure')
    col_cgpa = find_column(['CGPA', 'Note'], 'CGPA')

    # Nettoyage et sécurisation de la cible y (suppression des labels parasites comme 2.0)
    df[col_target] = pd.to_numeric(df[col_target], errors='coerce')
    df = df[df[col_target].isin([0, 1, 0.0, 1.0])].dropna()
    df[col_target] = df[col_target].astype(int)

    # MAP DE TEXTE POUR LES GRAPHIQUES (Pour ne pas afficher 0 et 1 bruts)
    df_plots = df.copy()
    df_plots['Statut Dépression'] = df_plots[col_target].map({0: 'Sain (Non Dépressif)', 1: 'Dépressif'})

    # ==========================================
    # ENCODAGE SÉCURISÉ DES SOURCELS TEXTUELLES
    # ==========================================
    df_encoded = df.copy()
    label_encoders = {}

    for col in df_encoded.columns:
        if col != col_target:
            if df_encoded[col].dtype == "object" or df_encoded[col].dtype == "string" or not np.issubdtype(df_encoded[col].dtype, np.number):
                le = LabelEncoder()
                df_encoded[col] = df_encoded[col].astype(str).str.strip()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoders[col] = le

    # Séparation Features / Target
    X = df_encoded.drop(col_target, axis=1)
    y = df_encoded[col_target]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================
    # ENTRAÎNEMENT DES 3 MODÈLES (SQUELETTE INITIAL)
    # ==========================================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    accuracies = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, preds)
        trained_models[name] = model

    best_model_name = max(accuracies, key=accuracies.get)
    best_acc = accuracies[best_model_name]

    # ==========================================
    # STRUCTURE DES ONGLETS (TABS)
    # ==========================================
    tab1, tab2, tab3 = st.tabs([
        "📊 1. Data Pipeline Ingestion", 
        "🤖 2. Comparative Diagnostic", 
        "🔮 3. Real-Time AI Simulator"
    ])

    # --- ONGLET 1 : EXPLORATION ET KPIs ---
    with tab1:
        st.markdown("### Ingestion Framework & Real-Time Metrics")
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f"<div class='kpi-box'><h5>Volume Stream (HDFS)</h5><h2>{len(df)} Lignes</h2><span style='color:green;'>Dataset Chargé</span></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-box'><h5>Meilleur Algorithme</h5><h2>{best_model_name}</h2><span style='color:#00BCD4;'>Précision: {best_acc*100:.1f}%</span></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-box'><h5>Âge Moyen</h5><h2>{df[col_age].mean():.1f} Ans</h2><span style='color:#1F6FEB;'>Moyenne Étudiants</span></div>", unsafe_allow_html=True)
        with k4: st.markdown(f"<div class='kpi-box'><h5>Dimensions du Vecteur</h5><h2>{X.shape[1]} Descripteurs</h2><span style='color:green;'>Pipeline Prêt</span></div>", unsafe_allow_html=True)
        
        st.write("##")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("#### Aperçu des Données (Flux Hadoop/Spark View)")
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.markdown("#### Distribution de la Dépression (Sans Mauve)")
            # Utilisation de couleurs Bleues, Cyans et Oranges
            fig_dep = px.pie(df_plots, names='Statut Dépression', hole=0.4, 
                             color_discrete_sequence=['#1F6FEB', '#FF9800'])
            st.plotly_chart(fig_dep, use_container_width=True)

    # --- ONGLET 2 : DIAGNOSTIC DES MODÈLES ---
    with tab2:
        st.markdown("### Évaluation Comparative des Algorithmes (Module ML)")
        m1, m2, m3 = st.columns(3)
        for i, (name, score) in enumerate(accuracies.items()):
            cols = [m1, m2, m3]
            cols[i].metric(label=f"Précision {name}", value=f"{score * 100:.2f} %", delta="Calculé")

        st.write("##")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"#### Matrice de Confusion — {best_model_name}")
            best_preds = trained_models[best_model_name].predict(X_test)
            cm = confusion_matrix(y_test, best_preds)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               x=["Prédit: Sain", "Prédit: Dépressif"], y=["Réel: Sain", "Réel: Dépressif"])
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with cc2:
            st.markdown("#### Performance Graphique Comparée")
            df_acc = pd.DataFrame(list(accuracies.items()), columns=['Modèle', 'Accuracy'])
            fig_bar = px.bar(df_acc, x='Modèle', y='Accuracy', color='Accuracy', color_continuous_scale='Cividis')
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- ONGLET 3 : SIMULATEUR ---
    with tab3:
        st.markdown("### Simulateur d'Inférence en Temps Réel")
        st.info(f"Le système de calcul utilise actuellement le modèle optimal : **{best_model_name}**")
        
        user_inputs = []
        cols_ui = st.columns(3)
        
        for idx, col in enumerate(X.columns):
            current_col = cols_ui[idx % 3]
            with current_col:
                if col in label_encoders:
                    original_labels = label_encoders[col].classes_
                    selected_text = st.selectbox(f"{col}", original_labels, key=f"sim_{col}")
                    encoded_val = label_encoders[col].transform([selected_text])[0]
                    user_inputs.append(encoded_val)
                else:
                    if df[col].dtype == 'float64' or df[col].dtype == 'float32':
                        val = st.slider(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()), key=f"sim_{col}")
                    else:
                        val = st.slider(f"{col}", int(df[col].min()), int(df[col].max()), int(df[col].mean()), key=f"sim_{col}")
                    user_inputs.append(val)
        
        st.write("##")
        if st.button("🚀 Lancer l'Inférence Algorithmique", use_container_width=True):
            with st.spinner("Modèle en cours de calcul..."):
                time.sleep(0.3)
                prediction = trained_models[best_model_name].predict([user_inputs])[0]
                
            if prediction == 1:
                st.error(f"⚠️ STATUT CRITIQUE DÉTECTÉ — Risque d'effondrement psychologique élevé.")
            else:
                st.success(f"✅ STATUT STABLE DÉTECTÉ — Équilibre optimal des indicateurs de santé mentale.")

except Exception as e:
    st.error(f"Erreur d'exécution système : {e}")
