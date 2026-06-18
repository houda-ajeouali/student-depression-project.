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
            <span>Module : <b>Data Science & Machine Learning</b></span> |<span>Realisé par : <b style="color: #FFCC00>Ajeouali Houda&& Bahi Ismail</b></span>|<span>Filière : <b>IA S6</b></span> |<span>Encadré par : <b style="color: #FFCC00;">Monsieur Aaroud</b></span>
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
    # =========================
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

    # Palette Jaune/Orange pour les graphes Plotly
    orange_yellow_scale = ['#FFCC00', '#FF9900', '#FF6600', '#CC3300']

    # =========================
    # TABS STRUCTURE (L'affichage Pro)
    # =========================
    tab1, tab2, tab3 = st.tabs([
        "📊 1. Data Pipeline & Ingestion", 
        "🤖 2. Advanced Model Diagnostic", 
        "🔮 3. Real-Time AI Simulator"
    ])

    # --- TAB 1: DATA VIEW & KPIS ---
    with tab1:
        st.markdown("### Ingestion Framework & Real-Time Metrics")
        k1, k2, k3, k4 = st.columns(4)
        with k1: st.markdown(f"<div class='kpi-box'><h5>Volume Stream (HDFS)</h5><h2>{len(df)} Rows</h2><span style='color:#FFCC00;'>Live Dataset</span></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi-box'><h5>Meilleur Modèle</h5><h2>{best_model_name}</h2><span style='color:#FF9900;'>Score: {best_acc*100:.1f}%</span></div>", unsafe_allow_html=True)
        with k3: st.markdown(f"<div class='kpi-box'><h5>Âge Moyen Étudiants</h5><h2>{df[col_age].mean():.1f} Ans</h2><span style='color:#FFCC00;'>Target Population</span></div>", unsafe_allow_html=True)
        with k4: st.markdown(f"<div class='kpi-box'><h5>Vecteur Dimensions</h5><h2>{X.shape[1]} Features</h2><span style='color:#FF9900;'>Engine Synchronized</span></div>", unsafe_allow_html=True)
        
        st.write("##")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.markdown("#### Datastream Raw View")
            st.dataframe(df.head(10), use_container_width=True)
        with c2:
            st.markdown("#### Distribution Réelle de la Cible (Depression)")
            fig_dep = px.pie(df, names=col_target, hole=0.4, color_discrete_sequence=['#FFCC00', '#FF6600'])
            fig_dep.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_dep, use_container_width=True)

    # --- TAB 2: MULTI-MODELS DIAGNOSTIC ---
    with tab2:
        st.markdown("### Évaluation Comparative des Algorithmes")
        
        m1, m2, m3 = st.columns(3)
        for i, (name, score) in enumerate(accuracies.items()):
            cols = [m1, m2, m3]
            cols[i].metric(label=f"Précision {name}", value=f"{score * 100:.2f} %", delta="Calculé")

        st.write("##")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown(f"#### Matrice de Confusion — {best_model_name} (Top)")
            best_preds = trained_models[best_model_name].predict(X_test)
            cm = confusion_matrix(y_test, best_preds)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=orange_yellow_scale,
                               x=["Prédit: Sain", "Prédit: Dépressif"], y=["Réel: Sain", "Réel: Dépressif"])
            fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_cm, use_container_width=True)
            
        with cc2:
            st.markdown("#### Comparaison Graphique des Performances")
            df_acc = pd.DataFrame(list(accuracies.items()), columns=['Modèle', 'Accuracy'])
            fig_bar = px.bar(df_acc, x='Modèle', y='Accuracy', color='Accuracy', color_continuous_scale=orange_yellow_scale)
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig_bar, use_container_width=True)

    # --- TAB 3: SIMULATEUR DYNAMIQUE ---
    with tab3:
        st.markdown("### Simulateur d'Inférence Algorithmique (Sécurisé)")
        st.info(f"Le système utilise actuellement le modèle optimal : **{best_model_name}**")
        
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
            with st.spinner("Calcul en cours..."):
                time.sleep(0.4)
                prediction = trained_models[best_model_name].predict([user_inputs])[0]
                
            if prediction == 1:
                st.error(f"⚠️ STATUT CRITIQUE DÉTECTÉ — Risque d'effondrement psychologique élevé.")
            else:
                st.success(f"✅ STATUT STABLE DÉTECTÉ — Équilibre optimal des indicateurs.")

except Exception as e:
    st.error(f"Erreur d'exécution système : {e}")
