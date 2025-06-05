import os
import streamlit as st
import pandas as pd
import numpy as np
import spacy
import joblib
from unidecode import unidecode
from gensim.models import Word2Vec

# === Définir les chemins ===
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
MODEL_W2V_PATH = os.path.join(SRC_DIR, "word2vec.model")
KMEANS_PATH = os.path.join(SRC_DIR, "kmeans_model.joblib")


# === Chargement des modèles et ressources ===
nlp = spacy.load("fr_core_news_md")
model_w2v = Word2Vec.load(MODEL_W2V_PATH)
model_kmeans = joblib.load(KMEANS_PATH)

# === Fonctions ===

def preprocess(text):
    text = unidecode(text.lower())
    doc = nlp(text)
    return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

def full_preprocessing(text):
    tokens = preprocess(text)
    return [t for t in tokens if  len(t) > 3]

def get_comment_vector(tokens):
    vectors = [model_w2v.wv[word] for word in tokens if word in model_w2v.wv]
    if not vectors:
        return None
    vecteur_moyen = np.mean(vectors, axis=0).astype(np.float64)
    return vecteur_moyen.reshape(1, -1)

def predict_theme(text):
    cleaned_tokens = full_preprocessing(text)
    vector = get_comment_vector(cleaned_tokens)
    if vector is None:
        return None
    return model_kmeans.predict(vector)[0]

## Définition de la thématique de chaque cluster
def get_cluster_theme(cluster_id):
    themes = {
        0: "Thème 1 : Bienfaits du sport et dimension culturelle ou sociale",
        1: "Thème 2 : Intégration du sport dans le quotidien, notamment à l’école primaire",
        2: "Thème 3 : Accessibilité des infrastructures (ascenseurs, escalators, affichage)",
        3: "Thème 4 : Sécurité et accompagnement social dans la pratique sportive",
        4: "Thème 5 : Espaces ludiques et encouragement à la participation"
    }
    return themes.get(cluster_id, "Thème inconnu")


# === Interface utilisateur ===

st.set_page_config(page_title="Prédiction de Thèmes", layout="centered")
st.title("🔍 Prédiction de Thème d'Avis Client")
st.write("Entrez un commentaire client pour prédire automatiquement son thème.")

# Champ de texte
default_text = "Il faut créer plus de terrains de sport pour les jeunes. Le sport est essentiel pour la santé."
text_input = st.text_area("💬 Entrez un commentaire :", value=default_text, height=150)

# Bouton d'action
if st.button("🎯 Prédire le thème"):
    if not text_input.strip():
        st.warning("⛔ Veuillez entrer un commentaire.")
    else:
        cluster = predict_theme(text_input)
        if cluster is None:
            st.error("❌ Impossible de classer ce commentaire (trop court ou mots inconnus du modèle).")
        else:
            theme = get_cluster_theme(cluster)
            st.success(f"📌 Cluster : {cluster} — 🧠 {theme}")
