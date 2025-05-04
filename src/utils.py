

import spacy
from unidecode import unidecode
from collections import Counter

from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

from numpy import stack

from gensim.models import Word2Vec
import numpy as np

from sklearn.cluster import KMeans
import joblib  # Pour sauvegarder le modèle
from numpy import stack

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.manifold import TSNE


# La fonction de preprocessing complète est définie ici


# Chargement du modèle spaCy
nlp = spacy.load('fr_core_news_md')

def full_preprocessing(text_series, top_n=20):
    """
    Applique un pipeline complet de nettoyage NLP :
    - Mise en minuscules et suppression des accents
    - Tokenisation et lemmatisation avec spaCy
    - Suppression des stop words et tokens non alphabétiques
    - Suppression des top N mots les plus fréquents dans le corpus

    :param text_series: Série pandas contenant les textes bruts
    :param top_n: Nombre de mots fréquents à supprimer
    :return: Liste de listes de tokens nettoyés
    """
    # Étape 1 : prétraitement initial
    def preprocess(text):
        text = unidecode(text.lower())
        doc = nlp(text)
        return [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    
    # Appliquer le prétraitement à tous les textes
    token_lists = text_series.astype(str).apply(preprocess)

    # Étape 2 : suppression des mots les plus fréquents
    all_tokens = [token for tokens in token_lists for token in tokens]
    most_common = set([w for w, _ in Counter(all_tokens).most_common(top_n)])
    
    # Nettoyage final
    cleaned_tokens = token_lists.apply(lambda tokens: [t for t in tokens if t not in most_common])
    
    return cleaned_tokens

        ## Exemple d'utilisation
        # df['tokens'] = full_preprocessing(df['content'])
        
'''
    Tu obtiens une colonne tokens prête pour Word2Vec ou un autre modèle.
'''








def vectorize_with_word2vec(token_lists, vector_size=50, min_count=1, sg=0, save_path=None):
    """
    Entraîne un modèle Word2Vec sur les listes de tokens et transforme chaque document
    en une liste de vecteurs (un par token).
    
    :param token_lists: Liste de listes de tokens (ex: df['tokens'])
    :param vector_size: Dimension des vecteurs Word2Vec
    :param min_count: Seuil minimal pour qu'un mot soit pris en compte
    :param sg: 1 = Skip-gram ; 0 = CBOW
    :param save_path: Chemin pour sauvegarder le modèle (facultatif)
    :return: modèle Word2Vec entraîné + liste des documents vectorisés
    """
    # Taille maximale du contexte
    window_size = max(len(tokens) for tokens in token_lists)
    
    # Entraînement du modèle Word2Vec
    model = Word2Vec(
        sentences=token_lists,
        vector_size=vector_size,
        window=window_size,
        min_count=min_count,
        sg=sg
    )
    
    # Sauvegarde du modèle si demandé
    if save_path:
        model.save(save_path)
    
    # Fonction pour transformer une liste de tokens en liste de vecteurs
    def get_token_vectors(tokens):
        return [model.wv[token] for token in tokens if token in model.wv]

    # Application à tous les documents
    doc_vectors = token_lists.apply(get_token_vectors)
    
    return model, doc_vectors


# Exemple d'utilisation
# model, df['vectors'] = vectorize_with_word2vec(df['tokens'], vector_size=50, save_path="word2vec.model")




# Embedding moyen pour chaque document

import numpy as np

def compute_mean_embedding(doc_vectors, vector_size):
    """
    Calcule l'embedding moyen d'un document.
    
    :param doc_vectors: liste de vecteurs numpy pour un document
    :param vector_size: taille des vecteurs Word2Vec (doit être cohérente avec le modèle)
    :return: vecteur moyen (numpy array)
    """
    if len(doc_vectors) == 0:
        return np.zeros(vector_size)
    return np.mean(doc_vectors, axis=0)

# Exemple d'utilisation
# df['doc_vector'] = df['vectors'].apply(lambda v: compute_mean_embedding(v, vector_size=50))








# représentation des documents avec TSNE



def visualize_embeddings_tsne(model, top_n=300, perplexity=40, n_iter=1000, random_state=42):
    """
    Visualise les embeddings Word2Vec avec t-SNE en 2D.
    
    :param model: modèle Word2Vec entraîné (gensim)
    :param top_n: nombre de mots les plus fréquents à visualiser
    :param perplexity: paramètre t-SNE (autour de 30–50 pour ce type de données)
    :param n_iter: nombre d’itérations t-SNE
    :param random_state: pour reproductibilité
    :return: None (affiche un graphique interactif Plotly)
    """
    # Sélection des mots les plus fréquents
    mots = list(model.wv.index_to_key[:top_n])
    vecteurs = model.wv[mots]

    # Réduction de dimension avec t-SNE
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, n_iter=n_iter)
    vecteurs_2D = tsne.fit_transform(vecteurs)

    # Création du DataFrame pour Plotly
    df_tsne = pd.DataFrame({
        "mot": mots,
        "x": vecteurs_2D[:, 0],
        "y": vecteurs_2D[:, 1]
    })

    # Affichage avec Plotly
    fig = px.scatter(
        df_tsne,
        x="x",
        y="y",
        hover_name="mot",
        title=f"Visualisation des {top_n} mots avec t-SNE"
    )
    fig.update_traces(marker=dict(size=6, color="blue"))
    fig.update_layout(width=1000, height=700)
    fig.show()

# Exemple d'utilisation
# visualize_embeddings_tsne(model, top_n=300) les paramètres peuvent être ajustés selon les besoins



'''

 Clustering avec KMeans
 Le but ici est de regrouper les documents en clusters basés sur leurs embeddings moyens.
 
'''

def apply_kmeans_clustering(X, k=5, save_path=None, random_state=42):
    """
    Applique un clustering KMeans sur des vecteurs de documents.
    
    :param X: Matrice (array ou DataFrame) de vecteurs par document (shape: n_docs × n_features)
    :param k: Nombre de clusters (thèmes) à trouver
    :param save_path: Chemin pour sauvegarder le modèle KMeans (facultatif)
    :param random_state: Pour la reproductibilité
    :return: modèle KMeans entraîné, labels (cluster de chaque document)
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    labels = kmeans.fit_predict(X)
    
    # Sauvegarde optionnelle du modèle
    if save_path:
        joblib.dump(kmeans, save_path)
    
    return kmeans, labels



# Exemple d'utilisation
# Supposons que tu as une colonne `doc_vector` (vecteurs moyens de chaque document)
# X = stack(df['doc_vector'].values)
# kmeans_model, df['cluster'] = apply_kmeans_clustering(X, k=5, save_path="kmeans_model.joblib")

## Pour charger le modèle KMeans plus tard, tu peux utiliser joblib
# import joblib
# kmeans_model = joblib.load("kmeans_model.joblib")





'''
visualisation des clusters avec t-SNE
'''




def visualize_clusters_2d(vectors, labels, method='tsne', title='Visualisation des clusters'):
    """
    Réduction en 2D + affichage des clusters avec couleurs différentes.
    
    :param vectors: matrice numpy ou liste de vecteurs (n_documents × n_features)
    :param labels: liste ou array des labels de cluster (longueur = n_documents)
    :param method: 'tsne' ou 'pca'
    :param title: titre du graphique
    """
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Méthode non reconnue. Choisir 'tsne' ou 'pca'.")
    
    reduced = reducer.fit_transform(vectors)
    
    df_visu = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "cluster": labels.astype(str)
    })
    
    fig = px.scatter(
        df_visu,
        x="x",
        y="y",
        color="cluster",
        title=title,
        labels={"cluster": "Cluster"},
        width=900,
        height=600
    )
    fig.update_traces(marker=dict(size=6))
    fig.show()


# Exemple d'utilisation

# X = stack(df['doc_vector'].values)  # Matrice des vecteurs
# visualize_clusters_2d(X, df['cluster'], method='tsne')
