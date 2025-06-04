import streamlit as st

st.title("🧠 Prédiction de Cluster à partir d'un Avis Client")

# Entrée utilisateur
text = st.text_area("✍️ Dites nous ce que vous pensez :", height=150)

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def vectorize_with_w2v(tokens):
    vectors = [model_w2v.wv[token] for token in tokens if token in model_w2v.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model_w2v.vector_size)

# def vectorize_with_tfidf(text):
#     return tfidf.transform([text]).toarray()

if st.button("Prédire le cluster"):
    if text.strip() == "":
        st.warning("Merci de saisir un avis avant de lancer la prédiction.")
    else:
        tokens = preprocess(text)
        vec = vectorize_with_w2v(tokens).reshape(1, -1)
        cluster = kmeans.predict(vec)[0]
        st.success(f"✅ Cet avis appartient au **cluster {cluster}**.")