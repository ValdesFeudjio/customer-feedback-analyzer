## 🧠 Pipeline de Traitement NLP

Ce projet vise à identifier automatiquement les **thèmes présents dans un corpus de textes** (avis clients, commentaires, etc.) grâce à un pipeline de traitement en **trois étapes clés**, cela a pour objectif d'ajuster les stratégies marketing selon l'appartenance thématique des clients  :

---

### ⚙️ 1. Prétraitement du texte avec spaCy

Le texte brut est nettoyé et préparé grâce à une **fonction unique de prétraitement** basée sur `spaCy`. Les opérations effectuées sont :

- **Tokenisation** : séparation du texte en unités de sens (mots, ponctuations, etc.)
- **Lemmatisation** : réduction des mots à leur forme de base (ex. *"mangeait"* → *"manger"*)
- **Suppression des stop words**, ponctuation, espaces superflus, caractères spéciaux
- **Mise en minuscules** pour homogénéiser le corpus

🔧 *Cette étape permet d’obtenir une version standardisée et exploitable des textes.*

---

### 📐 2. Vectorisation avec Word2Vec

Chaque document est transformé en vecteur à l’aide du modèle **Word2Vec** (via la librairie `gensim`) :

- Apprentissage des vecteurs de mots sur le corpus
- Chaque document est ensuite représenté par la **moyenne des vecteurs de ses mots** (embedding moyen)
- On obtient une **matrice X** (n_documents × n_dimensions) qui représente l’ensemble du corpus de manière numérique

🧩 *Cela permet de capturer la sémantique des documents.*

---

### 📊 3. Clustering des documents avec KMeans

L’étape finale consiste à regrouper les documents en **k clusters** à l’aide de l’algorithme **KMeans** (`sklearn`) :

- Chaque cluster représente un **thème latent** détecté dans le corpus
- Chaque document est automatiquement **classé dans l’un des k thèmes**
- Il est ensuite possible d’analyser les clusters, visualiser les répartitions ou explorer les textes par thème

🔍 *Cette approche permet de segmenter le corpus selon les grandes idées ou sentiments qu’il contient.*
