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



        ┌────────────────────┐
        │ Données brutes     │
        │ (avis cient .json) │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Étape 1 :          │
        │ Prétraitement NLP  │
        │ (spaCy)            │
        │ - Tokenisation     │
        │ - Lemmatisation    │
        │ - Nettoyage        │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Étape 2 :          │
        │ Vectorisation      │
        │ (Word2Vec)         │
        │ - Embedding moyen  │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Étape 3 :          │
        │ Clustering         │
        │ (KMeans)           │
        │ - Regroupement     │
        │   par thème        │
        └────────┬───────────┘
                 │
                 ▼
        ┌────────────────────┐
        │ Résultats          │
        │ - Répartition      │
        │   thématique       │
        │ - Analyse marketing│
        └────────────────────┘





## 📊 Thématiques identifiées par Clustering

Grâce à l'algorithme de **KMeans**, nous avons regroupé les réponses en plusieurs clusters pour identifier les thèmes principaux exprimés par les citoyens concernant la pratique sportive en France.

### 🏷️ Cluster 0 - **Création de nouvelles infrastructures sportives**

* **Mots-clés** : `creer`, `salle`, `scolaire`, `personne`, `bienfait`
* **Description** :
  Ce cluster regroupe les réponses exprimant le **besoin de créer de nouvelles infrastructures sportives** pour faciliter l'accès à la pratique du sport. Les citoyens mettent en avant l'importance de construire des **salles de sport**, notamment dans un cadre **scolaire**, pour promouvoir les bienfaits de l'activité physique.

---

### 🏷️ Cluster 1 - **Soutien public et accompagnement des personnes**

* **Mots-clés** : `pouvoir`, `public`, `personne`, `aider`, `creer`
* **Description** :
  Ce cluster représente les réponses où les citoyens expriment un **besoin d’accompagnement de la part des autorités publiques**. Les réponses mettent en avant l'importance de **l'aide publique** pour faciliter l'accès aux activités sportives, notamment pour les personnes en difficulté.

---

### 🏷️ Cluster 2 - **Sport et éducation**

* **Mots-clés** : `matiere`, `competition`, `organisation`, `etude`, `maternelle`
* **Description** :
  Les réponses de ce cluster portent sur la **pratique sportive dans le cadre éducatif**. Les citoyens insistent sur l'importance d'intégrer le sport dès l'école maternelle, d'organiser des **compétitions sportives** et de faire de l'éducation physique une **matière essentielle**.

---

### 🏷️ Cluster 3 - **Manque de temps pour pratiquer le sport**

* **Mots-clés** : `pouvoir`, `pratiquer`, `creer`, `heure`, `permettre`
* **Description** :
  Les réponses de ce cluster reflètent les **contraintes de temps** rencontrées par les citoyens pour pratiquer le sport. Ils expriment un besoin de **flexibilité horaire** et la nécessité de **créer des espaces adaptés** pour intégrer l'activité physique dans un emploi du temps chargé.

---

### 🏷️ Cluster 4 - **Accessibilité aux installations sportives**

* **Mots-clés** : `permettre`, `acce`, `age`, `accessible`, `place`
* **Description** :
  Ce cluster regroupe les réponses mettant en avant les **problèmes d’accessibilité aux infrastructures sportives**. Les citoyens évoquent le manque de **places disponibles** et la difficulté d'accès pour certains **groupes d’âge**. Ils souhaitent des équipements plus **proches et accessibles**.

