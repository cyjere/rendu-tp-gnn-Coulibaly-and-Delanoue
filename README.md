# GNN Project — Prédiction du pays d'un aéroport par Graph Neural Networks

## Auteurs

Ce projet a été réalisé par Yennipena Jérémie COULIBALY et Michel Junior DELANOUE dans le cadre du cours **Intelligence Artificielle Bio-Inspirée** — Université Lyon 1.

## Question de recherche

> **Dans quelle mesure la structure topologique du réseau aérien améliore-t-elle la prédiction du pays d'un aéroport par rapport à ses seules caractéristiques individuelles ?**

Ce projet explore l'apport de la structure du graphe (connexions entre aéroports) pour la tâche de classification de nœuds, en comparant des modèles qui utilisent uniquement les attributs individuels (longitude, latitude, population, ville) à des modèles qui exploitent également la topologie du réseau aérien.

## Structure du projet

```text
.
├── gnn_tp.ipynb   # Notebook du projet
├── airportsAndCoordAndPop.graphml   # Graphe du réseau aérien (format GraphML)
└── README.md                        # Ce fichier
```

---

## Données

### Source

Le graphe est chargé depuis un fichier **GraphML** (`airportsAndCoordAndPop.graphml`) contenant :

- **3 363 nœuds** (aéroports)
- **13 547 arêtes** (liaisons aériennes)

### Attributs des nœuds

Chaque nœud (aéroport) possède les attributs suivants :

| Attribut | Type | Description |
|---|---|---|
| `lon` | float | Longitude de l'aéroport |
| `lat` | float | Latitude de l'aéroport |
| `population` | int | Population de la ville desservie |
| `country` | string | **Pays de l'aéroport** (variable cible) |
| `city_name` | string | Nom de la ville |

## Objectif de la tâche

**Classification de nœuds (Node Classification)** :

- Prédire le **pays** d'un aéroport à partir de ses caractéristiques.
- Comparer deux approches :
  1. **Baseline** : utilisation des seules caractéristiques individuelles (`lon`, `lat`, `population`).
  2. **GNN** : utilisation conjointe des caractéristiques et de la structure topologique du graphe (liens entre les aéroports).

## Approche méthodologique

### 1. Exploration et visualisation

- Chargement du graphe avec `networkx`.
- Statistiques de base : nombre de nœuds et d'arêtes.
- Visualisation initiale (`spring_layout`), peu lisible pour un graphe de cette taille.
- Amélioration de la visualisation pour un grand graphe (sous-échantillonnage, layout optimisé).

### 2. Encodage des caractéristiques

- Encodage de la variable cible `country` avec `LabelEncoder`.
- Normalisation des caractéristiques numériques (`lon`, `lat`, `population`).
- Encodage des attributs catégoriels (`city_name`, si utilisé).

### 3. Modélisation GNN

Modèles utilisés via `torch_geometric` :

- **GCNConv** (*Graph Convolutional Network*).
- **VGAE** (*Variational Graph Auto-Encoder*), pour l'apprentissage non supervisé et les embeddings.
- Autres couches GNN, selon le notebook.

### 4. Évaluation

- Séparation des données en ensembles d'entraînement, de validation et de test (*train / validation / test*).
- Métriques : accuracy et matrice de confusion.
- Comparaison de la baseline et du modèle GNN.

### 5. Analyse des embeddings

- Visualisation t-SNE des embeddings appris.
- Analyse de la séparation des classes (pays) dans l'espace latent.

## Dépendances

Installation des principales dépendances :

```bash
pip install numpy pandas matplotlib seaborn scikit-learn networkx torch torch-geometric tqdm
```

| Bibliothèque | Usage |
|---|---|
| `numpy` | Calcul numérique |
| `pandas` | Manipulation des données |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Prétraitement, métriques et t-SNE |
| `networkx` | Manipulation et visualisation du graphe |
| `torch` | Framework de deep learning |
| `torch_geometric` | Couches et modèles GNN (GCNConv, VGAE, etc.) |
| `tqdm` | Barres de progression |

## Exécution

1. Cloner ou télécharger le projet.
2. Placer le fichier `airportsAndCoordAndPop.graphml` à la racine du projet.
3. Ouvrir le notebook souhaité, par exemple :

   ```bash
   jupyter notebook gnn_tp-grapheNetworks.ipynb
   ```

4. Exécuter les cellules du notebook dans l'ordre.

> **Remarque :** adaptez le nom du notebook à celui que vous souhaitez exécuter. Les noms indiqués dans la structure du projet doivent correspondre aux fichiers réellement présents dans le dépôt.

## Résultats attendus

- **Baseline** : les performances peuvent être limitées, car les caractéristiques individuelles (notamment latitude et longitude) ne suffisent pas toujours à distinguer précisément les pays, par exemple dans certaines régions où les aéroports sont proches géographiquement.
- **GNN** : le modèle peut améliorer la prédiction grâce à la propagation d'information via les connexions aériennes.
- **Conclusion attendue** : la topologie du réseau aérien peut apporter une information complémentaire utile à la classification des aéroports par pays.

> Les améliorations de performance mentionnées ci-dessus sont des hypothèses à vérifier expérimentalement ; elles ne constituent pas des résultats mesurés tant que les notebooks n'ont pas été exécutés et évalués.

## Pistes d'amélioration

- Ajouter des caractéristiques de centralité : degré, *betweenness centrality*, PageRank.
- Tester d'autres architectures GNN : GraphSAGE, GAT et GIN.
- Utiliser des embeddings de graphe pré-entraînés : Node2Vec et DeepWalk.
- Gérer le déséquilibre des classes, notamment la forte représentation de certains pays.
- Mettre en place une validation croisée et une recherche d'hyperparamètres.
