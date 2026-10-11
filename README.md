# GNN Project — Prédiction du pays d'un aéroport par Graph Neural Networks

## Auteurs

Projet réalisé par Yennipena Jérémie COULIBALY et Michel Junior DELANOUE dans le cadre du cours **Intelligence Artificielle Bio-Inspirée** — Université Claude Bernard Lyon 1.

## Question de recherche

> **Dans quelle mesure la structure topologique du réseau aérien améliore-t-elle la prédiction du pays d'un aéroport par rapport à ses seules caractéristiques individuelles ?**

Le détail de la démarche, du protocole expérimental et des résultats se trouve dans le rapport (`templateRapportM2IA-GNN/main.pdf`). Ce README se concentre sur la reproductibilité.

## Structure du projet

```text
.
├── final_notebook.ipynb              # Notebook à exécuter — voir ci-dessous
├── gnn_tp.ipynb                      # [OBSOLÈTE] expérimentation, non maintenu
├── projet_gnn.ipynb                  # [OBSOLÈTE] expérimentation, non maintenu
├── projet_gnn_cmmented.ipynb         # [OBSOLÈTE] expérimentation, non maintenu
├── airportsAndCoordAndPop.graphml    # Graphe du réseau aérien (format GraphML)
├── requirements.txt                  # Dépendances Python, versions figées
├── templateRapportM2IA-GNN/          # Rapport LaTeX (main.tex → main.pdf)
└── README.md
```

## Notebook à considérer

**`final_notebook.ipynb` est le seul notebook à exécuter.** Les autres notebooks à la racine (`gnn_tp.ipynb`, `projet_gnn.ipynb`, `projet_gnn_cmmented.ipynb`) sont des expérimentations intermédiaires conservées pour l'historique du projet, mais ils sont obsolètes et ne reflètent pas le protocole final décrit dans le rapport.

## Reproductibilité

### 1. Environnement

```bash
python -m venv venv
source venv/bin/activate      # ou venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

Testé avec Python 3.13. Voir la note dans `requirements.txt` si l'installation de `torch` pose problème selon votre configuration (CPU/CUDA).

### 2. Données

Le fichier `airportsAndCoordAndPop.graphml` est déjà présent à la racine du projet — aucune étape supplémentaire n'est nécessaire.

### 3. Exécution

Ouvrir `final_notebook.ipynb` (Jupyter ou VS Code) et exécuter toutes les cellules dans l'ordre, de haut en bas. Le notebook est organisé en deux parties :

1. **Prétraitement** — chargement du graphe, qualité des données, filtrage des classes trop petites, split train/val/test.
2. **Partie 1 — Machine Learning / Deep Learning** — baselines sur les caractéristiques des nœuds seules (LogisticRegression, RandomForest, MLP).
3. **Partie 2 — Graphe (GNN)** — GCN, GAT, SGC, comparés en conditions structure seule / features + structure.

Toutes les métriques sont moyennées sur 5 seeds fixées (0 à 4), pour une reproductibilité exacte des résultats du rapport. L'exécution complète du notebook prend environ 15 à 20 minutes sur un CPU standard (les boucles d'entraînement à 5 seeds sont la partie la plus longue).

### 4. Rapport

Le rapport est dans `templateRapportM2IA-GNN/`.

## Usage de l'IA générative

Conformément aux consignes, nous précisons ici l'usage fait de l'IA générative dans ce projet.

- **Modèle utilisé** : Claude (Anthropic), modèle Claude Sonnet 5.
- **Rédaction** : l'IA a été utilisée pour générer un premier brouillon de ce README, ensuite relu et corrigé par nous.
- **État de l'art** : pour un premier repérage rapide des références pertinentes, nous avons notamment utilisé [Elicit](https://elicit.com), un outil de recherche bibliographique qui s'appuie lui-même sur de la gen IA en arrière-plan.
- **Aide sur le code** : nous avons utilisé l'IA générative comme assistant de programmation, par exemple pour :
  - Trouver une implémentation existante plutôt que la réimplémenter — *« On veut utiliser une fonction de coût pondérée, y a-t-il déjà une implémentation dans scikit-learn ? »*
  - Choisir un outil adapté à un besoin de visualisation — *« Nous avons un énorme graphe mais les nœuds ont des attributs lon/lat, ce serait plus lisible si on pouvait tout afficher sur une carte »* (l'IA a proposé Plotly, utilisé dans le notebook pour la carte interactive du réseau aérien).

Toutes les analyses, choix méthodologiques, interprétations des résultats et le code final restent de notre responsabilité. L'IA a servi d'assistant ponctuel, pas de source des conclusions du projet.
