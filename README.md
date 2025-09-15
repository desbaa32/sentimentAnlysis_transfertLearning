# Projet d'Analyse de Sentiments par Transfert Learning

## Description
Ce projet académique en NLP vise à classifier des tweets selon leur sentiment en utilisant une approche de transfert learning. Un modèle pré-entraîné sur 3 classes (positif, négatif, neutre) est adapté pour une classification plus fine en 7 catégories émotionnelles.

## Objectifs
- Classification de tweets par sentiment
- Adaptation d'un modèle existant par transfert learning
- Passage de 3 à 7 classes de sentiments
- Optimisation des performances du modèle

## Fonctionnalités
- Prétraitement avancé du texte (nettoyage, lemmatisation, gestion des émoticônes)
- Architecture avec embedding GloVe et couches Bidirectionnelles LSTM
- Mécanisme de transfert learning pour l'adaptation du modèle
- Visualisation des résultats (matrices de confusion, métriques)

## Installation
```bash
pip install tensorflow nltk pandas numpy matplotlib seaborn unidecode
```

## Utilisation
1. Préparation des données :
```python
from my_utils import read_data, reduced_dim, dropNan_value
cleaned_data = dropNan_value(reduced_dim(data, 100000, 100000, 10725))
```

2. Prétraitement du texte :
```python
from my_utils import data_preprocessing, tokenize_pad_sequences
processed_text = data_preprocessing(data, 'tweet')
```

3. Transfert learning :
```python
from my_utils import getModel, getTransfertModel
transfer_model = getTransfertModel(getModel())
```

## Structure du projet
- `my_utils.py` - fonctions utilitaires de prétraitement et modélisation
- Modèles pré-entraînés (3 classes et 7 classes)
- Données de tweets à classifier

## Résultats
Le modèle atteint des performances significatives dans la classification fine des sentiments, démontrant l'efficacité de l'approche par transfert learning.
