# Comparateur de zones de chalandise

Application Streamlit pour comparer deux adresses (ex. Toulouse vs Montpellier) sur un rayon circulaire ou une isochrone, puis agréger des données socio-démographiques INSEE utiles pour un business plan.

## Fonctionnalités

- comparaison de 2 adresses ;
- rayon circulaire **ou** isochrone piéton / voiture ;
- agrégation de **Filosofi 2021 – carreaux 200 m** ;
- intégration du **recensement 2021 – carreaux 1 km** ;
- 2 cartes interactives côte à côte, synthèse visuelle et export CSV ;
- comptage des commerces et établissements scolaires dans le périmètre sélectionné (source OpenStreetMap / Overpass) ;
- détection automatique des bons `.gpkg` locaux avant tout téléchargement.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Lancement

```bash
python -m streamlit run app.py
```

## Où mettre les fichiers `.gpkg`

Tu peux déposer tes fichiers `.gpkg` dans l'un de ces dossiers :

- `data/` à la racine du projet
- `C:\Users\<toi>\.cache\business_catchment_app\data\`

L'application inspecte les colonnes des fichiers pour reconnaître automatiquement :

- le bon **Filosofi 2021** (`ind`, `men`, etc.)
- le bon **Recensement 2021** (`pop`, `pop0014`, etc.)

Pour **Filosofi 200 m**, privilégie explicitement le fichier **métropole**, typiquement nommé `carreaux_200m_met.gpkg`.
L'app privilégie désormais ce fichier si plusieurs `.gpkg` sont présents, et refuse de sélectionner silencieusement une variante Martinique / Réunion.

## Notes importantes

- Si l'archive INSEE contient un `.7z` imbriqué, l'application peut l'extraire si la dépendance optionnelle `py7zr` est installée.
- Si tu as déjà extrait manuellement les bons `.gpkg`, il est préférable de les coller dans `data/` et de laisser l'app les utiliser directement.
- L'export CSV se fait désormais en mémoire : plus de dépendance à `/tmp`, donc plus d'erreur Windows sur ce point.

## Dépendances principales

- streamlit
- pandas
- geopandas
- shapely
- pyproj
- fiona
- requests
- pydeck
- py7zr
- plotly


## Troubleshooting récent
- **Erreur Streamlit `UnhashableParamError`** : corrigée dans cette version en évitant de passer un objet `DatasetInfo` non hachable au cache de `build_zone()`.
- **Export CSV Windows** : l'export se fait en mémoire, sans écriture vers `/tmp`.
- **Archives INSEE avec `.7z` imbriqué** : l'app sait maintenant sélectionner le bon `.gpkg` en vérifiant les colonnes attendues du dataset et en privilégiant explicitement le fichier **métropole**.
- **Population = 0 partout** : vérifie d'abord que le fichier chargé est bien un `*_met.gpkg`, que le point géocodé tombe dans la zone du dataset et que la projection du périmètre est correcte.


## Extension récente

- la synthèse textuelle a été remplacée par une **synthèse visuelle** sous forme de diagrammes ;
- la table de comparaison inclut les **salons de thé**, **boulangeries/pâtisseries**, **restaurants**, **commerces**, ainsi que les établissements scolaires par niveau ;
- les cartes peuvent afficher ou masquer de très petits marqueurs pour les écoles, salons de thé et boulangeries/pâtisseries ;
- le bloc “Résumé géocodage”, l'encart de premier lancement et la section “Détails par adresse” ont été supprimés pour alléger la page.
