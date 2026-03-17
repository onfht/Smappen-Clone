# Comparateur de zones de chalandise

Application Streamlit pour comparer deux adresses (ex. Toulouse vs Montpellier) sur un rayon circulaire ou une isochrone, puis agréger des données socio-démographiques INSEE utiles pour un business plan.

## Fonctionnalités

- comparaison de 2 adresses ;
- rayon circulaire **ou** isochrone piéton / voiture ;
- agrégation de **Filosofi 2021 – carreaux 200 m** ;
- intégration du **recensement 2021 – carreaux 1 km** ;
- 2 cartes interactives côte à côte et comparaison visuelle ;
- comptage des commerces et établissements scolaires dans le périmètre sélectionné (source OpenStreetMap / Overpass) ;
- détection automatique de **plusieurs fichiers locaux** pour le même dataset (ex. `filosofi_toulouse.gpkg` + `filosofi_montpellier.gpkg`) ;
- optimisation Streamlit : les **petits extraits locaux** sont gardés en mémoire pour accélérer les requêtes suivantes.

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

## Déploiement Streamlit : stratégie recommandée

Pour Streamlit Community Cloud, la meilleure approche est :

1. **préparer localement des extraits compacts** pour Toulouse et Montpellier ;
2. les placer dans `data/` dans le repo ;
3. pousser ces fichiers sur GitHub ;
4. laisser l'app utiliser ces fichiers **sans téléchargement INSEE au runtime**.

Le script `scripts/build_compact_city_extracts.py` sert à fabriquer ces extraits à partir des GPKG INSEE complets.

Exemple :

```bash
python scripts/build_compact_city_extracts.py   --filosofi-gpkg "C:/path/to/carreaux_200m_met.gpkg"   --rp-gpkg "C:/path/to/rp2021_carreaux_1km_met.gpkg"   --buffer-km 60   --output-dir data
```

Le script produit typiquement :

- `data/filosofi_toulouse.gpkg`
- `data/filosofi_montpellier.gpkg`
- `data/rp_toulouse.gpkg`
- `data/rp_montpellier.gpkg`

L'application détecte ensuite automatiquement ces fichiers et les combine si besoin.

## Où mettre les fichiers `.gpkg`

Tu peux déposer tes fichiers `.gpkg` dans l'un de ces dossiers :

- `data/` à la racine du projet
- `~/.cache/business_catchment_app/data/`

Pour un déploiement Cloud, privilégie **`data/` dans le repo**.

## Conseils perf

- privilégie des extraits **Toulouse / Montpellier** plutôt que les GPKG France entière ;
- si possible, garde les fichiers unitaires sous ~150–180 Mo : l'app les gardera en mémoire et les requêtes suivantes seront bien plus rapides ;
- si un fichier dépasse la limite GitHub de 100 Mo, réduis le buffer du script ou découpe davantage par zone.

## Dépendances principales

- streamlit
- pandas
- geopandas
- shapely
- pyproj
- pyogrio
- requests
- pydeck
- py7zr
- plotly

## Notes

- si aucun fichier local n'est présent, l'app peut encore retomber sur le téléchargement officiel INSEE ;
- sur Streamlit Cloud, cette voie est nettement plus lente et moins robuste ;
- les compteurs commerces / écoles restent alimentés par OpenStreetMap / Overpass.
