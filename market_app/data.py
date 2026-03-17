from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import shape

from .clients import make_session

try:  # optional dependency used only when INSEE ships nested .7z archives
    import py7zr  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    py7zr = None

FILOSOFI_2021_GPKG_URL = (
    "https://www.insee.fr/fr/statistiques/fichier/8735162/Filosofi2021_carreaux_200m_gpkg.zip"
)
RP_2021_1KM_GPKG_URL = (
    "https://www.insee.fr/fr/statistiques/fichier/8272002/rp2021_carreaux_1km_gpkg.zip"
)

CACHE_DIR = Path.home() / ".cache" / "business_catchment_app"
DATA_DIR = CACHE_DIR / "data"
ZIP_DIR = CACHE_DIR / "zip"
PROJECT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(slots=True)
class DatasetInfo:
    name: str
    url: str
    zip_path: Path | None
    gpkg_path: Path
    layer: str
    crs: CRS
    source: str
    bounds: tuple[float, float, float, float]


FILOSOFI_NUMERIC_COLUMNS = [
    "ind",
    "men",
    "men_pauv",
    "men_1ind",
    "men_5ind",
    "men_prop",
    "men_fmp",
    "ind_snv",
    "men_surf",
    "men_coll",
    "men_mais",
    "log_av45",
    "log_45_70",
    "log_70_90",
    "log_ap90",
    "log_inc",
    "log_soc",
    "ind_0_3",
    "ind_4_5",
    "ind_6_10",
    "ind_11_17",
    "ind_18_24",
    "ind_25_39",
    "ind_40_54",
    "ind_55_64",
    "ind_65_79",
    "ind_80p",
    "ind_inc",
    "i_est_200",
    "i_est_1km",
]

RP_NUMERIC_COLUMNS = [
    "pop",
    "pop0014",
    "pop1564",
    "pop65p",
    "popf",
    "poph",
    "popfr",
    "popue",
    "pophorsue",
    "popmigr0",
    "popmigrfr",
    "popmigrhorsfr",
    "carreau_traite_secret",
    "c21_act1564",
    "c21_act1564_cs1",
    "c21_act1564_cs2",
    "c21_act1564_cs3",
    "c21_act1564_cs4",
    "c21_act1564_cs5",
    "c21_act1564_cs6",
]


TERRITORY_FILENAME_PREFERENCES = {
    "filosofi": ["_met.gpkg", "_metropole.gpkg", "metropole"],
    "rp": ["_met.gpkg", "_metropole.gpkg", "metropole"],
}


def _normalized_name(path: Path) -> str:
    return path.name.lower().replace("-", "_")


def _territory_priority(path: Path, dataset_key: str) -> int:
    name = _normalized_name(path)
    preferences = TERRITORY_FILENAME_PREFERENCES.get(dataset_key, [])
    for rank, token in enumerate(preferences):
        if token in name:
            return rank
    if any(token in name for token in ["_mart", "martinique", "_ren", "reunion", "réunion"]):
        return 50
    return 10


def _sort_candidates(candidates: list[Path], dataset_key: str) -> list[Path]:
    return sorted(candidates, key=lambda path: (_territory_priority(path, dataset_key), _normalized_name(path)))


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    session = make_session()
    with session.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(dest)


def _safe_layers(gpkg_path: Path) -> list[str]:
    try:
        return list(fiona.listlayers(gpkg_path))
    except Exception:
        return []


def _layer_columns(gpkg_path: Path, layer: str) -> set[str]:
    with fiona.open(gpkg_path, layer=layer) as src:
        return {column.lower() for column in src.schema.get("properties", {}).keys()}


def _is_matching_dataset(gpkg_path: Path, expected_columns: set[str]) -> tuple[bool, str | None]:
    for layer in _safe_layers(gpkg_path):
        try:
            columns = _layer_columns(gpkg_path, layer)
        except Exception:
            continue
        if expected_columns.issubset(columns):
            return True, layer
    return False, None


def _project_candidate_dirs() -> list[Path]:
    dirs = [PROJECT_DATA_DIR, DATA_DIR]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def _find_local_gpkg(dataset_key: str) -> tuple[Path, str] | None:
    """Return the best local dataset candidate, or None to trigger the download path.

    Important behavior: if local .gpkg files exist but none match the expected mainland
    dataset, we do *not* raise here. We silently fall back to the official download path.
    This keeps the app usable when the project or cache folders contain stale Martinique /
    Réunion files or partial extractions from previous runs.
    """
    dataset_key = dataset_key.lower()
    if dataset_key == "filosofi":
        expected_columns = {"ind", "men"}
    elif dataset_key == "rp":
        expected_columns = {"pop", "pop0014"}
    else:
        raise ValueError(f"Dataset key inconnu: {dataset_key}")

    candidates: list[Path] = []
    for directory in _project_candidate_dirs():
        candidates.extend(directory.rglob("*.gpkg"))

    matching_candidates: list[Path] = []
    for candidate in _sort_candidates(list(set(candidates)), dataset_key):
        matches, _ = _is_matching_dataset(candidate, expected_columns)
        if matches:
            matching_candidates.append(candidate)

    if not matching_candidates:
        return None

    if dataset_key == "filosofi":
        mainland_candidates = [
            candidate
            for candidate in matching_candidates
            if _territory_priority(candidate, dataset_key) == 0
        ]
        if mainland_candidates:
            return mainland_candidates[0], "local"
        return None

    mainland_candidates = [
        candidate
        for candidate in matching_candidates
        if _territory_priority(candidate, dataset_key) == 0
    ]
    if mainland_candidates:
        return mainland_candidates[0], "local"

    return matching_candidates[0], "local"


def _extract_7z(archive_path: Path, target_dir: Path) -> list[Path]:
    if py7zr is None:
        raise RuntimeError(
            "Le package INSEE contient un .7z imbriqué. Installe la dépendance optionnelle 'py7zr' "
            "ou place directement le bon fichier .gpkg dans le dossier 'data/' du projet."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        archive.extractall(path=target_dir)
    return sorted(target_dir.rglob("*.gpkg"))


def _extract_supported_gpkg(zip_path: Path, target_dir: Path, dataset_key: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    expected_columns = {"ind", "men"} if dataset_key == "filosofi" else {"pop", "pop0014"}

    def _pick_matching_candidate(candidates: list[Path]) -> Path | None:
        matching = [
            candidate
            for candidate in _sort_candidates(candidates, dataset_key)
            if _is_matching_dataset(candidate, expected_columns)[0]
        ]
        if not matching:
            return None

        if dataset_key == "filosofi":
            mainland = [candidate for candidate in matching if _territory_priority(candidate, dataset_key) == 0]
            if mainland:
                return mainland[0]
            return None

        return matching[0]

    with zipfile.ZipFile(zip_path, "r") as archive:
        names = archive.namelist()
        gpkg_names = [name for name in names if name.lower().endswith(".gpkg")]
        gpkg_names = sorted(gpkg_names, key=lambda name: (_territory_priority(Path(name), dataset_key), name.lower()))

        extracted_candidates: list[Path] = []
        for gpkg_name in gpkg_names:
            target_path = target_dir / Path(gpkg_name).name
            if not target_path.exists():
                extracted = Path(archive.extract(gpkg_name, path=target_dir))
                if extracted != target_path:
                    shutil.move(str(extracted), str(target_path))
            extracted_candidates.append(target_path)

        matching_candidate = _pick_matching_candidate(extracted_candidates)
        if matching_candidate is not None:
            return matching_candidate

        seven_z_names = [name for name in names if name.lower().endswith(".7z")]
        for nested_name in seven_z_names:
            nested_path = target_dir / Path(nested_name).name
            if not nested_path.exists():
                extracted = Path(archive.extract(nested_name, path=target_dir))
                if extracted != nested_path:
                    shutil.move(str(extracted), str(nested_path))
            gpkg_candidates = _extract_7z(nested_path, target_dir)
            matching_candidate = _pick_matching_candidate(gpkg_candidates)
            if matching_candidate is not None:
                return matching_candidate

        if extracted_candidates or seven_z_names:
            raise RuntimeError(
                f"Des fichiers .gpkg ont bien été extraits depuis {zip_path.name}, mais aucun ne correspond au dataset attendu ({dataset_key}) pour la métropole. "
                "Cherche un nom du type 'carreaux_200m_met.gpkg'."
            )

    raise RuntimeError(
        f"Aucun fichier .gpkg exploitable trouvé dans {zip_path.name}. "
        f"Place le bon .gpkg manuellement dans {PROJECT_DATA_DIR} ou {DATA_DIR}."
    )


def _dataset_info(name: str, url: str, dataset_key: str) -> DatasetInfo:
    local_match = _find_local_gpkg(dataset_key)
    if local_match is not None:
        gpkg_path, source = local_match
        zip_path = None
    else:
        zip_path = ZIP_DIR / Path(url).name
        if not zip_path.exists():
            _download(url, zip_path)
        gpkg_path = _extract_supported_gpkg(zip_path, DATA_DIR, dataset_key)
        source = "download"

    expected_columns = {"ind", "men"} if dataset_key == "filosofi" else {"pop", "pop0014"}
    matches, layer = _is_matching_dataset(gpkg_path, expected_columns)
    if not matches or layer is None:
        raise RuntimeError(
            f"Le fichier {gpkg_path.name} ne correspond pas au dataset attendu ({dataset_key}). "
            "Vérifie que tu as copié le bon .gpkg."
        )

    with fiona.open(gpkg_path, layer=layer) as src:
        crs = CRS.from_user_input(src.crs_wkt or src.crs)
        bounds = tuple(float(value) for value in src.bounds)

    return DatasetInfo(
        name=name,
        url=url,
        zip_path=zip_path,
        gpkg_path=gpkg_path,
        layer=layer,
        crs=crs,
        source=source,
        bounds=bounds,
    )


def ensure_filosofi_dataset() -> DatasetInfo:
    return _dataset_info("Filosofi 2021 - carreaux 200 m", FILOSOFI_2021_GPKG_URL, "filosofi")


def ensure_rp_dataset() -> DatasetInfo:
    return _dataset_info("Recensement 2021 - carreaux 1 km", RP_2021_1KM_GPKG_URL, "rp")


def _to_numeric(gdf: gpd.GeoDataFrame, columns: list[str]) -> gpd.GeoDataFrame:
    for column in columns:
        if column in gdf.columns:
            gdf[column] = pd.to_numeric(gdf[column], errors="coerce").fillna(0.0)
    return gdf


def load_subset(dataset: DatasetInfo, zone_geometry) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(dataset.gpkg_path, layer=dataset.layer, bbox=zone_geometry.bounds)
    if gdf.empty:
        return gdf

    gdf.columns = [column.lower() for column in gdf.columns]
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf.set_geometry(gdf.geometry.buffer(0))

    if "ind" in gdf.columns:
        gdf = _to_numeric(gdf, FILOSOFI_NUMERIC_COLUMNS)
    elif "pop" in gdf.columns:
        gdf = _to_numeric(gdf, RP_NUMERIC_COLUMNS)

    return gdf


def geojson_to_projected_geometry(geojson_payload: dict, target_crs: CRS):
    """Convert several possible API response shapes into a projected shapely geometry."""
    if "type" in geojson_payload and geojson_payload["type"] == "FeatureCollection":
        features = geojson_payload.get("features", [])
        if not features:
            raise RuntimeError("La réponse GeoJSON ne contient aucune feature.")
        raw_geometry = features[0]["geometry"]
    elif "features" in geojson_payload:
        features = geojson_payload.get("features", [])
        if not features:
            raise RuntimeError("La réponse GeoJSON ne contient aucune feature.")
        raw_geometry = features[0]["geometry"]
    elif "geometry" in geojson_payload:
        raw_geometry = geojson_payload["geometry"]
    else:
        raw_geometry = geojson_payload

    geom_wgs84 = gpd.GeoSeries([shape(raw_geometry)], crs="EPSG:4326")
    return geom_wgs84.to_crs(target_crs).iloc[0]
