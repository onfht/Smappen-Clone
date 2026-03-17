from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
from pyogrio import list_layers as pyogrio_list_layers, read_info
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
COMPACT_TOKENS = ("toulouse", "montpellier", "compact", "streamlit", "tm_")
FULL_LOAD_CACHE_MAX_MB = 180


@dataclass(slots=True)
class DatasetPart:
    path: Path
    layer: str
    crs: CRS
    source: str
    bounds: tuple[float, float, float, float]
    size_mb: float


@dataclass(slots=True)
class DatasetInfo:
    name: str
    url: str
    zip_path: Path | None
    parts: tuple[DatasetPart, ...]
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


def _size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 2)
    except FileNotFoundError:
        return 0.0


def _candidate_rank(path: Path, dataset_key: str) -> tuple[int, int, float, str]:
    name = _normalized_name(path)
    compact_rank = 0 if any(token in name for token in COMPACT_TOKENS) else 1
    return (compact_rank, _territory_priority(path, dataset_key), _size_mb(path), name)


def _sort_candidates(candidates: list[Path], dataset_key: str) -> list[Path]:
    return sorted(candidates, key=lambda path: _candidate_rank(path, dataset_key))


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
        return [str(name) for name, _geometry_type in pyogrio_list_layers(gpkg_path)]
    except Exception:
        return []


def _layer_columns(gpkg_path: Path, layer: str) -> set[str]:
    info = read_info(gpkg_path, layer=layer)
    return {str(column).lower() for column in info.get("fields", [])}


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


def _matching_local_candidates(dataset_key: str, directories: list[Path]) -> list[Path]:
    if dataset_key == "filosofi":
        expected_columns = {"ind", "men"}
    elif dataset_key == "rp":
        expected_columns = {"pop", "pop0014"}
    else:
        raise ValueError(f"Dataset key inconnu: {dataset_key}")

    candidates: list[Path] = []
    for directory in directories:
        if directory.exists():
            candidates.extend(directory.rglob("*.gpkg"))

    matching_candidates: list[Path] = []
    for candidate in _sort_candidates(list(set(candidates)), dataset_key):
        matches, _ = _is_matching_dataset(candidate, expected_columns)
        if matches:
            matching_candidates.append(candidate)
    return matching_candidates


def _find_local_gpkg_parts(dataset_key: str) -> list[Path]:
    """Return matching local datasets.

    Behavior optimized for deployment:
    - if project data/ contains one or several matching files, use those only;
    - if some project files are clearly compact extracts (toulouse / montpellier / compact),
      prefer all of them together so the app can compare both cities without touching the
      heavyweight cache;
    - otherwise, if nothing exists in the project, fall back to the best cache candidate.
    """
    project_candidates = _matching_local_candidates(dataset_key, [PROJECT_DATA_DIR])
    if project_candidates:
        compact_project_candidates = [
            path for path in project_candidates if any(token in _normalized_name(path) for token in COMPACT_TOKENS)
        ]
        selected = compact_project_candidates or project_candidates
        return _sort_candidates(selected, dataset_key)

    cache_candidates = _matching_local_candidates(dataset_key, [DATA_DIR])
    if not cache_candidates:
        return []

    mainland_candidates = [
        candidate for candidate in cache_candidates if _territory_priority(candidate, dataset_key) == 0
    ]
    ranked = _sort_candidates(mainland_candidates or cache_candidates, dataset_key)
    return ranked[:1]


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
        gpkg_names = sorted(gpkg_names, key=lambda name: _candidate_rank(Path(name), dataset_key))

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


def _build_part(gpkg_path: Path, source: str, expected_columns: set[str]) -> DatasetPart:
    matches, layer = _is_matching_dataset(gpkg_path, expected_columns)
    if not matches or layer is None:
        raise RuntimeError(
            f"Le fichier {gpkg_path.name} ne correspond pas au dataset attendu. "
            "Vérifie que tu as copié le bon .gpkg."
        )
    info = read_info(gpkg_path, layer=layer)
    crs = CRS.from_user_input(info["crs"])
    bounds = tuple(float(value) for value in info["total_bounds"])
    return DatasetPart(
        path=gpkg_path,
        layer=layer,
        crs=crs,
        source=source,
        bounds=bounds,
        size_mb=_size_mb(gpkg_path),
    )


def _dataset_info(name: str, url: str, dataset_key: str) -> DatasetInfo:
    expected_columns = {"ind", "men"} if dataset_key == "filosofi" else {"pop", "pop0014"}
    local_matches = _find_local_gpkg_parts(dataset_key)

    if local_matches:
        zip_path = None
        parts = tuple(_build_part(path, "local", expected_columns) for path in local_matches)
        source = "local"
    else:
        zip_path = ZIP_DIR / Path(url).name
        if not zip_path.exists():
            _download(url, zip_path)
        gpkg_path = _extract_supported_gpkg(zip_path, DATA_DIR, dataset_key)
        parts = (_build_part(gpkg_path, "download", expected_columns),)
        source = "download"

    if not parts:
        raise RuntimeError(f"Aucune source locale ou téléchargée exploitable pour {name}.")

    first_crs = parts[0].crs
    if any(part.crs != first_crs for part in parts[1:]):
        raise RuntimeError("Les fichiers locaux détectés n'ont pas tous la même projection.")

    bounds = (
        min(part.bounds[0] for part in parts),
        min(part.bounds[1] for part in parts),
        max(part.bounds[2] for part in parts),
        max(part.bounds[3] for part in parts),
    )

    return DatasetInfo(
        name=name,
        url=url,
        zip_path=zip_path,
        parts=parts,
        crs=first_crs,
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


def _normalize_subset(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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


def _should_keep_full_part_in_memory(part: DatasetPart) -> bool:
    return part.size_mb > 0 and part.size_mb <= FULL_LOAD_CACHE_MAX_MB


@lru_cache(maxsize=8)
def _load_full_part_cached(path_str: str, layer: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path_str, layer=layer, engine="pyogrio")
    return _normalize_subset(gdf)


def _bbox_intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def load_subset(dataset: DatasetInfo, zone_geometry) -> gpd.GeoDataFrame:
    zone_bounds = tuple(float(v) for v in zone_geometry.bounds)
    frames: list[gpd.GeoDataFrame] = []

    for part in dataset.parts:
        if not _bbox_intersects(part.bounds, zone_bounds):
            continue

        if _should_keep_full_part_in_memory(part):
            gdf = _load_full_part_cached(str(part.path), part.layer)
            if gdf.empty:
                continue
            subset = gdf[gdf.intersects(zone_geometry)].copy()
        else:
            gdf = gpd.read_file(part.path, layer=part.layer, bbox=zone_bounds, engine="pyogrio")
            subset = _normalize_subset(gdf)

        if not subset.empty:
            frames.append(subset)

    if not frames:
        return gpd.GeoDataFrame(geometry=[], crs=dataset.crs)

    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=dataset.crs)


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
