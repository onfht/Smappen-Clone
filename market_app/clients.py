from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GEOCODER_URL = "https://data.geopf.fr/geocodage/search"
ISOCHRONE_URL = "https://data.geopf.fr/navigation/isochrone"
GET_CAPABILITIES_URL = "https://data.geopf.fr/navigation/getCapabilities"


@dataclass(slots=True)
class GeocodedAddress:
    input_address: str
    matched_label: str
    lon: float
    lat: float
    score: float | None


class ApiError(RuntimeError):
    """Raised when an external API call fails."""


@dataclass(slots=True)
class IsochroneResult:
    payload: dict[str, Any]
    meta: dict[str, Any]


@dataclass(slots=True)
class AttemptResult:
    ok: bool
    resource: str
    profile: str | None
    method: str
    status_code: int | None
    elapsed_s: float
    detail: str
    payload: dict[str, Any] | None = None


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "business-catchment-app/1.0",
            "Accept": "application/json",
        }
    )
    return session


def geocode_address(address: str, session: requests.Session | None = None) -> GeocodedAddress:
    session = session or make_session()
    response = session.get(
        GEOCODER_URL,
        params={"q": address, "limit": 1},
        timeout=30,
    )
    if response.status_code >= 400:
        raise ApiError(f"Échec du géocodage ({response.status_code}) pour l'adresse : {address}")

    payload = response.json()
    features = payload.get("features", [])
    if not features:
        raise ApiError(f"Aucun résultat de géocodage pour : {address}")

    feature = features[0]
    coordinates = feature["geometry"]["coordinates"]
    properties = feature.get("properties", {})

    return GeocodedAddress(
        input_address=address,
        matched_label=properties.get("label", address),
        lon=float(coordinates[0]),
        lat=float(coordinates[1]),
        score=properties.get("score"),
    )


def _summarize_response_text(response: requests.Response) -> str:
    text = (response.text or "").strip().replace("\n", " ")
    return text[:500] if text else "Réponse vide"


def _build_variant_params(
    *,
    lon: float,
    lat: float,
    minutes: int,
    resource: str,
    profile: str | None,
) -> dict[str, str]:
    params = {
        "point": f"{lon},{lat}",
        "resource": resource,
        "costValue": str(minutes),
        "costType": "time",
        "direction": "departure",
        "timeUnit": "minute",
        "geometryFormat": "geojson",
        "crs": "EPSG:4326",
    }
    if profile:
        params["profile"] = profile
    return params


def _iter_isochrone_variants(profile: str) -> list[tuple[str, str | None]]:
    profile = profile.lower().strip()
    if profile == "pedestrian":
        return [
            ("bdtopo-iso-pieton", None),
            ("bdtopo-valhalla", "pedestrian"),
            ("bdtopo-osrm", "pedestrian"),
            ("bdtopo-pgr", "pedestrian"),
            ("bdtopo-pgr-pieton", None),
            ("bdtopo-osrm-pieton", None),
        ]
    if profile == "car":
        return [
            ("bdtopo-valhalla", "car"),
            ("bdtopo-osrm", "car"),
            ("bdtopo-pgr", "car"),
            ("bdtopo-pgr-voiture", None),
            ("bdtopo-osrm-voiture", None),
            ("bdtopo-iso-voiture", None),
        ]
    return [
        ("bdtopo-valhalla", profile),
        ("bdtopo-osrm", profile),
        ("bdtopo-pgr", profile),
    ]


def _try_isochrone_variant(
    *,
    lon: float,
    lat: float,
    minutes: int,
    resource: str,
    profile: str | None,
    session: requests.Session,
    http_method: str,
) -> AttemptResult:
    params = _build_variant_params(lon=lon, lat=lat, minutes=minutes, resource=resource, profile=profile)
    start = time.time()

    if http_method == "GET":
        response = session.get(ISOCHRONE_URL, params=params, timeout=(30, 120))
    elif http_method == "POST":
        response = session.post(ISOCHRONE_URL, json=params, timeout=(30, 120))
    else:
        raise ValueError(f"Méthode HTTP non supportée : {http_method}")

    elapsed = time.time() - start

    if response.status_code >= 400:
        return AttemptResult(
            ok=False,
            resource=resource,
            profile=profile,
            method=http_method,
            status_code=response.status_code,
            elapsed_s=elapsed,
            detail=_summarize_response_text(response),
        )

    try:
        payload = response.json()
    except ValueError:
        return AttemptResult(
            ok=False,
            resource=resource,
            profile=profile,
            method=http_method,
            status_code=response.status_code,
            elapsed_s=elapsed,
            detail="Réponse JSON invalide",
        )

    if not payload:
        return AttemptResult(
            ok=False,
            resource=resource,
            profile=profile,
            method=http_method,
            status_code=response.status_code,
            elapsed_s=elapsed,
            detail="Réponse vide",
        )

    return AttemptResult(
        ok=True,
        resource=resource,
        profile=profile,
        method=http_method,
        status_code=response.status_code,
        elapsed_s=elapsed,
        detail="OK",
        payload=payload,
    )


def fetch_isochrone(
    lon: float,
    lat: float,
    *,
    minutes: int,
    profile: str,
    session: requests.Session | None = None,
) -> IsochroneResult:
    """Try several official/resource-compatible variants of the IGN isochrone API."""
    session = session or make_session()

    attempts: list[AttemptResult] = []

    for resource, resource_profile in _iter_isochrone_variants(profile):
        for http_method in ("GET", "POST"):
            attempt = _try_isochrone_variant(
                lon=lon,
                lat=lat,
                minutes=minutes,
                resource=resource,
                profile=resource_profile,
                session=session,
                http_method=http_method,
            )
            attempts.append(attempt)

            if attempt.ok:
                return IsochroneResult(
                    payload=attempt.payload or {},
                    meta={
                        "used_resource": resource,
                        "used_profile": resource_profile,
                        "used_method": http_method,
                        "attempts": [asdict(attempt) for attempt in attempts],
                        "capabilities_url": GET_CAPABILITIES_URL,
                    },
                )

    compact_attempts = [
        f"{a.method} resource={a.resource} profile={a.profile or '-'} status={a.status_code} detail={a.detail}"
        for a in attempts[-6:]
    ]
    raise ApiError(
        "Échec du calcul d'isochrone sur toutes les variantes testées. "
        f"Derniers essais : {' | '.join(compact_attempts)}"
    )
