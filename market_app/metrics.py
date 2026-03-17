from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import geopandas as gpd
import pandas as pd


@dataclass(slots=True)
class AggregationResult:
    raw: dict[str, float]
    derived: dict[str, float]
    intersected_cells: int
    zone_area_km2: float


def _weighted_subset(gdf: gpd.GeoDataFrame, zone_geometry) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    subset = gdf[gdf.intersects(zone_geometry)].copy()
    if subset.empty:
        return subset

    subset["cell_area"] = subset.geometry.area
    subset["inter_area"] = subset.geometry.intersection(zone_geometry).area
    subset = subset[subset["inter_area"] > 0].copy()
    if subset.empty:
        return subset
    subset["weight"] = (subset["inter_area"] / subset["cell_area"]).clip(lower=0, upper=1)
    return subset


def _safe_div(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _round_payload(payload: dict[str, float | None], digits: int = 4) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for key, value in payload.items():
        if value is None:
            output[key] = None
        else:
            output[key] = round(float(value), digits)
    return output


def aggregate_filosofi(gdf: gpd.GeoDataFrame, zone_geometry) -> AggregationResult:
    subset = _weighted_subset(gdf, zone_geometry)
    zone_area_km2 = zone_geometry.area / 1_000_000

    if subset.empty:
        return AggregationResult(raw={}, derived={}, intersected_cells=0, zone_area_km2=zone_area_km2)

    fields = [
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
    ]

    raw = {
        field: float((subset[field] * subset["weight"]).sum())
        for field in fields
        if field in subset.columns
    }

    population = raw.get("ind", 0.0)
    households = raw.get("men", 0.0)
    age_0_17 = sum(raw.get(field, 0.0) for field in ["ind_0_3", "ind_4_5", "ind_6_10", "ind_11_17"])
    age_18_39 = raw.get("ind_18_24", 0.0) + raw.get("ind_25_39", 0.0)
    age_40_64 = raw.get("ind_40_54", 0.0) + raw.get("ind_55_64", 0.0)
    age_65_plus = raw.get("ind_65_79", 0.0) + raw.get("ind_80p", 0.0)

    imputation_population_share = None
    if "i_est_200" in subset.columns and population > 0:
        imputation_population_share = float(
            ((subset["ind"] * subset["weight"] * subset["i_est_200"]).sum()) / population
        )

    derived = {
        "zone_area_km2": zone_area_km2,
        "population_density": _safe_div(population, zone_area_km2),
        "avg_household_size": _safe_div(population, households),
        "poverty_rate_households": _safe_div(raw.get("men_pauv", 0.0), households),
        "single_person_household_rate": _safe_div(raw.get("men_1ind", 0.0), households),
        "large_household_rate": _safe_div(raw.get("men_5ind", 0.0), households),
        "owner_rate": _safe_div(raw.get("men_prop", 0.0), households),
        "single_parent_rate": _safe_div(raw.get("men_fmp", 0.0), households),
        "collective_housing_rate": _safe_div(raw.get("men_coll", 0.0), households),
        "house_rate": _safe_div(raw.get("men_mais", 0.0), households),
        "social_housing_rate": _safe_div(raw.get("log_soc", 0.0), households),
        "avg_dwelling_area_m2_per_household": _safe_div(raw.get("men_surf", 0.0), households),
        "avg_winsorized_living_standard_per_person": _safe_div(raw.get("ind_snv", 0.0), population),
        "share_0_17": _safe_div(age_0_17, population),
        "share_18_39": _safe_div(age_18_39, population),
        "share_40_64": _safe_div(age_40_64, population),
        "share_65_plus": _safe_div(age_65_plus, population),
        "pre_1945_housing_rate": _safe_div(raw.get("log_av45", 0.0), households),
        "housing_1945_1969_rate": _safe_div(raw.get("log_45_70", 0.0), households),
        "housing_1970_1989_rate": _safe_div(raw.get("log_70_90", 0.0), households),
        "housing_1990_plus_rate": _safe_div(raw.get("log_ap90", 0.0), households),
        "imputation_population_share": imputation_population_share,
    }

    return AggregationResult(
        raw=_round_payload(raw),
        derived=_round_payload(derived),
        intersected_cells=int(len(subset)),
        zone_area_km2=zone_area_km2,
    )


def aggregate_rp2021(gdf: gpd.GeoDataFrame, zone_geometry) -> AggregationResult:
    subset = _weighted_subset(gdf, zone_geometry)
    zone_area_km2 = zone_geometry.area / 1_000_000

    if subset.empty:
        return AggregationResult(raw={}, derived={}, intersected_cells=0, zone_area_km2=zone_area_km2)

    fields = [
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
        "c21_act1564",
        "c21_act1564_cs1",
        "c21_act1564_cs2",
        "c21_act1564_cs3",
        "c21_act1564_cs4",
        "c21_act1564_cs5",
        "c21_act1564_cs6",
    ]

    raw = {
        field: float((subset[field] * subset["weight"]).sum())
        for field in fields
        if field in subset.columns
    }

    population = raw.get("pop", 0.0)
    born_abroad = raw.get("popue", 0.0) + raw.get("pophorsue", 0.0)
    moved = raw.get("popmigrfr", 0.0) + raw.get("popmigrhorsfr", 0.0)

    secrecy_share = None
    if "carreau_traite_secret" in subset.columns and population > 0 and "pop" in subset.columns:
        secrecy_share = float(
            ((subset["pop"] * subset["weight"] * subset["carreau_traite_secret"]).sum()) / population
        )

    csp_total = raw.get("c21_act1564", 0.0)
    if csp_total == 0:
        csp_total = sum(raw.get(f"c21_act1564_cs{i}", 0.0) for i in range(1, 7))

    derived = {
        "share_under_15": _safe_div(raw.get("pop0014", 0.0), population),
        "share_15_64": _safe_div(raw.get("pop1564", 0.0), population),
        "share_65_plus": _safe_div(raw.get("pop65p", 0.0), population),
        "female_share": _safe_div(raw.get("popf", 0.0), population),
        "born_abroad_share": _safe_div(born_abroad, population),
        "moved_from_elsewhere_share": _safe_div(moved, population),
        "secrecy_treated_population_share": secrecy_share,
        "csp_cs1_share": _safe_div(raw.get("c21_act1564_cs1", 0.0), csp_total),
        "csp_cs2_share": _safe_div(raw.get("c21_act1564_cs2", 0.0), csp_total),
        "csp_cs3_share": _safe_div(raw.get("c21_act1564_cs3", 0.0), csp_total),
        "csp_cs4_share": _safe_div(raw.get("c21_act1564_cs4", 0.0), csp_total),
        "csp_cs5_share": _safe_div(raw.get("c21_act1564_cs5", 0.0), csp_total),
        "csp_cs6_share": _safe_div(raw.get("c21_act1564_cs6", 0.0), csp_total),
    }

    return AggregationResult(
        raw=_round_payload(raw),
        derived=_round_payload(derived),
        intersected_cells=int(len(subset)),
        zone_area_km2=zone_area_km2,
    )


def build_comparison_frame(
    address_results: dict[str, dict],
    metric_paths: Iterable[tuple[str, str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, block_name, metric_name in metric_paths:
        row: dict[str, object] = {"metric": label}
        for address_name, payload in address_results.items():
            value = payload.get(block_name, {}).get(metric_name)
            row[address_name] = value
        rows.append(row)
    return pd.DataFrame(rows)
