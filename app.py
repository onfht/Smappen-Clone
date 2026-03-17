from __future__ import annotations

from dataclasses import asdict
import json
import math
import re

import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st
import plotly.graph_objects as go
from shapely.geometry import Point, shape

from market_app.clients import ApiError, fetch_isochrone, geocode_address, make_session
from market_app.data import (
    ensure_filosofi_dataset,
    ensure_rp_dataset,
    geojson_to_projected_geometry,
    load_subset,
)
from market_app.metrics import aggregate_filosofi, aggregate_rp2021, build_comparison_frame

st.set_page_config(
    page_title="Comparateur de zones de chalandise",
    page_icon="🍪",
    layout="wide",
)

DEFAULT_A = "5 Rue Léon Gambetta, 31000 Toulouse"
DEFAULT_B = "8 Rue du Faubourg du Courreau, 34000 Montpellier"
CARTO_LIGHT_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
ADDRESS_COLORS = {"Adresse A": "#D86D8C", "Adresse B": "#0077B6"}
ADDRESS_FILL_COLORS = {"Adresse A": [216, 109, 140, 120], "Adresse B": [0, 119, 182, 120]}

PERCENT_METRICS = {
    "poverty_rate_households",
    "single_person_household_rate",
    "large_household_rate",
    "owner_rate",
    "single_parent_rate",
    "collective_housing_rate",
    "house_rate",
    "social_housing_rate",
    "share_0_17",
    "share_18_39",
    "share_40_64",
    "share_65_plus",
    "pre_1945_housing_rate",
    "housing_1945_1969_rate",
    "housing_1970_1989_rate",
    "housing_1990_plus_rate",
    "imputation_population_share",
    "share_under_15",
    "share_15_64",
    "female_share",
    "born_abroad_share",
    "moved_from_elsewhere_share",
    "secrecy_treated_population_share",
    "csp_cs1_share",
    "csp_cs2_share",
    "csp_cs3_share",
    "csp_cs4_share",
    "csp_cs5_share",
    "csp_cs6_share",
}

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

POI_LABELS = {
    "tea_rooms": "Salons de thé",
    "bakeries_pastries": "Boulangeries & pâtisseries",
    "restaurants": "Restaurants",
    "shops": "Commerces",
    "kindergartens": "Écoles maternelles",
    "primary_schools": "Écoles primaires",
    "middle_schools": "Collèges",
    "high_schools": "Lycées",
    "higher_education": "Supérieur",
    "schools_total": "Total établissements scolaires",
}

INTEGER_METRICS = {
    "ind",
    "men",
    "tea_rooms",
    "bakeries_pastries",
    "restaurants",
    "shops",
    "kindergartens",
    "primary_schools",
    "middle_schools",
    "high_schools",
    "higher_education",
    "schools_total",
}

INCLUDE_CENSUS = True
MAP_POI_CATEGORIES = [
    "kindergartens",
    "primary_schools",
    "middle_schools",
    "high_schools",
    "higher_education",
    "tea_rooms",
    "bakeries_pastries",
]
MAP_POI_COLORS = {
    "kindergartens": [244, 143, 177, 220],
    "primary_schools": [255, 202, 40, 220],
    "middle_schools": [102, 187, 106, 220],
    "high_schools": [121, 85, 72, 220],
    "higher_education": [57, 73, 171, 220],
    "tea_rooms": [0, 137, 123, 220],
    "bakeries_pastries": [251, 140, 0, 220],
}


def _safe_lower(value) -> str:
    return str(value or "").strip().lower()


def _extract_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        parsed = int(round(float(value)))
        return parsed if parsed >= 0 else None
    match = re.search(r"\d+", str(value).replace(" ", ""))
    if not match:
        return None
    parsed = int(match.group(0))
    return parsed if parsed >= 0 else None


def _element_identifier(element: dict) -> str:
    return f"{element.get('type', 'x')}/{element.get('id', '0')}"


def _element_lon_lat(element: dict):
    if element.get("type") == "node":
        return element.get("lon"), element.get("lat")
    center = element.get("center") or {}
    return center.get("lon"), center.get("lat")


def _build_poi_point(category_key: str, lon: float, lat: float, element: dict, tags: dict) -> dict:
    name = tags.get("name") or POI_LABELS.get(category_key, category_key)
    return {
        "id": _element_identifier(element),
        "coordinates": [lon, lat],
        "name": name,
        "label": tags.get("name") or POI_LABELS.get(category_key, category_key),
        "category": category_key,
        "category_label": POI_LABELS.get(category_key, category_key),
    }


def _is_tea_room(tags: dict) -> bool:
    amenity = _safe_lower(tags.get("amenity"))
    shop = _safe_lower(tags.get("shop"))
    cuisine = _safe_lower(tags.get("cuisine"))
    name = _safe_lower(tags.get("name"))
    combined = " ".join(filter(None, [amenity, shop, cuisine, name]))
    if amenity == "tea_room" or shop == "tea":
        return True
    if "bubble_tea" in cuisine or "bubble tea" in combined:
        return True
    tea_keywords = ["salon de thé", "salon de the", "tea room", "thé", " the ", "tea "]
    if any(keyword in combined for keyword in tea_keywords):
        return True
    if amenity == "cafe" and "tea" in cuisine:
        return True
    return False


def _is_bakery_or_pastry(tags: dict) -> bool:
    shop = _safe_lower(tags.get("shop"))
    name = _safe_lower(tags.get("name"))
    if shop in {"bakery", "pastry", "confectionery"}:
        return True
    return any(keyword in name for keyword in ["boulangerie", "pâtisserie", "patisserie", "viennoiserie"])


def _classify_school(tags: dict) -> str | None:
    amenity = _safe_lower(tags.get("amenity"))
    school_fr = _safe_lower(tags.get("school:FR") or tags.get("school:fr"))
    school = _safe_lower(tags.get("school"))
    isced = _safe_lower(tags.get("isced:level"))
    name = _safe_lower(tags.get("name"))
    operator_type = _safe_lower(tags.get("operator:type"))
    combined = " | ".join(filter(None, [amenity, school_fr, school, isced, name, operator_type]))

    if amenity == "kindergarten" or any(keyword in combined for keyword in ["maternelle", "preschool", "kindergarten"]):
        return "kindergartens"
    if amenity == "university" or any(keyword in combined for keyword in [
        "universit", "facult", "campus", "iut", "insa", "ecole d'ing", "école d'ing",
        "ecole supérieure", "école supérieure", "grande école", "grand ecole",
        "business school", "enseignement supérieur",
    ]):
        return "higher_education"
    if amenity == "college" or "collège" in combined or "college" in combined:
        return "middle_schools"
    if "lycée" in combined or "lycee" in combined:
        return "high_schools"
    if amenity == "school" or "école" in combined or "ecole" in combined:
        if any(keyword in combined for keyword in ["maternelle", "preschool", "kindergarten"]):
            return "kindergartens"
        return "primary_schools"
    return None


def _overpass_query_for_bbox(minx: float, miny: float, maxx: float, maxy: float) -> str:
    bbox = f"{miny},{minx},{maxy},{maxx}"
    return f"""
[out:json][timeout:60];
(
  nwr["amenity"="restaurant"]({bbox});
  nwr["amenity"="cafe"]({bbox});
  nwr["shop"]({bbox});
  nwr["amenity"="kindergarten"]({bbox});
  nwr["amenity"="school"]({bbox});
  nwr["amenity"="college"]({bbox});
  nwr["amenity"="university"]({bbox});
);
out center tags qt;
"""


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_poi_summary(zone_geojson_json: str):
    feature_collection = json.loads(zone_geojson_json)
    zone_shape = shape(feature_collection["features"][0]["geometry"])
    minx, miny, maxx, maxy = zone_shape.bounds
    query = _overpass_query_for_bbox(minx, miny, maxx, maxy)

    last_error = None
    payload = None
    session = get_session()
    headers = {"Content-Type": "text/plain; charset=utf-8"}
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            response = session.post(endpoint, data=query.encode("utf-8"), headers=headers, timeout=90)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:  # pragma: no cover - depends on live services
            last_error = exc

    if payload is None:
        raise RuntimeError(f"Impossible de récupérer les points d'intérêt : {last_error}")

    counts = {key: 0 for key in POI_LABELS}
    id_sets = {
        "tea_rooms": set(),
        "bakeries_pastries": set(),
        "restaurants": set(),
        "shops": set(),
        "kindergartens": set(),
        "primary_schools": set(),
        "middle_schools": set(),
        "high_schools": set(),
        "higher_education": set(),
    }
    poi_points = {key: [] for key in MAP_POI_CATEGORIES}
    poi_point_ids = {key: set() for key in MAP_POI_CATEGORIES}

    for element in payload.get("elements", []):
        lon, lat = _element_lon_lat(element)
        if lon is None or lat is None:
            continue
        if not zone_shape.covers(Point(lon, lat)):
            continue

        tags = element.get("tags") or {}
        element_id = _element_identifier(element)
        amenity = _safe_lower(tags.get("amenity"))
        shop = _safe_lower(tags.get("shop"))

        if amenity == "restaurant":
            id_sets["restaurants"].add(element_id)
        if shop:
            id_sets["shops"].add(element_id)
        if _is_tea_room(tags):
            id_sets["tea_rooms"].add(element_id)
            if element_id not in poi_point_ids["tea_rooms"]:
                poi_points["tea_rooms"].append(_build_poi_point("tea_rooms", lon, lat, element, tags))
                poi_point_ids["tea_rooms"].add(element_id)
        if _is_bakery_or_pastry(tags):
            id_sets["bakeries_pastries"].add(element_id)
            if element_id not in poi_point_ids["bakeries_pastries"]:
                poi_points["bakeries_pastries"].append(_build_poi_point("bakeries_pastries", lon, lat, element, tags))
                poi_point_ids["bakeries_pastries"].add(element_id)

        school_category = _classify_school(tags)
        if school_category:
            id_sets[school_category].add(element_id)
            if school_category in poi_points and element_id not in poi_point_ids[school_category]:
                poi_points[school_category].append(_build_poi_point(school_category, lon, lat, element, tags))
                poi_point_ids[school_category].add(element_id)

    for key, values in id_sets.items():
        counts[key] = len(values)

    counts["schools_total"] = sum(
        counts[key]
        for key in ["kindergartens", "primary_schools", "middle_schools", "high_schools", "higher_education"]
    )

    meta = {
        "sources": {
            "poi_counts": "OpenStreetMap / Overpass",
            "schools": "OpenStreetMap / Overpass",
        },
        "elements_in_bbox": len(payload.get("elements", [])),
        "status": "ok",
    }
    return counts, poi_points, meta


def _safe_number(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _format_delta_ratio(a_value, b_value, metric_key: str) -> str:
    a_num = _safe_number(a_value)
    b_num = _safe_number(b_value)
    if a_num is None or b_num is None:
        return "—"
    if b_num == 0:
        if a_num == 0:
            return "1,00x"
        return "∞"
    return f"{(a_num / b_num):.2f}x".replace(".", ",")


def _format_abs_and_relative_gap(high: float, low: float, metric_key: str) -> tuple[str, str | None]:
    abs_gap = high - low
    rel_gap = ((abs_gap / low) * 100) if low > 0 else None
    if metric_key in PERCENT_METRICS:
        abs_text = f"{abs_gap * 100:.1f} pts".replace(".", ",")
        rel_text = f"{rel_gap:.1f}%".replace(".", ",") if rel_gap is not None else None
        return abs_text, rel_text
    if "standard" in metric_key:
        abs_text = f"€{abs_gap:,.0f}".replace(",", " ")
    elif metric_key in INTEGER_METRICS or abs_gap >= 100:
        abs_text = f"{abs_gap:,.0f}".replace(",", " ")
    else:
        abs_text = f"{abs_gap:.2f}".replace(".", ",")
    rel_text = f"{rel_gap:.1f}%".replace(".", ",") if rel_gap is not None else None
    return abs_text, rel_text


def _metric_delta_text(a_value, b_value, *, percent=False, currency=False):
    if a_value is None or b_value is None or pd.isna(a_value) or pd.isna(b_value):
        return None
    a_value = float(a_value)
    b_value = float(b_value)
    if a_value == b_value:
        return "les deux adresses sont à parité"

    metric_key = "share_18_39" if percent else "ind"
    winner = "Adresse A" if a_value > b_value else "Adresse B"
    high = max(a_value, b_value)
    low = min(a_value, b_value)
    abs_gap, rel_gap = _format_abs_and_relative_gap(high, low, metric_key)

    def fmt(value: float) -> str:
        if currency:
            return f"€{value:,.0f}".replace(",", " ")
        if value >= 100:
            return f"{value:,.0f}".replace(",", " ")
        return f"{value:.2f}".replace(".", ",")

    if percent:
        if rel_gap is not None:
            return f"{winner} ressort devant avec {high:.1%} contre {low:.1%}, soit +{abs_gap} (+{rel_gap})."
        return f"{winner} ressort devant avec {high:.1%} contre {low:.1%}, soit +{abs_gap}."

    abs_gap_text = f"€{abs_gap}" if currency else abs_gap
    if rel_gap is not None:
        return f"{winner} ressort devant avec {fmt(high)} contre {fmt(low)}, soit +{abs_gap_text} (+{rel_gap})."
    return f"{winner} ressort devant avec {fmt(high)} contre {fmt(low)}, soit +{abs_gap_text}."


def make_manual_comparison_rows(results: dict[str, dict], rows_spec: list[tuple[str, str, str, str]]):
    rows = []
    for category, label, payload_key, metric_key in rows_spec:
        row = {"category": category, "metric": label}
        for address_name, payload in results.items():
            row[address_name] = (payload.get(payload_key) or {}).get(metric_key)
        rows.append(row)
    return pd.DataFrame(rows)


def format_comparison_frame(df: pd.DataFrame, metric_key_map: dict[str, str]) -> pd.DataFrame:
    formatted_rows = []
    for _, row in df.iterrows():
        metric_name = row["metric"]
        metric_key = metric_key_map.get(metric_name, metric_name)
        formatted_rows.append(
            {
                "category": row.get("category"),
                "Indicateur": metric_name,
                "Adresse A": format_metric(metric_key, row.get("Adresse A")),
                "Adresse B": format_metric(metric_key, row.get("Adresse B")),
                "Delta": _format_delta_ratio(row.get("Adresse A"), row.get("Adresse B"), metric_key),
            }
        )
    return pd.DataFrame(formatted_rows)



def format_metric(metric_name: str, value) -> str:
    if value is None or pd.isna(value):
        return "—"
    if metric_name in PERCENT_METRICS:
        return f"{value:.1%}"
    if metric_name in INTEGER_METRICS:
        return f"{int(round(float(value))):,}".replace(",", " ")
    if "standard" in metric_name:
        return f"€{value:,.0f}".replace(",", " ")
    if "density" in metric_name:
        return f"{value:,.0f}".replace(",", " ")
    if "area" in metric_name:
        return f"{value:,.1f}".replace(",", " ")
    if value >= 100:
        return f"{value:,.0f}".replace(",", " ")
    return f"{value:.2f}".replace(",", " ")


@st.cache_resource(show_spinner="Téléchargement / préparation du dataset Filosofi 2021…")
def get_filosofi_dataset():
    return ensure_filosofi_dataset()


@st.cache_resource(show_spinner="Téléchargement / préparation du recensement 2021…")
def get_rp_dataset():
    return ensure_rp_dataset()


@st.cache_resource
def get_session():
    return make_session()


@st.cache_data(show_spinner=False)
def make_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _rgba_to_hex(color: list[int]) -> str:
    return "#" + "".join(f"{int(channel):02X}" for channel in color[:3])


def _plotly_base_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def _render_metric_kpi(title: str, metric_key: str, a_value, b_value):
    a_num = _safe_number(a_value)
    b_num = _safe_number(b_value)
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=0 if a_num is None else a_num,
            number={"valueformat": ".1%" if metric_key in PERCENT_METRICS else ",.0f"},
            delta={"reference": 0 if b_num is None else b_num, "relative": True, "valueformat": ".1%"},
            title={"text": title},
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    _plotly_base_layout(fig, height=165)
    fig.update_traces(number_font_size=28, title_font_size=14)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        f"A : {format_metric(metric_key, a_value)} | "
        f"B : {format_metric(metric_key, b_value)} | "
        f"Delta : {_format_delta_ratio(a_value, b_value, metric_key)}"
    )


def _display_value_for_chart(metric_key: str, value) -> float | None:
    value = _safe_number(value)
    if value is None:
        return None
    if metric_key in PERCENT_METRICS:
        return value * 100
    return value


def _build_grouped_bar_chart(df: pd.DataFrame, metric_key_map: dict[str, str], title: str) -> go.Figure:
    working = df.copy()
    working["Adresse A"] = [_display_value_for_chart(metric_key_map.get(metric, metric), value) or 0 for metric, value in zip(working["metric"], working["Adresse A"])]
    working["Adresse B"] = [_display_value_for_chart(metric_key_map.get(metric, metric), value) or 0 for metric, value in zip(working["metric"], working["Adresse B"])]

    working["max_value"] = working[["Adresse A", "Adresse B"]].max(axis=1)
    working = working.sort_values("max_value", ascending=True)
    metric_order = working["metric"].tolist()
    text_a_map = {metric: format_metric(metric_key_map.get(metric, metric), value) for metric, value in zip(df["metric"], df["Adresse A"])}
    text_b_map = {metric: format_metric(metric_key_map.get(metric, metric), value) for metric, value in zip(df["metric"], df["Adresse B"])}

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=working["Adresse A"], y=metric_order, name="Adresse A", orientation="h",
        marker=dict(color=ADDRESS_COLORS["Adresse A"]),
        text=[text_a_map[m] for m in metric_order], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Adresse A : %{text}<extra></extra>",
        offsetgroup="a",
        legendrank=1,
    ))
    fig.add_trace(go.Bar(
        x=working["Adresse B"], y=metric_order, name="Adresse B", orientation="h",
        marker=dict(color=ADDRESS_COLORS["Adresse B"]),
        text=[text_b_map[m] for m in metric_order], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Adresse B : %{text}<extra></extra>",
        offsetgroup="b",
        legendrank=2,
    ))
    fig.update_layout(barmode="group", title=title, xaxis_title="Valeur", yaxis_title=None)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(automargin=True)
    return _plotly_base_layout(fig, height=max(300, 70 * len(metric_order) + 80))


def _build_non_percent_100_stacked_chart(df: pd.DataFrame, metric_key_map: dict[str, str], title: str) -> go.Figure:
    working = df.copy()
    working["a_value"] = [_safe_number(value) or 0 for value in working["Adresse A"]]
    working["b_value"] = [_safe_number(value) or 0 for value in working["Adresse B"]]
    working["total"] = working["a_value"] + working["b_value"]
    working = working[working["total"] > 0].copy()
    if working.empty:
        return _plotly_base_layout(go.Figure(), height=320)

    working["Adresse A %"] = (working["a_value"] / working["total"] * 100).round(2)
    working["Adresse B %"] = (working["b_value"] / working["total"] * 100).round(2)
    working = working.sort_values("Adresse A %", ascending=False)

    metric_order = working["metric"].tolist()
    a_text = [f"{value:.0f}%" if value >= 11 else "" for value in working["Adresse A %"]]
    b_text = [f"{value:.0f}%" if value >= 11 else "" for value in working["Adresse B %"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metric_order,
        y=working["Adresse B %"],
        name="Adresse B",
        marker=dict(color=ADDRESS_COLORS["Adresse B"]),
        text=b_text,
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>Adresse B : %{customdata[0]}<br>Part du total A+B : %{y:.1f}%<extra></extra>",
        customdata=[[format_metric(metric_key_map.get(metric, metric), value)] for metric, value in zip(working["metric"], working["b_value"])],
        legendrank=2,
    ))
    fig.add_trace(go.Bar(
        x=metric_order,
        y=working["Adresse A %"],
        name="Adresse A",
        marker=dict(color=ADDRESS_COLORS["Adresse A"]),
        text=a_text,
        textposition="inside",
        hovertemplate="<b>%{x}</b><br>Adresse A : %{customdata[0]}<br>Part du total A+B : %{y:.1f}%<extra></extra>",
        customdata=[[format_metric(metric_key_map.get(metric, metric), value)] for metric, value in zip(working["metric"], working["a_value"])],
        legendrank=1,
    ))
    fig.update_layout(
        title=title,
        barmode="stack",
        xaxis_title=None,
        yaxis_title=None,
        bargap=0.22,
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_xaxes(tickangle=-18, showgrid=False)
    return _plotly_base_layout(fig, height=max(320, 105 * math.ceil(len(metric_order) / 4)))



def _build_single_address_stacked_chart(address_name: str, distribution: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    for label, value in distribution.items():
        fig.add_trace(
            go.Bar(
                x=[address_name],
                y=[(value or 0) * 100],
                name=label,
                marker=dict(color=AGE_SEGMENT_COLORS.get(label)),
                text=[f"{(value or 0):.1%}"],
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"{label}<br>%{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Répartition par âge — {address_name}",
        barmode="stack",
        xaxis_title=None,
        yaxis_title=None,
    )
    fig.update_yaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_xaxes(showgrid=False)
    return _plotly_base_layout(fig, height=360)



def _build_gender_donut_chart(address_name: str, female_share: float | None) -> go.Figure | None:
    if female_share is None:
        return None
    male_share = max(0.0, 1.0 - float(female_share))
    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Femmes", "Hommes"],
                values=[float(female_share), male_share],
                hole=0.62,
                sort=False,
                marker=dict(colors=GENDER_COLORS),
                textinfo="label+percent",
                hovertemplate="%{label}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(title=f"Répartition femmes / hommes — {address_name}")
    return _plotly_base_layout(fig, height=340)



def _build_stacked_columns_chart(distributions: dict[str, dict[str, float]], colors: dict[str, str], title: str) -> go.Figure:
    fig = go.Figure()
    addresses = list(distributions.keys())
    if not addresses:
        return _plotly_base_layout(go.Figure(), height=320)
    segments = list(next(iter(distributions.values())).keys())
    for segment in segments:
        fig.add_trace(
            go.Bar(
                x=addresses,
                y=[(distributions[address].get(segment) or 0) * 100 for address in addresses],
                name=segment,
                marker=dict(color=colors.get(segment)),
                text=[f"{(distributions[address].get(segment) or 0):.1%}" for address in addresses],
                textposition="inside",
                hovertemplate=f"{segment}<br>%{{x}} : %{{y:.1f}}%<extra></extra>",
            )
        )
    fig.update_layout(title=title, barmode="stack", xaxis_title=None, yaxis_title=None)
    fig.update_yaxes(range=[0, 100], ticksuffix="%", showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_xaxes(showgrid=False)
    return _plotly_base_layout(fig, height=380)



def _build_percent_grouped_bar_chart(df: pd.DataFrame, metric_key_map: dict[str, str], title: str) -> go.Figure:
    return _build_grouped_bar_chart(df, metric_key_map, title)



def _value_from_metric(df: pd.DataFrame, metric_label: str, address_name: str):
    subset = df[df["metric"] == metric_label]
    if subset.empty:
        return None
    return subset.iloc[0][address_name]



def _has_any_values(values: list[object]) -> bool:
    return any(_safe_number(value) is not None for value in values)



def render_category_charts(category_name: str, raw_group_df: pd.DataFrame, metric_key_map: dict[str, str]):
    working_df = raw_group_df.copy()
    if working_df.empty:
        return

    if category_name == "Profil sociodémographique":
        age_labels = [
            ("Part 0–17 ans", "0–17 ans"),
            ("Part 18–39 ans", "18–39 ans"),
            ("Part 40–64 ans", "40–64 ans"),
            ("Part 65+", "65+"),
        ]
        age_cols = st.columns(2)
        for idx, address_name in enumerate(["Adresse A", "Adresse B"]):
            age_distribution = {
                legend_label: _safe_number(_value_from_metric(working_df, metric_label, address_name)) or 0
                for metric_label, legend_label in age_labels
            }
            if sum(age_distribution.values()) > 0:
                with age_cols[idx]:
                    st.plotly_chart(
                        _build_single_address_stacked_chart(address_name, age_distribution),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

        gender_cols = st.columns(2)
        for idx, address_name in enumerate(["Adresse A", "Adresse B"]):
            female_share = _safe_number(_value_from_metric(working_df, "Part femmes (RP 2021)", address_name))
            fig = _build_gender_donut_chart(address_name, female_share)
            if fig is not None:
                with gender_cols[idx]:
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        remaining_metrics = working_df[working_df["metric"].isin(["Part mobilité résidentielle externe (RP 2021)"])]
        if not remaining_metrics.empty and _has_any_values(remaining_metrics["Adresse A"].tolist() + remaining_metrics["Adresse B"].tolist()):
            st.plotly_chart(
                _build_percent_grouped_bar_chart(remaining_metrics, metric_key_map, "Mobilité résidentielle"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        return

    if category_name == "Pouvoir d'achat & habitat":
        level_metrics = working_df[working_df["metric"].isin([
            "Niveau de vie moyen winsorisé / pers.",
            "Surface moyenne logement (m²/ménage)",
        ])].copy()
        if not level_metrics.empty and _has_any_values(level_metrics["Adresse A"].tolist() + level_metrics["Adresse B"].tolist()):
            st.plotly_chart(
                _build_non_percent_100_stacked_chart(level_metrics, metric_key_map, "Poids relatif sur les indicateurs de niveau de vie & surface"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

        rate_metrics = working_df[working_df["metric"].isin([
            "Taux de pauvreté ménages",
            "Taux de propriétaires",
            "Part logements sociaux",
            "Part population imputée",
        ])].copy()
        if not rate_metrics.empty and _has_any_values(rate_metrics["Adresse A"].tolist() + rate_metrics["Adresse B"].tolist()):
            st.plotly_chart(
                _build_percent_grouped_bar_chart(rate_metrics, metric_key_map, "Ratios d'habitat & fragilité sociale"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        return

    if category_name == "CSP des actifs 15–64 ans":
        csp_order = [
            "Agriculteurs",
            "Artisans / commerçants / chefs d'entreprise",
            "Cadres & prof. intellectuelles sup.",
            "Professions intermédiaires",
            "Employés",
            "Ouvriers",
        ]
        distributions = {
            address_name: {
                label: _safe_number(_value_from_metric(working_df, label, address_name)) or 0
                for label in csp_order
            }
            for address_name in ["Adresse A", "Adresse B"]
        }
        if any(sum(dist.values()) > 0 for dist in distributions.values()):
            st.plotly_chart(
                _build_stacked_columns_chart(distributions, CSP_SEGMENT_COLORS, "Répartition des CSP des actifs 15–64 ans"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.caption("CSP non disponibles dans le fichier RP 2021 chargé.")
        return

    working_df["chart_group"] = working_df["metric"].map(
        lambda label: _chart_group_for_metric(metric_key_map.get(label, label))
    )

    non_percent = working_df[working_df["chart_group"].isin(["count", "currency"])].copy()
    percent_df = working_df[working_df["chart_group"] == "percent"].copy()

    if not non_percent.empty:
        fig = _build_non_percent_100_stacked_chart(non_percent, metric_key_map, "Répartition relative A / B par indicateur")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if not percent_df.empty:
        fig = _build_percent_grouped_bar_chart(percent_df, metric_key_map, "Comparaison des parts / taux")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_map_legend(selected_categories: list[str]):
    items = [
        f'<span style="display:inline-flex;align-items:center;margin-right:12px;white-space:nowrap;"><span style="width:10px;height:10px;border-radius:999px;background:{ADDRESS_COLORS["Adresse A"]};display:inline-block;margin-right:6px;"></span>Zone A</span>',
        f'<span style="display:inline-flex;align-items:center;margin-right:12px;white-space:nowrap;"><span style="width:10px;height:10px;border-radius:999px;background:{ADDRESS_COLORS["Adresse B"]};display:inline-block;margin-right:6px;"></span>Zone B</span>',
    ]
    for category in selected_categories:
        items.append(
            f'<span style="display:inline-flex;align-items:center;margin-right:12px;white-space:nowrap;"><span style="width:10px;height:10px;border-radius:999px;background:{_rgba_to_hex(MAP_POI_COLORS.get(category, [90, 90, 90, 220]))};display:inline-block;margin-right:6px;border:1px solid rgba(0,0,0,0.15);"></span>{POI_LABELS.get(category, category)}</span>'
        )
    st.markdown('<div style="font-size:0.88rem; line-height:1.4; margin-top:-4px;">' + ' '.join(items) + '</div>', unsafe_allow_html=True)


def compute_zoom_for_geometry(zone_geojson: dict, center_lat: float, fallback_radius_km: float) -> float:
    geometry = shape(zone_geojson["features"][0]["geometry"])
    minx, miny, maxx, maxy = geometry.bounds
    lon_span = max(maxx - minx, 0.001)
    lat_span = max(maxy - miny, 0.001)

    lat_factor = max(math.cos(math.radians(center_lat)), 0.2)
    padding = 1.35
    map_width_px = 430
    map_height_px = 360

    meters_lon = lon_span * 111_320 * lat_factor * padding
    meters_lat = lat_span * 110_574 * padding
    fallback_span_m = max(fallback_radius_km, 0.1) * 2_000 * padding
    required_span_m = max(meters_lon, meters_lat, fallback_span_m)

    zoom_lon = math.log2((156543.03392 * lat_factor * map_width_px) / required_span_m)
    zoom_lat = math.log2((156543.03392 * map_height_px) / required_span_m)
    zoom = min(zoom_lon, zoom_lat)
    return max(5.0, min(16.0, round(zoom, 2)))


def make_map_deck(
    name: str,
    payload: dict,
    fill_color: list[int],
    radius_km: float,
    displayed_poi_categories: list[str],
) -> pdk.Deck:
    point = {
        "name": name,
        "coordinates": [payload["geocode"]["lon"], payload["geocode"]["lat"]],
        "label": payload["geocode"]["matched_label"],
        "category_label": "Adresse",
    }
    feature_collection = payload["zone_geojson"]
    features = []
    for feature in feature_collection["features"]:
        feature = dict(feature)
        feature["properties"] = {
            "name": name,
            "fillColor": fill_color,
            "label": payload["geocode"]["matched_label"],
            "category_label": "Périmètre",
        }
        features.append(feature)

    layers = [
        pdk.Layer(
            "GeoJsonLayer",
            {"type": "FeatureCollection", "features": features},
            get_fill_color="properties.fillColor",
            get_line_color=[30, 30, 30, 180],
            line_width_min_pixels=2,
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            [point],
            get_position="coordinates",
            get_radius=90,
            radius_min_pixels=6,
            radius_max_pixels=8,
            get_fill_color=[20, 20, 20, 220],
            pickable=True,
        ),
    ]

    poi_points = payload.get("poi_points") or {}
    for category in displayed_poi_categories:
        category_points = poi_points.get(category) or []
        if not category_points:
            continue
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                category_points,
                get_position="coordinates",
                get_radius=14,
                radius_min_pixels=2,
                radius_max_pixels=4,
                get_fill_color=MAP_POI_COLORS.get(category, [90, 90, 90, 220]),
                get_line_color=[255, 255, 255, 180],
                line_width_min_pixels=1,
                pickable=True,
            )
        )

    zoom = compute_zoom_for_geometry(feature_collection, payload["geocode"]["lat"], radius_km)
    return pdk.Deck(
        map_style=CARTO_LIGHT_STYLE,
        initial_view_state=pdk.ViewState(
            latitude=payload["geocode"]["lat"],
            longitude=payload["geocode"]["lon"],
            zoom=zoom,
            pitch=0,
        ),
        layers=layers,
        tooltip={"text": "{name}\n{label}\n{category_label}"},
    )


@st.cache_data(show_spinner=False)
def build_zone(
    lon: float,
    lat: float,
    catchment_mode: str,
    target_crs_wkt: str,
    radius_km: float,
    time_minutes: int,
    profile: str,
):
    target_crs = target_crs_wkt
    point_wgs84 = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326")
    point_projected = point_wgs84.to_crs(target_crs).iloc[0]

    if catchment_mode == "Rayon circulaire":
        geometry = point_projected.buffer(radius_km * 1000)
        return geometry, None

    isochrone_result = fetch_isochrone(
        lon,
        lat,
        minutes=time_minutes,
        profile=profile,
        session=get_session(),
    )
    geometry = geojson_to_projected_geometry(isochrone_result.payload, target_crs)
    return geometry, isochrone_result.meta


@st.cache_data(show_spinner=False)
def compute_for_address(
    address: str,
    catchment_mode: str,
    radius_km: float,
    time_minutes: int,
    profile: str,
    include_census: bool,
):
    session = get_session()
    filosofi_dataset = get_filosofi_dataset()
    rp_dataset = get_rp_dataset() if include_census else None

    geocoded = geocode_address(address, session=session)
    point_wgs84 = gpd.GeoSeries([Point(geocoded.lon, geocoded.lat)], crs="EPSG:4326")
    point_projected = point_wgs84.to_crs(filosofi_dataset.crs).iloc[0]
    filosofi_bounds = filosofi_dataset.bounds
    point_in_filosofi_bounds = (
        filosofi_bounds[0] <= point_projected.x <= filosofi_bounds[2]
        and filosofi_bounds[1] <= point_projected.y <= filosofi_bounds[3]
    )

    zone_geometry, isochrone_meta = build_zone(
        lon=geocoded.lon,
        lat=geocoded.lat,
        catchment_mode=catchment_mode,
        target_crs_wkt=filosofi_dataset.crs.to_wkt(),
        radius_km=radius_km,
        time_minutes=time_minutes,
        profile=profile,
    )

    filosofi_subset = load_subset(filosofi_dataset, zone_geometry)
    filosofi = aggregate_filosofi(filosofi_subset, zone_geometry)

    rp_payload = None
    if include_census and rp_dataset is not None:
        rp_subset = load_subset(rp_dataset, zone_geometry)
        rp_payload = aggregate_rp2021(rp_subset, zone_geometry)

    zone_geojson = gpd.GeoSeries([zone_geometry], crs=filosofi_dataset.crs).to_crs("EPSG:4326").__geo_interface__

    poi_counts = {}
    poi_points = {}
    poi_meta = {"status": "not_run"}
    try:
        poi_counts, poi_points, poi_meta = fetch_poi_summary(json.dumps(zone_geojson))
    except Exception as exc:  # pragma: no cover - depends on live services
        poi_meta = {
            "status": "error",
            "message": str(exc),
            "source": "OpenStreetMap / Overpass",
        }

    return {
        "geocode": asdict(geocoded),
        "zone_geojson": zone_geojson,
        "isochrone_meta": isochrone_meta,
        "filosofi_raw": filosofi.raw,
        "filosofi_derived": filosofi.derived,
        "filosofi_cells": filosofi.intersected_cells,
        "rp_raw": rp_payload.raw if rp_payload else {},
        "rp_derived": rp_payload.derived if rp_payload else {},
        "rp_cells": rp_payload.intersected_cells if rp_payload else 0,
        "zone_area_km2": filosofi.zone_area_km2,
        "poi_counts": poi_counts,
        "poi_points": poi_points,
        "poi_meta": poi_meta,
        "dataset_meta": {
            "filosofi_path": str(filosofi_dataset.gpkg_path),
            "filosofi_layer": filosofi_dataset.layer,
            "filosofi_source": filosofi_dataset.source,
            "filosofi_crs": filosofi_dataset.crs.to_string(),
            "filosofi_bounds": tuple(round(v, 2) for v in filosofi_dataset.bounds),
            "point_in_filosofi_bounds": point_in_filosofi_bounds,
            "rp_path": str(rp_dataset.gpkg_path) if rp_dataset else None,
            "rp_layer": rp_dataset.layer if rp_dataset else None,
            "rp_source": rp_dataset.source if rp_dataset else None,
            "rp_crs": rp_dataset.crs.to_string() if rp_dataset else None,
            "rp_bounds": tuple(round(v, 2) for v in rp_dataset.bounds) if rp_dataset else None,
        },
    }


def _comparison_sentence(label: str, a_value, b_value, metric_key: str) -> str | None:
    a_num = _safe_number(a_value)
    b_num = _safe_number(b_value)
    if a_num is None or b_num is None:
        return None
    if a_num == b_num:
        return f"{label} : parité entre Adresse A et Adresse B à {format_metric(metric_key, a_num)}."

    winner = "Adresse A" if a_num > b_num else "Adresse B"
    high = max(a_num, b_num)
    low = min(a_num, b_num)
    abs_gap, rel_gap = _format_abs_and_relative_gap(high, low, metric_key)
    winner_value = format_metric(metric_key, a_num if winner == "Adresse A" else b_num)
    loser_value = format_metric(metric_key, b_num if winner == "Adresse A" else a_num)
    sentence = f"{label} : {winner} mène avec {winner_value} contre {loser_value}"
    if rel_gap is not None:
        sentence += f", soit +{abs_gap} (+{rel_gap})."
    else:
        sentence += f", soit +{abs_gap}."
    return sentence


def _score_metric_for_a(a_value, b_value, *, higher_is_better: bool) -> float | None:
    a_num = _safe_number(a_value)
    b_num = _safe_number(b_value)
    if a_num is None or b_num is None:
        return None
    baseline = max(abs(b_num), 1.0)
    signed_gap = (a_num - b_num) / baseline
    return signed_gap if higher_is_better else -signed_gap


def _advantage_text(label: str, a_value, b_value, metric_key: str, *, higher_is_better: bool) -> str | None:
    a_num = _safe_number(a_value)
    b_num = _safe_number(b_value)
    if a_num is None or b_num is None or a_num == b_num:
        return None

    favorable = (a_num > b_num) if higher_is_better else (a_num < b_num)
    abs_gap, rel_gap = _format_abs_and_relative_gap(max(a_num, b_num), min(a_num, b_num), metric_key)
    sign = "+" if favorable else "-"
    text = f"**{label}** : Adresse A {format_metric(metric_key, a_num)} vs Adresse B {format_metric(metric_key, b_num)}"
    if rel_gap is not None:
        text += f" ({sign}{abs_gap} ; {sign}{rel_gap})."
    else:
        text += f" ({sign}{abs_gap})."
    return text


def _summary_chart_specs():
    return [
        ("Population estimée", "filosofi_raw", "ind"),
        ("Ménages estimés", "filosofi_raw", "men"),
        ("Niveau de vie moyen winsorisé / pers.", "filosofi_derived", "avg_winsorized_living_standard_per_person"),
        ("Part 18–39 ans", "filosofi_derived", "share_18_39"),
        ("Restaurants", "poi_counts", "restaurants"),
        ("Total établissements scolaires", "poi_counts", "schools_total"),
    ]


def render_summary_visuals(results: dict[str, dict]):
    if (_safe_number(results["Adresse A"]["filosofi_raw"].get("ind")) or 0) == 0 and (_safe_number(results["Adresse B"]["filosofi_raw"].get("ind")) or 0) == 0:
        st.warning(
            "Aucune population n'a été agrégée sur ce périmètre. Vérifie le bon fichier .gpkg et le type de zone choisi."
        )
        return

    specs = _summary_chart_specs()
    for row_start in range(0, len(specs), 3):
        columns = st.columns(3)
        for column, (title, payload_key, metric_key) in zip(columns, specs[row_start:row_start + 3]):
            a_value = (results["Adresse A"].get(payload_key) or {}).get(metric_key)
            b_value = (results["Adresse B"].get(payload_key) or {}).get(metric_key)
            with column:
                _render_metric_kpi(title, metric_key, a_value, b_value)
def _chart_group_for_metric(metric_key: str) -> str:
    if metric_key in PERCENT_METRICS:
        return "percent"
    if "standard" in metric_key:
        return "currency"
    return "count"


CHART_GROUP_TITLES = {
    "count": "Volumes",
    "currency": "Montants / niveaux",
    "percent": "Parts / taux",
}


AGE_SEGMENT_COLORS = {
    "0–17 ans": "#8ECAE6",
    "18–39 ans": "#219EBC",
    "40–64 ans": "#FFB703",
    "65+": "#FB8500",
}
GENDER_COLORS = ["#F06292", "#5C6BC0"]
CSP_SEGMENT_COLORS = {
    "Agriculteurs": "#66BB6A",
    "Artisans / commerçants / chefs d'entreprise": "#FFA726",
    "Cadres & prof. intellectuelles sup.": "#5C6BC0",
    "Professions intermédiaires": "#26A69A",
    "Employés": "#AB47BC",
    "Ouvriers": "#8D6E63",
}


def build_advantages_and_drawbacks(results: dict[str, dict]) -> tuple[list[str], list[str]]:
    a = results["Adresse A"]
    b = results["Adresse B"]
    candidate_specs = [
        ("Population estimée", a["filosofi_raw"].get("ind"), b["filosofi_raw"].get("ind"), "ind", True),
        ("Ménages estimés", a["filosofi_raw"].get("men"), b["filosofi_raw"].get("men"), "men", True),
        ("Niveau de vie moyen winsorisé / pers.", a["filosofi_derived"].get("avg_winsorized_living_standard_per_person"), b["filosofi_derived"].get("avg_winsorized_living_standard_per_person"), "avg_winsorized_living_standard_per_person", True),
        ("Part 18–39 ans", a["filosofi_derived"].get("share_18_39"), b["filosofi_derived"].get("share_18_39"), "share_18_39", True),
        ("Taux de pauvreté ménages", a["filosofi_derived"].get("poverty_rate_households"), b["filosofi_derived"].get("poverty_rate_households"), "poverty_rate_households", False),
        ("Restaurants", (a.get("poi_counts") or {}).get("restaurants"), (b.get("poi_counts") or {}).get("restaurants"), "restaurants", True),
        ("Commerces", (a.get("poi_counts") or {}).get("shops"), (b.get("poi_counts") or {}).get("shops"), "shops", True),
        ("Salons de thé", (a.get("poi_counts") or {}).get("tea_rooms"), (b.get("poi_counts") or {}).get("tea_rooms"), "tea_rooms", False),
        ("Boulangeries & pâtisseries", (a.get("poi_counts") or {}).get("bakeries_pastries"), (b.get("poi_counts") or {}).get("bakeries_pastries"), "bakeries_pastries", False),
        ("Total établissements scolaires", (a.get("poi_counts") or {}).get("schools_total"), (b.get("poi_counts") or {}).get("schools_total"), "schools_total", True),
    ]

    positives = []
    negatives = []
    for label, a_value, b_value, metric_key, higher_is_better in candidate_specs:
        score = _score_metric_for_a(a_value, b_value, higher_is_better=higher_is_better)
        text_item = _advantage_text(label, a_value, b_value, metric_key, higher_is_better=higher_is_better)
        if score is None or score == 0 or text_item is None:
            continue
        if score > 0:
            positives.append((score, text_item))
        else:
            negatives.append((abs(score), text_item))

    top_positive = [text for _, text in sorted(positives, key=lambda item: item[0], reverse=True)[:3]]
    top_negative = [text for _, text in sorted(negatives, key=lambda item: item[0], reverse=True)[:3]]
    return top_positive, top_negative


def narrative(results: dict[str, dict], catchment_mode: str) -> str:
    return ""


st.title("🍪 Comparateur de zones de chalandise pour business plan")
st.caption(
    "Compare 2 adresses avec un rayon circulaire ou une isochrone, puis agrège les données socio-démographiques INSEE autour de chaque site."
)

with st.sidebar:
    st.header("Paramètres")
    address_a = st.text_input("Adresse A", value=DEFAULT_A)
    address_b = st.text_input("Adresse B", value=DEFAULT_B)
    run = st.button("Comparer", type="primary", use_container_width=True)
    catchment_mode = st.radio("Type de zone", ["Rayon circulaire", "Isochrone piéton", "Isochrone voiture"], index=1)
    radius_km = st.slider("Rayon (km)", min_value=0.5, max_value=50.0, value=1.5, step=0.5)
    time_minutes = st.slider("Temps d'isochrone (minutes)", min_value=1, max_value=45, value=15, step=1)
    selected_poi_labels = st.multiselect(
        "Points à afficher sur les cartes",
        options=[POI_LABELS[key] for key in MAP_POI_CATEGORIES],
        default=[POI_LABELS[key] for key in MAP_POI_CATEGORIES],
    )

selected_poi_categories = [key for key in MAP_POI_CATEGORIES if POI_LABELS[key] in selected_poi_labels]
include_census = INCLUDE_CENSUS

if run:
    try:
        profile = "pedestrian" if catchment_mode == "Isochrone piéton" else "car"
        mode_label = "Rayon circulaire" if catchment_mode == "Rayon circulaire" else catchment_mode

        with st.spinner("Calcul des zones et agrégation des données…"):
            results = {
                "Adresse A": compute_for_address(
                    address=address_a,
                    catchment_mode=mode_label,
                    radius_km=radius_km,
                    time_minutes=time_minutes,
                    profile=profile,
                    include_census=include_census,
                ),
                "Adresse B": compute_for_address(
                    address=address_b,
                    catchment_mode=mode_label,
                    radius_km=radius_km,
                    time_minutes=time_minutes,
                    profile=profile,
                    include_census=include_census,
                ),
            }

        st.subheader("Synthèse visuelle")
        render_summary_visuals(results)

        if all((results[name]["filosofi_cells"] == 0 for name in results)):
            st.warning(
                "Les deux zones retournent 0 carreau Filosofi intersecté. Cela pointe généralement vers un mauvais .gpkg, "
                "une extraction partielle, ou un problème de projection."
            )

        st.subheader("Top 3 des avantages / inconvénients de l'adresse A")
        pros, cons = build_advantages_and_drawbacks(results)
        col_pros, col_cons = st.columns(2)
        with col_pros:
            st.markdown("**Avantages de l'adresse A vs B**")
            if pros:
                st.markdown("\n".join([f"{idx}. {item}" for idx, item in enumerate(pros, start=1)]))
            else:
                st.write("Pas d'avantage saillant détecté sur les indicateurs calculés.")
        with col_cons:
            st.markdown("**Inconvénients de l'adresse A vs B**")
            if cons:
                st.markdown("\n".join([f"{idx}. {item}" for idx, item in enumerate(cons, start=1)]))
            else:
                st.write("Pas d'inconvénient saillant détecté sur les indicateurs calculés.")

        st.subheader("Cartes")
        map_col_a, map_col_b = st.columns(2)
        for container, name in zip([map_col_a, map_col_b], results.keys()):
            payload = results[name]
            with container:
                st.markdown(f"**{name}**")
                st.caption(payload["geocode"]["matched_label"])
                st.pydeck_chart(
                    make_map_deck(
                        name=name,
                        payload=payload,
                        fill_color=ADDRESS_FILL_COLORS[name],
                        radius_km=radius_km,
                        displayed_poi_categories=selected_poi_categories,
                    ),
                    use_container_width=True,
                )
        render_map_legend(selected_poi_categories)

        grouped_metric_specs = {
            "Bassin de clientèle": [
                ("Population estimée", "filosofi_raw", "ind"),
                ("Ménages estimés", "filosofi_raw", "men"),
                ("Densité pop. (hab/km²)", "filosofi_derived", "population_density"),
                ("Taille moyenne ménage", "filosofi_derived", "avg_household_size"),
            ],
            "Pouvoir d'achat & habitat": [
                ("Niveau de vie moyen winsorisé / pers.", "filosofi_derived", "avg_winsorized_living_standard_per_person"),
                ("Taux de pauvreté ménages", "filosofi_derived", "poverty_rate_households"),
                ("Taux de propriétaires", "filosofi_derived", "owner_rate"),
                ("Part logements sociaux", "filosofi_derived", "social_housing_rate"),
                ("Surface moyenne logement (m²/ménage)", "filosofi_derived", "avg_dwelling_area_m2_per_household"),
                ("Part population imputée", "filosofi_derived", "imputation_population_share"),
            ],
            "Profil sociodémographique": [
                ("Part 0–17 ans", "filosofi_derived", "share_0_17"),
                ("Part 18–39 ans", "filosofi_derived", "share_18_39"),
                ("Part 40–64 ans", "filosofi_derived", "share_40_64"),
                ("Part 65+", "filosofi_derived", "share_65_plus"),
            ],
            "Environnement commercial": [
                (POI_LABELS["tea_rooms"], "poi_counts", "tea_rooms"),
                (POI_LABELS["bakeries_pastries"], "poi_counts", "bakeries_pastries"),
                (POI_LABELS["restaurants"], "poi_counts", "restaurants"),
                (POI_LABELS["shops"], "poi_counts", "shops"),
            ],
            "Enseignement": [
                (POI_LABELS["kindergartens"], "poi_counts", "kindergartens"),
                (POI_LABELS["primary_schools"], "poi_counts", "primary_schools"),
                (POI_LABELS["middle_schools"], "poi_counts", "middle_schools"),
                (POI_LABELS["high_schools"], "poi_counts", "high_schools"),
                (POI_LABELS["higher_education"], "poi_counts", "higher_education"),
                (POI_LABELS["schools_total"], "poi_counts", "schools_total"),
            ],
        }

        grouped_metric_specs["Profil sociodémographique"].extend(
            [
                ("Part <15 ans (RP 2021)", "rp_derived", "share_under_15"),
                ("Part 15–64 ans (RP 2021)", "rp_derived", "share_15_64"),
                ("Part femmes (RP 2021)", "rp_derived", "female_share"),
                ("Part mobilité résidentielle externe (RP 2021)", "rp_derived", "moved_from_elsewhere_share"),
            ]
        )
        grouped_metric_specs["CSP des actifs 15–64 ans"] = [
            ("Agriculteurs", "rp_derived", "csp_cs1_share"),
            ("Artisans / commerçants / chefs d'entreprise", "rp_derived", "csp_cs2_share"),
            ("Cadres & prof. intellectuelles sup.", "rp_derived", "csp_cs3_share"),
            ("Professions intermédiaires", "rp_derived", "csp_cs4_share"),
            ("Employés", "rp_derived", "csp_cs5_share"),
            ("Ouvriers", "rp_derived", "csp_cs6_share"),
        ]

        flat_metric_specs = []
        metric_key_map = {}
        for category, specs in grouped_metric_specs.items():
            for label, block_name, metric_name in specs:
                flat_metric_specs.append((category, label, block_name, metric_name))
                metric_key_map[label] = metric_name

        comparison_df = make_manual_comparison_rows(results, flat_metric_specs)
        styled_df = format_comparison_frame(comparison_df, metric_key_map)

        st.subheader("Tableau de comparaison")
        for category in grouped_metric_specs:
            st.markdown(f"**{category}**")
            raw_group_df = comparison_df[comparison_df["category"] == category].copy()
            group_df = styled_df[styled_df["category"] == category].drop(columns=["category"])
            st.dataframe(group_df, use_container_width=True, hide_index=True)
            render_category_charts(category, raw_group_df, metric_key_map)

        poi_status = [payload.get("poi_meta", {}).get("status") for payload in results.values()]
        if any(status == "error" for status in poi_status):
            st.warning(
                "Les comptages commerces / écoles n'ont pas pu être récupérés pour au moins une adresse. La partie socio-démographique INSEE reste disponible."
            )
        else:
            st.caption(
                "Population, ménages, âge, logement et proxy de niveau de vie : INSEE Filosofi 2021 (carreaux 200 m). "
                "Sexe, lieu de naissance et mobilité résidentielle : INSEE Recensement de la population 2021 (carreaux 1 km). "
                "Salons de thé, boulangeries/pâtisseries, restaurants, commerces et établissements scolaires : OpenStreetMap via Overpass, filtrés dans le périmètre sélectionné."
            )

        export_df = comparison_df.copy()
        export_df.insert(
            3,
            "Delta",
            [
                _format_delta_ratio(row["Adresse A"], row["Adresse B"], metric_key_map.get(row["metric"], row["metric"]))
                for _, row in export_df.iterrows()
            ],
        )
        csv_bytes = make_csv_bytes(export_df)
        st.download_button(
            "Télécharger la comparaison (CSV)",
            data=csv_bytes,
            file_name="comparaison_zones_chalandise.csv",
            mime="text/csv",
        )

    except ApiError as exc:
        st.error(str(exc))
    except Exception as exc:  # pragma: no cover - keeps UI informative in real runs
        st.exception(exc)
else:
    st.markdown(
        """
### Ce que fait l'app
- géocode 2 adresses avec l'API Géoplateforme ;
- construit soit un **rayon circulaire**, soit une **isochrone** ;
- agrège les **carreaux 200 m Filosofi 2021** pour la population, les ménages, l'âge, le logement et un proxy de niveau de vie ;
- intègre aussi le **recensement 2021 en carreaux de 1 km** pour compléter avec sexe, lieu de naissance et mobilité résidentielle.

### Utilisation recommandée pour ton business plan
- compare d'abord un **rayon 1 km / 1,5 km** entre Toulouse et Montpellier ;
- puis compare une **isochrone piéton 10–15 min** pour approcher une clientèle de proximité ;
- exporte le CSV et recopie la synthèse dans la section *zone de chalandise / bassin de clientèle* du dossier bancaire.

"""
    )
