from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
from pyogrio import list_layers, read_info
from shapely.geometry import Point

CITY_CENTERS = {
    "toulouse": (1.4442, 43.6045),
    "montpellier": (3.8772, 43.6110),
}


def build_city_buffer(city_key: str, buffer_km: float, target_crs: str):
    lon, lat = CITY_CENTERS[city_key]
    geom = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").to_crs(target_crs).iloc[0]
    return geom.buffer(buffer_km * 1000)


def detect_layer(path: Path) -> str:
    layers = list_layers(path)
    if len(layers) == 0:
        raise RuntimeError(f"No layer found in {path}")
    return str(layers[0][0])


def extract_city(source_gpkg: Path, layer: str, city_key: str, buffer_km: float, output_path: Path):
    info = read_info(source_gpkg, layer=layer)
    target_crs = info["crs"]
    buffer_geom = build_city_buffer(city_key, buffer_km, target_crs)
    subset = gpd.read_file(source_gpkg, layer=layer, bbox=buffer_geom.bounds, engine="pyogrio")
    subset = subset[subset.geometry.notnull()].copy()
    subset = subset[subset.intersects(buffer_geom)].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    subset.to_file(output_path, driver="GPKG")
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"{output_path.name}: {len(subset):,} lignes | {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Build compact INSEE extracts for Toulouse and Montpellier.")
    parser.add_argument("--filosofi-gpkg", type=Path, required=True)
    parser.add_argument("--rp-gpkg", type=Path, required=True)
    parser.add_argument("--filosofi-layer", default=None)
    parser.add_argument("--rp-layer", default=None)
    parser.add_argument("--buffer-km", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    filosofi_layer = args.filosofi_layer or detect_layer(args.filosofi_gpkg)
    rp_layer = args.rp_layer or detect_layer(args.rp_gpkg)

    for city_key in CITY_CENTERS:
        extract_city(
            args.filosofi_gpkg,
            filosofi_layer,
            city_key,
            args.buffer_km,
            args.output_dir / f"filosofi_{city_key}.gpkg",
        )
        extract_city(
            args.rp_gpkg,
            rp_layer,
            city_key,
            args.buffer_km,
            args.output_dir / f"rp_{city_key}.gpkg",
        )


if __name__ == "__main__":
    main()
