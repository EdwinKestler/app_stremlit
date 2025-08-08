"""Vectorisation and summary helpers."""

from __future__ import annotations

import os
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def raster_to_vector(raster_path: str, out_path: str) -> gpd.GeoDataFrame:
    """Convert a binary mask raster to vector polygons saved to ``out_path``.

    Parameters
    ----------
    raster_path: str
        Path to the mask raster.
    out_path: str
        Destination GeoPackage path.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of polygons derived from the mask.
    """
    with rasterio.open(raster_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    geoms = []
    for geom, value in shapes(mask, transform=transform):
        if value == 0:
            continue
        geoms.append(shape(geom))

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    return gdf


def summarise(
    gdf: gpd.GeoDataFrame, out_csv: str, class_col: str = "segment_class"
) -> pd.DataFrame:
    """Summarise total area and percentage coverage by segment class.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygons and a column specifying the
        segment class for each polygon.
    out_csv : str
        Destination path for the summary CSV.
    class_col : str, optional
        Name of the column in ``gdf`` storing the segment class labels,
        by default "segment_class".

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``segment_class``, ``area_m2`` and
        ``coverage_pct``.
    """

    if class_col not in gdf.columns:
        raise ValueError(f"Missing '{class_col}' column in GeoDataFrame")

    summary = gdf.copy()
    summary["area_m2"] = summary.geometry.area
    grouped = summary.groupby(class_col, as_index=False)["area_m2"].sum()
    total_area = grouped["area_m2"].sum()
    grouped["coverage_pct"] = (grouped["area_m2"] / total_area) * 100
    grouped.to_csv(out_csv, index=False)
    return grouped
