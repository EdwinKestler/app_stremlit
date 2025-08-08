"""High level geospatial processing pipeline."""

from __future__ import annotations

import os
from typing import Iterable, Dict

import geopandas as gpd
import pandas as pd

from downloader import download_imagery
from segmenter import run_langsam, run_sam2
from .config import PipelineConfig
from vectorizer import raster_to_vector, summarise


def run_pipeline(config: PipelineConfig, text_prompts: Iterable[str]) -> Dict[str, str]:
    """Execute the end-to-end processing pipeline.

    Steps
    -----
    1. Download imagery covering ``config.bbox`` at ``config.zoom`` using ``tms_to_geotiff``.
    2. Run LangSAM to generate semantic masks for each item in
       ``text_prompts``.
    3. Run SAM2 to obtain a general segmentation mask.
    4. Vectorise all masks and export GeoPackage/CSV summaries grouped by
       segment class.

    Parameters
    ----------
    config: PipelineConfig
        Pipeline configuration with bounding box, directories, thresholds
        and model information.
    text_prompts: Iterable[str]
        Prompts for LangSAM.

    Returns
    -------
    dict
        Mapping of product names to file paths.
    """
    os.makedirs(config.out_dir, exist_ok=True)

    image_path = download_imagery(config=config)

    gdfs = []
    prompt_gpkgs = {}
    for prompt in text_prompts:
        mask_path = run_langsam(image_path=image_path, text_prompts=[prompt], config=config)
        gpkg_path = os.path.join(config.out_dir, f"segment_{prompt}.gpkg")
        gdf = raster_to_vector(mask_path, gpkg_path)
        gdf["segment_class"] = prompt
        gdfs.append(gdf)
        prompt_gpkgs[prompt] = gpkg_path

    sam2_mask = run_sam2(image_path=image_path, config=config)
    sam_gpkg_path = os.path.join(config.out_dir, "sam2_segments.gpkg")
    sam_gdf = raster_to_vector(sam2_mask, sam_gpkg_path)
    sam_gdf["segment_class"] = "Other objects (SAM)"
    gdfs.append(sam_gdf)

    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs if gdfs else None)

    csv_path = os.path.join(config.out_dir, "summary.csv")
    summarise(combined, csv_path)

    outputs = {
        "image": image_path,
        "sam2_mask": sam2_mask,
        "csv": csv_path,
        "gpkg_sam2": sam_gpkg_path,
    }
    outputs.update({f"gpkg_{p}": path for p, path in prompt_gpkgs.items()})
    return outputs
