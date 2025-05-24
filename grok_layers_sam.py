import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime
from samgeo import SamGeo, tms_to_geotiff
from samgeo.text_sam import LangSAM
import leafmap
import torch
from rasterio.features import shapes

# === CONFIGURACIÓN GENERAL ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]
zoom = 19
output_dir = "output"
checkpoint_dir = "checkpoints"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

MIN_AREA = 10  # Minimum polygon area to be considered valid

def setup_directories():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

def download_tms_image(bbox, zoom):
    image_path = os.path.join(output_dir, f"esri_image_{timestamp}.tif")
    try:
        tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)
        if os.path.getsize(image_path) < 10_000:
            raise ValueError("Archivo TMS descargado parece estar corrupto o vacío.")
        print(f"✅ Imagen TMS descargada: {image_path}")
        return image_path
    except Exception as e:
        print(f"❌ Error al descargar la imagen TMS: {e}")
        raise

def process_text_segmentation(text_prompts, image_path, profile):
    lang_checkpoint = os.path.join(checkpoint_dir, "mobile_sam.pt")
    if not os.path.exists(lang_checkpoint):
        print("⚠️ Checkpoint de LangSAM no encontrado. Por favor descargalo manualmente o implementa la función de descarga automática.")
    lang_sam = LangSAM()
    prompt_outputs = {}

    print("🧠 Generando máscaras por texto:", text_prompts)
    for prompt in text_prompts:
        print(f"   📌 Procesando prompt: '{prompt}'")
        try:
            lang_sam.predict(image=image_path, text_prompt=prompt, box_threshold=0.24, text_threshold=0.24)
            temp_mask_path = os.path.join(output_dir, f"mask_{prompt}_{timestamp}.tif")
            lang_sam.show_anns(cmap="Greens", add_boxes=True, alpha=1, blend=False, output=temp_mask_path)

            with rasterio.open(temp_mask_path) as src:
                mask = src.read(1)
                transform = src.transform
                crs = src.crs

            geoms = []
            props = []
            for geom, value in shapes(mask, transform=transform):
                polygon = shape(geom)
                if value == 0 or polygon.area < MIN_AREA:
                    continue
                geoms.append(polygon)
                props.append({"type": prompt, "label": f"{prompt.capitalize()} Feature"})

            vector_out = os.path.join(output_dir, f"segment_{prompt}_{timestamp}.gpkg")
            gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
            gdf.to_file(vector_out, driver="GPKG")
            gdf.to_file(vector_out.replace('.gpkg', '.shp'), driver="ESRI Shapefile")
            print(f"✅ Segmentación para '{prompt}' guardada en: {vector_out}")

            prompt_outputs[prompt] = vector_out
            os.remove(temp_mask_path)
        except Exception as e:
            print(f"❌ Error procesando prompt '{prompt}': {e}")
            raise

    return prompt_outputs

def run_sam_segmentation(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}. Please download it.")

    sam = SamGeo(model_type="vit_h", checkpoint=checkpoint_path, device=device)
    filtered_mask_path = os.path.join(output_dir, f"sam_mask_all_{timestamp}.tif")
    try:
        sam.generate(source=image_path, output=filtered_mask_path, batch=True, foreground=True)
        print(f"✅ Segmentación SAM completada: {filtered_mask_path}")
        return filtered_mask_path
    except Exception as e:
        print(f"❌ Error en segmentación SAM: {e}")
        raise

def subtract_langsam_from_sam(sam_gdf, prompt_outputs):
    for prompt, vector_out in prompt_outputs.items():
        gdf = gpd.read_file(vector_out).to_crs(sam_gdf.crs)
        combined = gdf.unary_union
        sam_gdf["geometry"] = sam_gdf.geometry.difference(combined)
    sam_gdf = sam_gdf[sam_gdf.geometry.is_valid & ~sam_gdf.geometry.is_empty]
    return sam_gdf

def save_sam_segments(filtered_mask_path):
    try:
        with rasterio.open(filtered_mask_path) as mask_src:
            mask_data = mask_src.read(1)
            transform = mask_src.transform
            crs = mask_src.crs

        geoms = []
        props = []
        for geom, value in shapes(mask_data, transform=transform):
            polygon = shape(geom)
            if value == 0 or polygon.area < MIN_AREA:
                continue
            geoms.append(polygon)
            props.append({"type": "SAM_segment", "label": "SAM Segment"})

        vector_out = os.path.join(output_dir, f"sam_segment_{timestamp}.gpkg")
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
        gdf.to_file(vector_out, driver="GPKG")
        gdf.to_file(vector_out.replace('.gpkg', '.shp'), driver="ESRI Shapefile")
        print(f"✅ Segmentación SAM final guardada en: {vector_out}")
        return vector_out
    except Exception as e:
        print(f"❌ Error al guardar segmentos SAM: {e}")
        raise

def create_interactive_map(bbox, zoom, prompt_outputs, sam_vector_out):
    layer_styles = {
        "tree": {"color": "#00CC00", "weight": 2, "fillColor": "#00FF00", "fillOpacity": 0.2},
        "water": {"color": "#3399FF", "weight": 2, "fillColor": "#66B2FF", "fillOpacity": 0.2},
        "building": {"color": "#8B4513", "weight": 2, "fillColor": "#CD853F", "fillOpacity": 0.3},
        "road": {"color": "#555555", "weight": 2, "fillColor": "#AAAAAA", "fillOpacity": 0.3},
        "SAM_segment": {"color": "#FF6600", "weight": 2, "fillColor": "#FF9933", "fillOpacity": 0.2}
    }

    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    adjusted_zoom = zoom - 1
    html_path = os.path.join(output_dir, f"segmentacion_mapa_{timestamp}.html")
    try:
        m = leafmap.Map(center=center, zoom=adjusted_zoom, scroll_wheel_zoom=True)
        m.add_basemap("Esri.WorldImagery")

        for prompt, vector_out in prompt_outputs.items():
            m.add_vector(
                vector_out,
                layer_name=f"{prompt.capitalize()} Segmentation",
                style=layer_styles[prompt],
                group="Text Segmentations"
            )

        m.add_vector(
            sam_vector_out,
            layer_name="SAM Segments",
            style=layer_styles["SAM_segment"],
            group="SAM Segmentations"
        )

        m.add_layer_control(position="topright")
        legend_dict = {
            "Trees (Green)": "#00FF00",
            "Water (Blue)": "#66B2FF",
            "Buildings (Brown)": "#CD853F",
            "Roads (Gray)": "#AAAAAA",
            "SAM Segments (Orange)": "#FF9933"
        }
        m.add_legend(title="Segmentation Layers", legend_dict=legend_dict, position="bottomright")
        m.to_html(outfile=html_path)

        print(f"📁 Mapa HTML exportado: {html_path}")
        return m
    except Exception as e:
        print(f"❌ Error al crear el mapa interactivo: {e}")
        raise

def calculate_class_coverage(prompt_outputs, sam_vector_out):
    import pandas as pd
    print("📊 Calculando cobertura por clase...")
    total_area = 0.0
    class_areas = {}

    # Sumar áreas de cada clase LangSAM
    for prompt, vector_out in prompt_outputs.items():
        gdf = gpd.read_file(vector_out)
        area = gdf.geometry.area.sum()
        class_areas[prompt] = area
        total_area += area

    # Agregar SAM_segment
    gdf_sam = gpd.read_file(sam_vector_out)
    sam_area = gdf_sam.geometry.area.sum()
    class_areas["SAM_segment"] = sam_area
    total_area += sam_area

    # Mostrar porcentajes
    coverage_data = []
    for cls, area in class_areas.items():
        pct = (area / total_area) * 100 if total_area > 0 else 0
        coverage_data.append({"class": cls, "area": area, "percentage": pct})
        print(f"   🔹 {cls.capitalize()}: {pct:.2f}% cobertura")

    # Save CSV
    df = pd.DataFrame(coverage_data)
    csv_path = os.path.join(output_dir, f"coverage_summary_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Cobertura exportada a CSV: {csv_path}")

    # Floating div to inject in HTML map
    coverage_colors = {
        "tree": "#00FF00",
        "water": "#66B2FF",
        "building": "#CD853F",
        "road": "#AAAAAA",
        "SAM_segment": "#FF9933"
    }
    coverage_html = """<div style='position: fixed; bottom: 20px; left: 20px; background-color: white; border: 1px solid #aaa; padding: 10px; z-index: 1000;'>
    <b>Resumen de Cobertura:</b><br>
    """ + "".join([
        f"<i style='background:{coverage_colors.get(row['class'], '#000')}; width: 12px; height: 12px; display: inline-block; margin-right: 5px;'></i>{row['class'].capitalize()}: {row['percentage']:.2f}%<br>"
        for row in coverage_data
    ]) + "</div>"

    # Add to map HTML
    with open(os.path.join(output_dir, f"segmentacion_mapa_{timestamp}.html"), "a") as f:
        f.write(coverage_html)

def main():
    setup_directories()
    image_path = download_tms_image(bbox, zoom)
    text_prompts = ["tree", "water", "building", "road"]
    with rasterio.open(image_path) as src:
        profile = src.profile
    prompt_outputs = process_text_segmentation(text_prompts, image_path, profile)
    filtered_mask_path = run_sam_segmentation(image_path)
    sam_vector_out = save_sam_segments(filtered_mask_path)
    print("🗺️ Mostrando mapa interactivo...")
    create_interactive_map(bbox, zoom, prompt_outputs, sam_vector_out)
    calculate_class_coverage(prompt_outputs, sam_vector_out)
    print("✅ Proceso completado.")

if __name__ == "__main__":
    main()
