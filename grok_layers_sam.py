import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime
from samgeo import SamGeo, tms_to_geotiff
from samgeo.text_sam import LangSAM
import folium
from folium import FeatureGroup, LayerControl, GeoJson
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
    """Create output and checkpoint directories if they don't exist."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

def download_tms_image(bbox, zoom):
    """Download TMS image and save it to the output directory."""
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
    """Generate individual masks for each text prompt using LangSAM and save as GeoPackage and Shapefile layers."""
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
    """Run automatic SAM segmentation on the input image."""
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

def save_sam_segments(filtered_mask_path):
    """Save SAM segments as GeoPackage and Shapefile layers."""
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
    """Create an interactive map using Folium with clear, toggleable layers, labels, and a legend."""
    # Define distinct styles with better contrast and lower opacity
    layer_styles = {
        "tree": {"color": "#00CC00", "weight": 2, "fillColor": "#00FF00", "fillOpacity": 0.2},  # Bright green for trees
        "water": {"color": "#3399FF", "weight": 2, "fillColor": "#66B2FF", "fillOpacity": 0.2},  # Light blue for water
        "building": {"color": "#800080", "weight": 2, "fillColor": "#CC00CC", "fillOpacity": 0.2},  # Purple for buildings
        "road": {"color": "#808080", "weight": 2, "fillColor": "#A9A9A9", "fillOpacity": 0.2},  # Gray for roads
        "SAM_segment": {"color": "#FF6600", "weight": 2, "fillColor": "#FF9933", "fillOpacity": 0.2},  # Bright orange for SAM
    }

    # Calculate center and adjust zoom to fit the bounding box
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    adjusted_zoom = zoom - 1

    html_path = os.path.join(output_dir, f"segmentacion_mapa_{timestamp}.html")
    try:
        # Create a Folium map
        m = folium.Map(location=center, zoom_start=adjusted_zoom, tiles=None)

        # Add Esri World Imagery as the basemap
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Esri World Imagery',
            overlay=False,
            control=True
        ).add_to(m)

        # Add scale bar if folium.plugins is available
        try:
            folium.plugins.ScaleControl(position="bottomleft").add_to(m)
        except AttributeError:
            print("⚠️ 'folium.plugins' no está disponible. Por favor, actualiza folium con 'pip install --upgrade folium'. El mapa se generará sin barra de escala.")

        # Create feature groups for text-based segmentations and SAM segments
        text_group = FeatureGroup(name="Text Segmentations", show=True)
        sam_group = FeatureGroup(name="SAM Segmentations", show=True)

        # Add text-based segmentations (e.g., tree, water, building, road)
        for prompt, vector_out in prompt_outputs.items():
            gdf = gpd.read_file(vector_out)
            gdf = gdf.to_crs(epsg=4326)  # Ensure the GeoDataFrame is in WGS84 for Folium
            geojson_data = gdf.__geo_interface__

            GeoJson(
                geojson_data,
                name=f"{prompt.capitalize()} Segmentation",
                style_function=lambda feature, p=prompt: {
                    "color": layer_styles[p]["color"],
                    "weight": layer_styles[p]["weight"],
                    "fillColor": layer_styles[p]["fillColor"],
                    "fillOpacity": layer_styles[p]["fillOpacity"]
                },
                popup=folium.GeoJsonPopup(fields=["label"]),
            ).add_to(text_group)

        # Add SAM segments
        sam_gdf = gpd.read_file(sam_vector_out)
        sam_gdf = sam_gdf.to_crs(epsg=4326)  # Ensure WGS84 for Folium
        sam_geojson_data = sam_gdf.__geo_interface__

        GeoJson(
            sam_geojson_data,
            name="SAM Segments",
            style_function=lambda feature: {
                "color": layer_styles["SAM_segment"]["color"],
                "weight": layer_styles["SAM_segment"]["weight"],
                "fillColor": layer_styles["SAM_segment"]["fillColor"],
                "fillOpacity": layer_styles["SAM_segment"]["fillOpacity"]
            },
            popup=folium.GeoJsonPopup(fields=["label"]),
        ).add_to(sam_group)

        # Add the feature groups to the map
        text_group.add_to(m)
        sam_group.add_to(m)

        # Add layer control (collapsible)
        LayerControl(position="topright", collapsed=True).add_to(m)

        # Add a custom legend using HTML
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
            <b>Segmentation Layers</b><br>
            <i style="background: #00FF00; width: 18px; height: 18px; display: inline-block; vertical-align: middle;"></i> Trees (Green)<br>
            <i style="background: #66B2FF; width: 18px; height: 18px; display: inline-block; vertical-align: middle;"></i> Water (Blue)<br>
            <i style="background: #CC00CC; width: 18px; height: 18px; display: inline-block; vertical-align: middle;"></i> Buildings (Purple)<br>
            <i style="background: #A9A9A9; width: 18px; height: 18px; display: inline-block; vertical-align: middle;"></i> Roads (Gray)<br>
            <i style="background: #FF9933; width: 18px; height: 18px; display: inline-block; vertical-align: middle;"></i> SAM Segments (Orange)
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save the map to HTML
        m.save(html_path)
        print(f"📁 Mapa HTML exportado: {html_path}")
        print("ℹ️ Usa los controles de capas para activar/desactivar las capas.")
        return m
    except Exception as e:
        print(f"❌ Error al crear el mapa interactivo: {e}")
        raise

def main():
    """
    Process a satellite image to segment features using SAM and LangSAM, visualizing each as clear, toggleable layers.

    This script:
    1. Downloads a satellite image using TMS.
    2. Uses LangSAM to create individual masks for specified text prompts (e.g., 'tree', 'water', 'building', 'road').
    3. Performs automatic segmentation with SAMGeo.
    4. Saves each segmentation (SAM and text-based) as GeoPackage and Shapefile layers.
    5. Visualizes the results on an interactive Folium map with clear, toggleable layers, labels, and a legend.

    Inputs:
    - bbox: Bounding box coordinates [minx, miny, maxx, maxy]
    - zoom: Zoom level for TMS download
    - output_dir: Directory to save outputs
    - checkpoint_dir: Directory containing SAM model checkpoint

    Outputs:
    - TIFF files for images and masks
    - GeoPackage and Shapefile files for each segmentation layer
    - HTML file for interactive map visualization with clear, toggleable layers
    """
    setup_directories()
    
    # Step 1: Download TMS image
    image_path = download_tms_image(bbox, zoom)

    # Step 2: Segment with LangSAM for text prompts (individual layers)
    text_prompts = ["tree", "water", "building", "road"]  # Added building and road
    with rasterio.open(image_path) as src:
        profile = src.profile
    prompt_outputs = process_text_segmentation(text_prompts, image_path, profile)

    # Step 3: Automatic SAM segmentation
    filtered_mask_path = run_sam_segmentation(image_path)

    # Step 4: Save SAM segments as a layer
    sam_vector_out = save_sam_segments(filtered_mask_path)

    # Step 5: Visualize with Folium (separate layers with enhanced clarity)
    print("🗺️ Mostrando mapa interactivo...")
    create_interactive_map(bbox, zoom, prompt_outputs, sam_vector_out)
    print("✅ Proceso completado.")

if __name__ == "__main__":
    main()