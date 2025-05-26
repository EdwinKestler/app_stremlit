import os
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import shape
from datetime import datetime
from samgeo import SamGeo, tms_to_geotiff
from samgeo.text_sam import LangSAM
import folium
from folium import GeoJson
import torch
import base64
import io
import matplotlib.pyplot as plt

# === Configuration ===
bbox = [-90.015147, 14.916566, -90.010159, 14.919471]  # Example bbox
zoom = 19
output_dir = "output"
checkpoint_dir = "checkpoints"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
MIN_AREA = 10  # Minimum polygon area

# Text prompts and thresholds
prompt_thresholds = {
    "tree": {"box_threshold": 0.3, "text_threshold": 0.24},
    "water": {"box_threshold": 0.24, "text_threshold": 0.24},
    "building": {"box_threshold": 0.28, "text_threshold": 0.25},
    "road": {"box_threshold": 0.24, "text_threshold": 0.2}
}

# GEE Layers (pre-downloaded Copernicus datasets)
gee_layers = [
    {"name": "Sentinel-2 RGB", "path": "data/s2harm_rgb_saa.tif", "bands": [1, 2, 3], "opacity": 0.75},
    {"name": "Copernicus Urban RGB", "path": "data/urban_rgb_copernicus.tif", "bands": [1, 2, 3], "opacity": 0.5},
    {"name": "NDBI Urban Mask", "path": "data/urban_mask_ndbi.tif", "bands": [1], "opacity": 0.35}
]

# === Helper Functions ===
def setup_directories():
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

def download_tms_image(bbox, zoom):
    image_path = os.path.join(output_dir, f"esri_image_{timestamp}.tif")
    tms_to_geotiff(output=image_path, bbox=bbox, zoom=zoom, source="Satellite", overwrite=True)
    print(f"✅ TMS Image Downloaded: {image_path}")
    return image_path

def tif_to_base64_image(tif_path, bands=[1, 2, 3]):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        image = src.read(bands)
        image = np.moveaxis(image, 0, -1)
        image = (image - image.min()) / (image.max() - image.min()) * 255
        image = image.astype(np.uint8)
    fig, ax = plt.subplots()
    ax.axis("off")
    plt.imshow(image)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return encoded, bounds

def process_text_segmentation(text_prompts, image_path):
    lang_sam = LangSAM()
    prompt_outputs = {}
    for prompt in text_prompts:
        lang_sam.predict(
            image=image_path,
            text_prompt=prompt,
            box_threshold=prompt_thresholds[prompt]["box_threshold"],
            text_threshold=prompt_thresholds[prompt]["text_threshold"]
        )
        mask_path = os.path.join(output_dir, f"mask_{prompt}_{timestamp}.tif")
        lang_sam.show_anns(cmap="Greens", add_boxes=True, alpha=1, blend=False, output=mask_path)
        
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs
        
        geoms, props = [], []
        for geom, value in shapes(mask, transform=transform):
            polygon = shape(geom)
            if value == 0 or polygon.area < MIN_AREA:
                continue
            geoms.append(polygon)
            props.append({"type": prompt, "label": f"{prompt.capitalize()} Feature"})
        
        vector_out = os.path.join(output_dir, f"segment_{prompt}_{timestamp}.gpkg")
        gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
        gdf.to_file(vector_out, driver="GPKG")
        prompt_outputs[prompt] = vector_out
        os.remove(mask_path)
    return prompt_outputs

def run_sam_segmentation(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = os.path.join(checkpoint_dir, "sam_vit_h_4b8939.pth")
    sam = SamGeo(model_type="vit_h", checkpoint=checkpoint_path, device=device)
    mask_path = os.path.join(output_dir, f"sam_mask_{timestamp}.tif")
    sam.generate(source=image_path, output=mask_path, batch=True, foreground=True)
    
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        transform = src.transform
        crs = src.crs
    
    geoms, props = [], []
    for geom, value in shapes(mask_data, transform=transform):
        polygon = shape(geom)
        if value == 0 or polygon.area < MIN_AREA:
            continue
        geoms.append(polygon)
        props.append({"type": "SAM_segment", "label": "SAM Segment"})
    
    vector_out = os.path.join(output_dir, f"sam_segment_{timestamp}.gpkg")
    gdf = gpd.GeoDataFrame(props, geometry=geoms, crs=crs)
    gdf.to_file(vector_out, driver="GPKG")
    return vector_out

def calculate_segmentation_urbanization(vector_out, total_area):
    gdf = gpd.read_file(vector_out)
    urban_area = gdf[gdf['type'].isin(['building', 'road'])]['geometry'].area.sum()
    return round((urban_area / total_area) * 100, 2)

def calcular_porcentaje_urbanizado(tif_path, urban_threshold=0.2):
    with rasterio.open(tif_path) as src:
        mask = src.read(1)
        mask = mask[np.isfinite(mask)]
        mask = mask[mask >= 0]
        if mask.size == 0:
            return 0.0
        max_val = mask.max()
        threshold = urban_threshold if max_val <= 1.0 else urban_threshold * max_val
        pixeles_urbanos = np.count_nonzero(mask > threshold)
        total_pixeles = mask.size
    return round((pixeles_urbanos / total_pixeles) * 100, 2)

def create_interactive_map(bbox, zoom, prompt_outputs, sam_vector_out, gee_layers):
    layer_styles = {
        "tree": {"color": "#00CC00", "fillColor": "#00FF00", "fillOpacity": 0.2},
        "water": {"color": "#3399FF", "fillColor": "#66B2FF", "fillOpacity": 0.2},
        "building": {"color": "#800080", "fillColor": "#CC00CC", "fillOpacity": 0.2},
        "road": {"color": "#808080", "fillColor": "#A9A9A9", "fillOpacity": 0.2},
        "SAM_segment": {"color": "#FF6600", "fillColor": "#FF9933", "fillOpacity": 0.2}
    }
    
    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]
    m = folium.Map(location=center, zoom_start=zoom-1, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri Satellite"
    ).add_to(m)
    
    # Add GEE layers
    for layer in gee_layers:
        encoded, bounds = tif_to_base64_image(layer["path"], layer["bands"])
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{encoded}",
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=layer["opacity"],
            name=layer["name"]
        ).add_to(m)
    
    # Add segmentation layers
    for prompt, vector_out in prompt_outputs.items():
        gdf = gpd.read_file(vector_out).to_crs(epsg=4326)
        GeoJson(
            gdf.__geo_interface__,
            name=f"{prompt.capitalize()} Segmentation",
            style_function=lambda feature, p=prompt: layer_styles[p],
            popup=folium.GeoJsonPopup(fields=["label"])
        ).add_to(m)
    
    gdf = gpd.read_file(sam_vector_out).to_crs(epsg=4326)
    GeoJson(
        gdf.__geo_interface__,
        name="SAM Segments",
        style_function=lambda feature: layer_styles["SAM_segment"],
        popup=folium.GeoJsonPopup(fields=["label"])
    ).add_to(m)
    
    # Urbanization metrics
    total_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # Simplified
    seg_urban = calculate_segmentation_urbanization(prompt_outputs["building"], total_area)
    ndbi_urban = calcular_porcentaje_urbanizado(gee_layers[2]["path"])
    
    urban_html = f"""
    <div style="position: fixed; bottom: 40px; left: 40px; width: 220px; background-color: white; border: 2px solid #00A878; border-radius: 8px; padding: 12px; font-family: 'Inter', sans-serif; font-size: 14px; box-shadow: 2px 2px 8px rgba(0,0,0,0.2); z-index: 9999;">
      <b>Urbanization Index</b><br>
      Segmentation: <span style="color:#00A878; font-weight:bold;">{seg_urban}%</span><br>
      NDBI: <span style="color:#00A878; font-weight:bold;">{ndbi_urban}%</span><br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(urban_html))
    
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(os.path.join(output_dir, f"integrated_map_{timestamp}.html"))
    print(f"✅ Map Saved: integrated_map_{timestamp}.html")

# === Main Execution ===
def main():
    setup_directories()
    image_path = download_tms_image(bbox, zoom)  # Replace with Sentinel-2 if desired
    text_prompts = ["tree", "water", "building", "road"]
    prompt_outputs = process_text_segmentation(text_prompts, image_path)
    sam_vector_out = run_sam_segmentation(image_path)
    create_interactive_map(bbox, zoom, prompt_outputs, sam_vector_out, gee_layers)

if __name__ == "__main__":
    main()