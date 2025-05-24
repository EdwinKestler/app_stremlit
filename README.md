# 🛰️ Proyecto de Segmentación Visual de Imágenes Satelitales

Este programa automatiza el proceso de análisis de imágenes satelitales para identificar y mostrar en un mapa diferentes tipos de elementos visibles en una zona.

## 📌 ¿Qué hace este programa?

1. **Descarga automática de una imagen satelital**  
   Toma una “foto desde el espacio” de un área específica usando un servicio de mapas (Esri Satellite).

2. **Busca objetos en la imagen según palabras clave**  
   El sistema entiende indicaciones como “árboles”, “agua”, “edificios” o “caminos”, y encuentra esas zonas en la imagen automáticamente.

3. **Segmentación general automática (SAM)**  
   Además de las búsquedas por palabra, detecta cualquier otro objeto visual destacable.  
   Evita duplicaciones, excluyendo áreas ya etiquetadas.

4. **Crea un mapa interactivo**  
   Genera una página web donde puedes ver el mapa con cada capa marcada por colores.  
   Puedes activar/desactivar las capas y explorar los datos visualmente.

5. **Calcula porcentajes de cobertura**  
   Mide qué porcentaje del área cubre cada tipo de objeto (ej. “15% edificios”).  
   Esta información se muestra visualmente en el mapa.

6. **Exporta resultados**  
   Guarda:
   - Un archivo `.csv` con los porcentajes y áreas por tipo de objeto.
   - Un archivo `.html` con el mapa interactivo.
   - Archivos geoespaciales `.gpkg` y `.shp` para usar en software de mapas (QGIS, ArcGIS, etc).

## 🗂️ Archivos generados

- `output/esri_image_<fecha>.tif`: imagen satelital base.
- `output/segment_<clase>_<fecha>.gpkg`: zonas segmentadas por tipo (árboles, agua, etc.).
- `output/sam_segment_<fecha>.gpkg`: otras zonas destacadas automáticamente.
- `output/segmentacion_mapa_<fecha>.html`: mapa interactivo.
- `output/coverage_summary_<fecha>.csv`: tabla resumen de áreas y porcentajes.

## 💡 Ejemplo de resumen generado

Árboles: 12.34%
Agua: 8.91%
Edificios: 20.10%
Caminos: 15.67%
Otros objetos (SAM): 43.00%

Este resumen también aparece como una caja flotante en la esquina del mapa web generado.

## ✅ Requisitos para ejecutar
- Python 3.9+
- Instalar librerías: `samgeo`, `leafmap`, `rasterio`, `geopandas`, `torch`, `shapely`

## 👨‍💻 Autor
Desarrollado por Edwin Kestler, para proyectos de agricultura de precisión, supervisión urbana y análisis ambiental.