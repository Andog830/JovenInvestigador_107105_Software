import rasterio
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from rasterio.features import rasterize
from skimage.segmentation import slic
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.filters.rank import entropy
from skimage.morphology import disk
import math
from datetime import datetime, timezone
from pyproj import Transformer

# ---------------------------------------------------
# ÁNGULOS SOLARES
# ---------------------------------------------------
def compute_solar_angles(lat, lon, dt_utc):
    try:
        from astral import LocationInfo
        from astral.sun import azimuth as _sun_az, elevation as _sun_el
    except ImportError:
        raise ImportError("Instala 'astral': pip install astral")
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    observer = LocationInfo(latitude=lat, longitude=lon).observer
    return float(_sun_az(observer, dt_utc)), float(_sun_el(observer, dt_utc))


def angular_diff(a, b):
    diff = abs((a - b) % 360)
    return min(diff, 360 - diff)


# ---------------------------------------------------
# LECTURA DE ULTRACAM  (sin cambios)
# ---------------------------------------------------
def read_ultracam(path):
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    if img.max() > 1:
        img /= img.max()
    return img, profile, transform, crs


# ---------------------------------------------------
# SEGMENTACIÓN SLIC  (sin cambios)
# ---------------------------------------------------
def segment_slic(img, n_segments=1500, compactness=10):
    stack = np.moveaxis(img, 0, -1)
    return slic(stack, n_segments=n_segments, compactness=compactness, start_label=1)


# ---------------------------------------------------
# INDICE DE SOMBRA  (sin cambios)
# ---------------------------------------------------
def shadow_index(img):
    blue, green, red, nir = img[0], img[1], img[2], img[3]
    si = 1 - ((red + green + blue + nir) / 4)
    si = (si - si.min()) / (si.max() - si.min())
    return si


# ---------------------------------------------------
# TEXTURA  (sin cambios)
# ---------------------------------------------------
def entropy_texture(gray_img, radius=5):
    gray_8bit = (gray_img * 255).astype(np.uint8)
    ent = entropy(gray_8bit, disk(radius))
    ent = ent.astype(np.float32)
    ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-6)
    return ent


# ---------------------------------------------------
# PIPELINE COMPLETO  (sin cambios)
# ---------------------------------------------------
def run_pipeline(path, azimuth=None, elevation=None, solar_time=None, lat=None, lon=None):
    img, profile, transform, crs = read_ultracam(path)
    if solar_time is not None and lat is not None and lon is not None:
        azimuth, elevation = compute_solar_angles(lat, lon, solar_time)
        print(f"Angulos calculados → azimut: {azimuth:.2f}°, elevacion: {elevation:.2f}°")
    elif azimuth is not None and elevation is not None:
        print(f"Angulos manuales → azimut: {azimuth:.2f}°, elevacion: {elevation:.2f}°")
    else:
        print("Advertencia: sin angulos solares.")

    seg = segment_slic(img, n_segments=2000, compactness=8)
    si = shadow_index(img)
    t = threshold_otsu(si)
    shadow_mask = si > t

    red, green, blue, nir = img[2], img[1], img[0], img[3]
    gray = (red + green + blue) / 3
    ent = entropy_texture(gray, radius=3)

    ndvi = (nir - red) / (nir + red + 1e-6)
    vegetation_mask = (ndvi > 0.25) & (ent > 0.4)
    shadow_mask = shadow_mask & (~vegetation_mask)
    shadow_mask = shadow_mask & (nir < 0.18)
    shadow_mask = remove_small_objects(shadow_mask, min_size=100)
    shadow_mask = shadow_mask.astype("uint8")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(si, cmap='gray');          plt.title("Shadow Index")
    plt.subplot(1, 3, 2); plt.imshow(ent, cmap='gray');         plt.title("Entropy")
    plt.subplot(1, 3, 3); plt.imshow(shadow_mask, cmap='gray'); plt.title("Final Shadow Mask")
    plt.show()

    profile.update(count=1, dtype="uint8", transform=transform, crs=crs,
                   compress="lzw", nodata=0)
    out_path = f"{path[:-4]}_shadow_mask_ultracam.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(shadow_mask, 1)
    print(f"Sombra exportada: {out_path}")


# ---------------------------------------------------
# UTILIDAD: tamaño real del pixel en metros
# ---------------------------------------------------
def _pixel_size_meters(transform, crs, center_lat=None):
    """
    Calcula el tamaño del pixel en metros a partir del transform y el CRS.

    - CRS proyectado (metros): usa transform.a directamente.
    - CRS geográfico (grados): convierte 1 grado de latitud/longitud a metros
      usando la latitud central de la imagen.
    """
    from pyproj import CRS as ProjCRS
    proj_crs = ProjCRS.from_user_input(crs)

    if proj_crs.is_projected:
        # Unidades ya están en metros
        px_x = abs(transform.a)
        px_y = abs(transform.e)
        return (px_x + px_y) / 2.0

    # CRS geográfico: necesitamos la latitud para la conversión
    # 1° de latitud  ≈ 111_320 m  (constante razonable)
    # 1° de longitud ≈ 111_320 * cos(lat) m
    if center_lat is None:
        raise ValueError(
            "El raster está en CRS geográfico (grados). "
            "Pasa 'center_lat' en grados decimales para convertir correctamente."
        )
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(lat_rad)

    px_x = abs(transform.a) * m_per_deg_lon   # grados/px → metros/px (lon)
    px_y = abs(transform.e) * m_per_deg_lat   # grados/px → metros/px (lat)
    return (px_x + px_y) / 2.0


# ---------------------------------------------------
# CÁLCULO DE ALTURAS
# ---------------------------------------------------
def compute_heights(
    shadow_path,
    polys_path,
    azimuth,
    elevation,
    output="building_heights.geojson",
    max_height_m=25.0,
    # Cuántos píxeles del borde emisor se muestrean para buscar sombra.
    # Usar todos es exacto pero lento para bordes de cientos de px;
    # con 30 muestras representativas es suficiente y mucho más rápido.
    max_edge_samples=30,
    # Distancia mínima de sombra mediana válida (en píxeles)
    min_shadow_px=1,
    # Percentil del borde emisor para medir la sombra.
    # p50 = mediana (robusto), p25 = más conservador (subestima menos)
    shadow_percentile=50,
    # Si es True, imprime progreso cada N edificios
    verbose_every=100,
    # Latitud central del área (necesaria si el CRS es geográfico).
    # Si se omite, se intenta calcular desde los polígonos.
    center_lat=None,
    solar_time=None,
    lat=None,
    lon=None,
):
    """
    Estima alturas de edificios a partir de la longitud de su sombra.

    Método:
    -------
    Para cada polígono de edificio:
      1. Detectar el borde emisor: píxeles del polígono cuyo vecino inmediato
         en la dirección de proyección de la sombra está fuera del polígono.
      2. Desde cada píxel del borde emisor, trazar una línea en la dirección
         de la sombra y registrar la distancia hasta el primer píxel de sombra
         encontrado (ignorando píxeles del propio edificio).
      3. Tomar el percentil `shadow_percentile` de todas las distancias válidas
         como longitud representativa de la sombra.
      4. Convertir esa longitud a metros usando la resolución real del pixel.
      5. Altura = longitud_sombra_m × tan(elevación_solar).

    Por qué este método es más robusto que buscar desde el centroide:
      - Los polígonos de OpenBuildings no coinciden exactamente con el techo,
        así que el centroide del borde puede apuntar a una zona sin sombra.
      - Muestrear todo el borde y tomar la mediana cancela los píxeles que
        caen sobre vegetación, otros edificios o ruido de la máscara.
    """

    # ------------------------------------------------------------------
    # 1. Leer máscara de sombra
    # ------------------------------------------------------------------
    with rasterio.open(shadow_path) as src:
        shadow = src.read(1).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    # ------------------------------------------------------------------
    # 2. Ángulos solares
    # ------------------------------------------------------------------
    if solar_time is not None and lat is not None and lon is not None:
        azimuth, elevation = compute_solar_angles(lat, lon, solar_time)
        print(f"Ángulos calculados → azimut: {azimuth:.2f}°, elevación: {elevation:.2f}°")
    else:
        print(f"Ángulos manuales → azimut: {azimuth:.2f}°, elevación: {elevation:.2f}°")

    if elevation <= 0:
        raise ValueError("La elevación solar debe ser > 0°.")

    tan_elev = math.tan(math.radians(elevation))

    # ------------------------------------------------------------------
    # 3. Leer polígonos y calcular latitud central si no se proveyó
    # ------------------------------------------------------------------
    buildings_gdf = gpd.read_file(polys_path)
    if buildings_gdf.crs is None:
        raise ValueError("El shapefile no tiene CRS definido.")
    if buildings_gdf.crs != crs:
        buildings_gdf = buildings_gdf.to_crs(crs)

    if center_lat is None:
        # Estimar desde el centroide del conjunto de polígonos
        # (reproyectar a WGS84 si es necesario)
        temp = buildings_gdf.to_crs("EPSG:4326")
        center_lat = float(temp.geometry.centroid.y.mean())
        print(f"Latitud central estimada: {center_lat:.4f}°")

    # ------------------------------------------------------------------
    # 4. Tamaño real del pixel en metros
    # ------------------------------------------------------------------
    px_size_m = _pixel_size_meters(transform, crs, center_lat=center_lat)
    print(f"Resolución del pixel: {px_size_m:.4f} m/px")

    # ------------------------------------------------------------------
    # 5. Límite de búsqueda en píxeles
    # ------------------------------------------------------------------
    max_shadow_m  = (max_height_m / tan_elev) * 1.3   # margen del 30%
    max_shadow_px = max(10, int(math.ceil(max_shadow_m / px_size_m)))
    print(f"Sombra máxima: {max_shadow_m:.1f} m → {max_shadow_px} px")

    # ------------------------------------------------------------------
    # 6. Vector unitario de proyección de la sombra
    #    Azimut: N=0°, E=90°, medido en sentido horario.
    #    La sombra cae en la dirección OPUESTA al sol.
    #    En el raster: col → derecha (+X), fila → abajo (+Y).
    #      sdx = sin(az)    (componente columna)
    #      sdy = -cos(az)   (componente fila; -cos porque Y raster apunta abajo)
    # ------------------------------------------------------------------
    az_rad = math.radians(azimuth)
    sdx =  math.sin(az_rad)
    sdy = -math.cos(az_rad)

    rows, cols = shadow.shape
    print(f"Vector sombra: sdx={sdx:.4f}, sdy={sdy:.4f}")

    # Paso entero para detección del borde emisor
    step_r = int(round(sdy))
    step_c = int(round(sdx))
    # Si ambos son cero (no debería pasar), forzar al menos un paso
    if step_r == 0 and step_c == 0:
        step_r = -1 if sdy < 0 else 1

    # ------------------------------------------------------------------
    # 7. Procesamiento por edificio
    # ------------------------------------------------------------------
    results = []
    skipped_geom      = 0
    skipped_no_edge   = 0
    skipped_no_shadow = 0
    skipped_sanity    = 0

    n_buildings = len(buildings_gdf)

    for idx, row_data in buildings_gdf.iterrows():
        if verbose_every and idx % verbose_every == 0:
            pct = 100 * idx / max(n_buildings, 1)
            print(f"  {idx}/{n_buildings} ({pct:.0f}%) — resultados hasta ahora: {len(results)}")

        geom = row_data.geometry
        if geom is None or geom.is_empty:
            skipped_geom += 1
            continue

        # ── Rasterizar el polígono ────────────────────────────────────
        building_mask = rasterize(
            [(geom, 1)],
            out_shape=shadow.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )
        brows, bcols = np.where(building_mask == 1)
        if len(brows) == 0:
            skipped_geom += 1
            continue

        # ── Detectar borde emisor ─────────────────────────────────────
        # Desplazar la máscara 1 paso en la dirección de la sombra;
        # los píxeles del edificio donde el vecino desplazado = 0 son el borde.
        nr = np.clip(brows + step_r, 0, rows - 1)
        nc = np.clip(bcols + step_c, 0, cols - 1)
        neighbor_in_building = building_mask[nr, nc] == 1

        # También incluir los píxeles en el límite del raster
        on_border = (
            (brows + step_r < 0) | (brows + step_r >= rows) |
            (bcols + step_c < 0) | (bcols + step_c >= cols)
        )
        edge_mask_bool = (~neighbor_in_building) | on_border
        e_rows = brows[edge_mask_bool]
        e_cols = bcols[edge_mask_bool]

        if len(e_rows) < 2:
            skipped_no_edge += 1
            continue

        # ── Muestrear el borde emisor ─────────────────────────────────
        # Si hay más píxeles que max_edge_samples, submuestrear uniformemente
        if len(e_rows) > max_edge_samples:
            idx_sample = np.linspace(0, len(e_rows) - 1, max_edge_samples, dtype=int)
            e_rows = e_rows[idx_sample]
            e_cols = e_cols[idx_sample]

        # ── Buscar sombra desde cada píxel del borde ─────────────────
        # Para cada píxel de borde, avanzar en la dirección de la sombra
        # hasta encontrar el primer píxel de sombra (fuera del edificio).
        distances = []

        for er, ec in zip(e_rows, e_cols):
            for d in range(1, max_shadow_px + 1):
                # Posición continua → entero
                r = int(er + sdy * d + 0.5)
                c = int(ec + sdx * d + 0.5)

                # Salir si fuera del raster
                if r < 0 or r >= rows or c < 0 or c >= cols:
                    break

                # Saltar píxeles del propio edificio
                # (el polígono puede ser más grande que el techo real)
                if building_mask[r, c] == 1:
                    continue

                if shadow[r, c] == 1:
                    distances.append(d)
                    break
                # Si encontramos zona clara (no sombra, no edificio), seguimos
                # buscando hasta max_shadow_px — la sombra puede tener huecos

        if len(distances) < max(2, len(e_rows) // 4):
            # Menos del 25% del borde encontró sombra → poco fiable
            skipped_no_shadow += 1
            continue

        # ── Longitud de sombra representativa ────────────────────────
        shadow_px = float(np.percentile(distances, shadow_percentile))

        if shadow_px < min_shadow_px:
            skipped_no_shadow += 1
            continue

        # ── Conversión a metros y altura ─────────────────────────────
        shadow_m = shadow_px * px_size_m
        height_m = shadow_m * tan_elev

        # Filtro de sanidad física
        if height_m < 2.0 or height_m > max_height_m * 2.0:
            skipped_sanity += 1
            continue

        # ── Score de confianza ────────────────────────────────────────
        # hit_rate  : fracción del borde que encontró sombra
        # spread    : qué tan consistentes son las distancias (baja varianza = bien)
        # size_conf : edificios grandes → estimación más estable
        hit_rate  = len(distances) / len(e_rows)
        spread    = 1.0 - min(1.0, np.std(distances) / max(shadow_px, 1))
        size_conf = min(1.0, len(brows) / 300.0)
        confidence = round(0.5 * hit_rate + 0.3 * spread + 0.2 * size_conf, 3)

        conf_level = (
            "alta"  if confidence >= 0.7 else
            "media" if confidence >= 0.4 else
            "baja"
        )

        results.append({
            "building_id": int(idx),
            "shadow_m":    round(shadow_m, 2),
            "height_m":    round(height_m, 2),
            "confidence":  confidence,
            "conf_level":  conf_level,
            "shadow_px":   int(round(shadow_px)),
            "hit_rate":    round(hit_rate, 3),
            "edge_px":     int(len(e_rows)),
            "building_px": int(len(brows)),
            "geometry":    geom,
        })

    # ------------------------------------------------------------------
    # 8. Exportar
    # ------------------------------------------------------------------
    if len(results) == 0:
        print("\nNo se estimaron alturas.")
        print(f"  Sin geometría/píxeles : {skipped_geom}")
        print(f"  Sin borde emisor      : {skipped_no_edge}")
        print(f"  Sin sombra detectada  : {skipped_no_shadow}")
        print(f"  Fuera de rango físico : {skipped_sanity}")
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    gdf = gpd.GeoDataFrame(results, crs=crs)
    gdf.to_file(output, driver="GeoJSON")

    # ------------------------------------------------------------------
    # 9. Métricas
    # ------------------------------------------------------------------
    n_in   = len(buildings_gdf)
    n_out  = len(gdf)
    high   = (gdf["confidence"] >= 0.7).sum()
    medium = ((gdf["confidence"] >= 0.4) & (gdf["confidence"] < 0.7)).sum()
    low    = (gdf["confidence"] <  0.4).sum()

    print(f"\n{'─'*47}")
    print(f"Edificios en shapefile       : {n_in}")
    print(f"  Sin geometría / sin píxeles: {skipped_geom}")
    print(f"  Sin borde emisor           : {skipped_no_edge}")
    print(f"  Sin sombra detectada       : {skipped_no_shadow}")
    print(f"  Fuera de rango físico      : {skipped_sanity}")
    print(f"  Con altura estimada        : {n_out}  ({100*n_out/max(n_in,1):.0f}%)")
    print(f"{'─'*47}")
    print(f"  Alta confianza   (≥0.7)    : {high}")
    print(f"  Media confianza  (0.4–0.7) : {medium}")
    print(f"  Baja confianza   (<0.4)    : {low}")
    print(f"{'─'*47}")
    print(f"Altura mediana               : {gdf['height_m'].median():.1f} m")
    print(f"Altura p25–p75               : {gdf['height_m'].quantile(0.25):.1f}–{gdf['height_m'].quantile(0.75):.1f} m")
    print(f"Sombra mediana               : {gdf['shadow_m'].median():.1f} m")
    print(f"Resolución usada             : {px_size_m:.4f} m/px")
    print(f"{'─'*47}")
    print(f"Exportado a: {output}")

    return gdf


# ---------------------------------------------------
# EJECUCIÓN
# ---------------------------------------------------
if __name__ == "__main__":

    # run_pipeline(
    #     "VEXCEL_ULTRACAM/110010002512122015-0147_PROJ.tif",
    #     azimuth=323, elevation=54.76,
    # )

    compute_heights(
        shadow_path  = "VEXCEL_ULTRACAM/110010002512122015-0147_PROJ_shadow_mask_ultracam.tif",
        polys_path   = "polys_110010002512122015-0147.shp",
        azimuth      = 323,
        elevation    = 54.76,
        output       = "heights110010002512122015-0147.geojson",
        max_height_m = 25.0,
        max_edge_samples    = 30,
        shadow_percentile   = 50,
        verbose_every       = 100,
        # center_lat = 4.71,  # Descomentar si la estimación automática falla
    )

    # debug_single_building(
    #     shadow_path="VEXCEL_ULTRACAM/110010002512122015-0147_PROJ_shadow_mask_ultracam.tif",
    #     polys_path="polys_110010002512122015-0147.shp",
    #     azimuth=323,  # o probar con 323.05 si quieres
    #     elevation=17.33,
    #     idx=0  # primer edificio
    # )