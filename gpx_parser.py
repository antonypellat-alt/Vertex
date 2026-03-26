"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — gpx_parser.py                                   ║
║         GPX parsing · Haversine · Race info                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

try:
    import defusedxml.ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET  # fallback si non installé
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def haversine_vec(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Haversine vectorisée — calcule les distances entre points consécutifs."""
    R = 6371000
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = np.diff(lat_r)
    dlon = np.diff(lon_r)
    a = np.sin(dlat/2)**2 + np.cos(lat_r[:-1]) * np.cos(lat_r[1:]) * np.sin(dlon/2)**2
    dist = R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return np.concatenate([[0.0], dist])


def parse_gpx(file_bytes: bytes) -> pd.DataFrame:
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError as e:
        raise ValueError(f"Fichier GPX invalide : {e}")

    tag = root.tag
    if '{' in tag:
        ns_uri = tag[1:tag.index('}')]
        ns = {'g': ns_uri}
    else:
        ns = {'g': 'http://www.topografix.com/GPX/1/1'}

    # v3.3 : collecte tous les segments <trkseg>
    trkpts = root.findall('.//g:trkpt', ns)
    if not trkpts:
        # F2 : fallback sans namespace GPX, mais on conserve ns pour les
        # extensions Garmin imbriquées ({ns_uri}hr, {ns_uri}cad).
        # ns={} efface le namespace → find('g:ele') silencieux.
        trkpts = root.findall('.//trkpt')
        # ns reste inchangé : les extensions iter() sont résolues par endswith()

    if len(trkpts) < 10:
        raise ValueError("GPX trop court — moins de 10 points.")

    rows = []
    for pt in trkpts:
        lat = float(pt.get('lat', 0))
        lon = float(pt.get('lon', 0))

        ele_el = pt.find('g:ele', ns) if ns else pt.find('ele')
        if ele_el is None:
            ele_el = pt.find('ele')  # F2 : fallback sans namespace
        ele = float(ele_el.text) if ele_el is not None else 0.0

        time_el = pt.find('g:time', ns) if ns else pt.find('time')
        if time_el is None:
            time_el = pt.find('time')  # F2 : fallback sans namespace
        t = None
        if time_el is not None:
            try:
                ts = time_el.text.replace('Z', '').replace('z', '')
                t = datetime.fromisoformat(ts) if 'T' in ts else None
            except Exception:
                pass

        hr = None
        for hr_el in pt.iter():
            if hr_el.tag.endswith('}hr') or hr_el.tag == 'hr':
                try:
                    v = int(hr_el.text)
                    if 30 < v < 250:
                        hr = v
                except Exception:
                    pass

        cad = None
        cad_ambiguous = False
        for cad_el in pt.iter():
            if cad_el.tag.endswith('}cad') or cad_el.tag == 'cad':
                try:
                    v = int(cad_el.text)
                    if v > 30:
                        # v3.1 : Garmin stocke la cadence unilatérale → ×2
                        # sauf si déjà > 110 (export SPM total direct)
                        # F3 : zone borderline 100-110 → multiplication douteuse, on flag
                        cad = v * 2 if v < 110 else v
                        cad_ambiguous = (100 <= v < 110)  # ex: 109 → 218, non garanti
                except Exception:
                    pass

        rows.append({'lat': lat, 'lon': lon, 'elevation': ele, 'time': t,
                     'hr': hr, 'cadence': cad, 'cad_ambiguous': cad_ambiguous})

    df = pd.DataFrame(rows)

    # Filtrer les points GPS invalides (fix perdu → coordonnées 0,0)
    df = df[(df['lat'] != 0.0) | (df['lon'] != 0.0)].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError(
            "Fichier GPX invalide : moins de 10 points GPS valides après "
            "filtrage des coordonnées nulles."
        )

    # v3.3 : Haversine vectorisée — ×20 plus rapide sur GPX longs
    dist_increments = haversine_vec(df['lat'].to_numpy(), df['lon'].to_numpy())
    df['distance'] = np.cumsum(dist_increments)

    if df['time'].notna().sum() > len(df) * 0.5:
        t0 = df['time'].iloc[0]
        df['time_s'] = (
            pd.to_datetime(df['time'], errors='coerce') - t0
        ).dt.total_seconds()
        # gap_flag : segments interpolés > 30s (pause GPS, tunnel)
        df['gap_flag'] = False
        time_diff = df['time_s'].diff()
        df.loc[time_diff > 30, 'gap_flag'] = True
        df['time_s'] = df['time_s'].interpolate()
        df['timestamps_estimated'] = False
    else:
        # Timestamps absents ou insuffisants — vitesse fictive 10 km/h
        df['time_s'] = df['distance'] / (10000/3600)
        df['gap_flag'] = False
        df['timestamps_estimated'] = True

    df['dt'] = df['time_s'].diff().fillna(1).clip(lower=0.1)
    df['dd'] = df['distance'].diff().fillna(0)
    df['velocity_raw'] = (df['dd'] / df['dt']).clip(0, 12)
    df['velocity'] = df['velocity_raw'].rolling(7, center=True, min_periods=1).mean()

    # v3.3 : Savitzky-Golay sur élévation — élimine bruit GPS haute fréquence
    ele_values = df['elevation'].to_numpy()
    window = min(31, len(ele_values) if len(ele_values) % 2 != 0 else len(ele_values) - 1)
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1
    try:
        ele_smooth = savgol_filter(ele_values, window_length=window, polyorder=2)
        df['elevation_smooth'] = ele_smooth
        df['elevation_degraded'] = False
    except Exception as _sg_err:
        df['elevation_smooth'] = df['elevation']
        df['elevation_degraded'] = True

    df['dz'] = df['elevation_smooth'].diff().fillna(0)
    df['grade'] = (df['dz'] / df['dd'].replace(0, float('nan')) * 100).fillna(0).clip(-40, 40)
    df['grade'] = df['grade'].rolling(5, center=True, min_periods=1).mean()

    if df['hr'].notna().sum() > 10:
        df['hr'] = df['hr'].interpolate(limit=10).rolling(5, center=True, min_periods=1).mean()
    if df['cadence'].notna().sum() > 10:
        df['cadence'] = df['cadence'].interpolate(limit=10).rolling(5, center=True, min_periods=1).mean()

    return df.reset_index(drop=True)


def extract_race_info(df: pd.DataFrame, filename: str) -> dict:
    total_dist = df['distance'].iloc[-1]
    total_time = df['time_s'].iloc[-1]
    elevation_gain = df['dz'].clip(lower=0).sum()
    elevation_loss = abs(df['dz'].clip(upper=0).sum())
    avg_velocity = df[df['velocity'] > 0.3]['velocity'].mean()

    has_hr  = df['hr'].notna().sum() > len(df) * 0.3
    hr_coverage_pct = round(df['hr'].notna().sum() / len(df) * 100, 1)
    # Seuil 80 spm : filtre les artefacts capteur poignet (post-multiplication ×2)
    has_cad = df['cadence'].notna().sum() > len(df) * 0.3

    hr_mean  = df.loc[df['hr'] > 50, 'hr'].mean() if has_hr else None
    hr_max   = df.loc[df['hr'] > 50, 'hr'].max()  if has_hr else None
    cad_mean = df.loc[df['cadence'] > 80, 'cadence'].mean() if has_cad else None

    return {
        'name': filename.replace('.gpx', '').replace('.tcx', '').replace('_', ' ').title(),
        'distance_km': total_dist / 1000,
        'total_time_s': total_time,
        'elevation_gain': elevation_gain,
        'elevation_loss': elevation_loss,
        'max_elevation': df['elevation'].max(),
        'min_elevation': df['elevation'].min(),
        'avg_velocity_ms': avg_velocity,
        'has_hr': has_hr,
        'hr_coverage_pct': hr_coverage_pct,
        'has_cad': has_cad,
        'hr_mean': hr_mean,
        'hr_max': hr_max,
        'cad_mean': cad_mean,
        'elevation_degraded': bool(df['elevation_degraded'].any()
                                   if 'elevation_degraded' in df.columns
                                   else False),
        'timestamps_estimated': bool(df['timestamps_estimated'].any()
                                     if 'timestamps_estimated' in df.columns
                                     else False),
    }
