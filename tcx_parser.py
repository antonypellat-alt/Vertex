"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — tcx_parser.py                                   ║
║         TCX parsing · Polar / Garmin                            ║
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

from gpx_parser import haversine_vec


# Namespaces TCX standard (Garmin Training Center)
NS = {
    'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
    'ext': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2',
}


def parse_tcx(file_bytes: bytes) -> pd.DataFrame:
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError as e:
        raise ValueError(f"Fichier TCX invalide : {e}")

    # Détection namespace dynamique (Polar peut varier)
    tag = root.tag
    if '{' in tag:
        ns_uri = tag[1:tag.index('}')]
        ns = {'tcx': ns_uri}
    else:
        ns = {'tcx': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2'}

    trackpoints = root.findall('.//tcx:Trackpoint', ns)
    if not trackpoints:
        # Fallback sans namespace
        trackpoints = root.findall('.//Trackpoint')
        ns = {}

    if len(trackpoints) < 10:
        raise ValueError("TCX trop court — moins de 10 points.")

    rows = []
    for tp in trackpoints:
        def find(tag):
            el = tp.find(f'tcx:{tag}', ns) if ns else tp.find(tag)
            return el

        # Temps
        t = None
        time_el = find('Time')
        if time_el is not None:
            try:
                ts = time_el.text.replace('Z', '').replace('z', '')
                t = datetime.fromisoformat(ts) if 'T' in ts else None
            except Exception:
                pass

        # Position GPS
        lat, lon = 0.0, 0.0
        pos_el = find('Position')
        if pos_el is not None:
            lat_el = pos_el.find('tcx:LatitudeDegrees', ns) if ns else pos_el.find('LatitudeDegrees')
            lon_el = pos_el.find('tcx:LongitudeDegrees', ns) if ns else pos_el.find('LongitudeDegrees')
            if lat_el is not None:
                lat = float(lat_el.text)
            if lon_el is not None:
                lon = float(lon_el.text)

        # Altitude
        ele = 0.0
        ele_el = find('AltitudeMeters')
        if ele_el is not None:
            try:
                ele = float(ele_el.text)
            except Exception:
                pass

        # FC — Fix B4 : lecture stricte HeartRateBpm/Value uniquement.
        # Le fallback .//Value supprimé — risque de capturer distance/vitesse.
        # Si absent → hr=None, has_hr=False gère le cas proprement en aval.
        hr = None
        hr_bpm_el = tp.find('.//tcx:HeartRateBpm/tcx:Value', ns) if ns else tp.find('.//HeartRateBpm/Value')
        if hr_bpm_el is not None:
            try:
                v = int(hr_bpm_el.text)
                if 30 < v < 250:
                    hr = v
            except Exception:
                pass

        # Cadence (RunCadence dans extensions Garmin, ou CadenceSpm Polar)
        cad = None
        for cad_tag in ['RunCadence', 'CadenceSpm', 'Cadence']:
            for el in tp.iter():
                if el.tag.endswith(f'}}{cad_tag}') or el.tag == cad_tag:
                    try:
                        v = int(el.text)
                        if v > 30:
                            cad = v * 2 if v < 110 else v
                    except Exception:
                        pass
                    break

        rows.append({'lat': lat, 'lon': lon, 'elevation': ele,
                     'time': t, 'hr': hr, 'cadence': cad, 'cad_ambiguous': False})

    df = pd.DataFrame(rows)

    # Filtre points sans GPS (lat=0, lon=0)
    df = df[(df['lat'] != 0.0) | (df['lon'] != 0.0)].reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("TCX : pas assez de points GPS valides.")

    # Haversine (réutilisée depuis gpx_parser)
    dist_increments = haversine_vec(df['lat'].to_numpy(), df['lon'].to_numpy())
    df['distance'] = np.cumsum(dist_increments)

    if df['time'].notna().sum() > len(df) * 0.5:
        t0 = df['time'].iloc[0]
        df['time_s'] = (
            pd.to_datetime(df['time'], errors='coerce') - t0
        ).dt.total_seconds()
        df['gap_flag'] = False
        time_diff = df['time_s'].diff()
        df.loc[time_diff > 30, 'gap_flag'] = True
        df['time_s'] = df['time_s'].interpolate()
    else:
        df['time_s'] = df['distance'] / (10000 / 3600)
        df['gap_flag'] = False

    df['dt'] = df['time_s'].diff().fillna(1).clip(lower=0.1)
    df['dd'] = df['distance'].diff().fillna(0)
    df['velocity_raw'] = (df['dd'] / df['dt']).clip(0, 12)
    df['velocity'] = df['velocity_raw'].rolling(7, center=True, min_periods=1).mean()

    ele_values = df['elevation'].to_numpy()
    window = min(31, len(ele_values) if len(ele_values) % 2 != 0 else len(ele_values) - 1)
    window = max(window, 5)
    if window % 2 == 0:
        window -= 1
    try:
        df['elevation_smooth'] = savgol_filter(ele_values, window_length=window, polyorder=2)
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
