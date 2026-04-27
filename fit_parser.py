"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — fit_parser.py                                   ║
║         FIT parsing · Suunto / Garmin natif · v1.0              ║
╚══════════════════════════════════════════════════════════════════╝
"""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from fitparse import FitFile
import io
from gpx_parser import haversine_vec

SEMICIRCLES_TO_DEG = 180.0 / (2 ** 31)

def parse_fit(file_bytes: bytes) -> pd.DataFrame:
    try:
        fitfile = FitFile(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Fichier FIT invalide : {e}")

    rows = []
    for record in fitfile.get_messages('record'):
        data = {d.name: d.value for d in record}
        t = data.get('timestamp')
        lat = data.get('position_lat')
        lon = data.get('position_long')
        ele = data.get('altitude') or data.get('enhanced_altitude') or 0.0
        hr = data.get('heart_rate')
        cad = data.get('cadence')
        speed = data.get('speed') or data.get('enhanced_speed')

        # Coordonnées en semicircles → degrés
        if lat is not None:
            lat = lat * SEMICIRCLES_TO_DEG
        if lon is not None:
            lon = lon * SEMICIRCLES_TO_DEG

        # Filtre HR valide
        if hr is not None and not (30 < hr < 250):
            hr = None

        # Cadence : Garmin stocke unilatérale → ×2 si < 110
        cad_ambiguous = False
        if cad is not None and cad > 30:
            cad_ambiguous = (100 <= cad < 110)
            cad = cad * 2 if cad < 110 else cad

        rows.append({
            'time': t,
            'lat': lat,
            'lon': lon,
            'ele': float(ele) if ele is not None else 0.0,
            'hr': int(hr) if hr is not None else None,
            'cadence': int(cad) if cad is not None else None,
            'cad_ambiguous': cad_ambiguous,
            'speed_raw': float(speed) if speed is not None else None,
        })

    if len(rows) < 10:
        raise ValueError("FIT trop court — moins de 10 points.")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=['lat', 'lon', 'time']).reset_index(drop=True)

    if len(df) < 10:
        raise ValueError("FIT : données GPS insuffisantes (< 10 points valides).")

    # Temps en secondes depuis le début
    df['time_s'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
    df = df[df['time_s'] >= 0].reset_index(drop=True)

    # Distance Haversine
    lat_arr = df['lat'].to_numpy()
    lon_arr = df['lon'].to_numpy()
    dist_pts = haversine_vec(lat_arr, lon_arr)
    df['distance'] = np.cumsum(dist_pts)

    # Élévation — lissage Savitzky-Golay
    ele_arr = df['ele'].to_numpy()
    if len(ele_arr) >= 11:
        ele_smooth = savgol_filter(ele_arr, window_length=11, polyorder=2)
    else:
        ele_smooth = ele_arr
    df['elevation'] = ele_smooth

    # Vitesse : utiliser speed_raw si dispo, sinon recalculer
    dt = np.diff(df['time_s'].to_numpy())
    dt = np.where(dt <= 0, 1, dt)
    if df['speed_raw'].notna().sum() > len(df) * 0.5:
        # Vitesse issue du fichier FIT (m/s)
        velocity = df['speed_raw'].fillna(0).to_numpy()
        velocity = np.concatenate([[velocity[1] if len(velocity) > 1 else 0], velocity[1:]])
    else:
        # Recalcul depuis distance/temps (même logique que gpx_parser)
        dd = np.diff(df['distance'].to_numpy())
        velocity = np.concatenate([[0.0], dd / dt])

    velocity = np.clip(velocity, 0, 12)  # cap 12 m/s = 43 km/h
    df['velocity'] = velocity

    # Grade (pente)
    d_ele = np.diff(ele_smooth)
    d_dist = np.diff(df['distance'].to_numpy())
    d_dist = np.where(d_dist < 0.1, 0.1, d_dist)
    grade = np.concatenate([[0.0], d_ele / d_dist * 100])
    grade = np.clip(grade, -40, 40)
    df['grade'] = grade
    df['dz'] = np.concatenate([[0.0], d_ele])

    # GAP flag (même logique que gpx_parser)
    df['gap_flag'] = df['grade'].abs() > 1.0

    df = df.drop(columns=['lat', 'lon', 'ele', 'time', 'speed_raw'], errors='ignore')
    return df
