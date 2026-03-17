"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — GPX Performance Analyzer  |  app.py            ║
║         FC · Cadence · GAP · Zones · Découplage · v3.2          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF

# ══════════════════════════════════════════════════════════════════
# 0 — PAGE CONFIG & GLOBAL STYLE
# ══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="VERTEX · Performance Intelligence",
    page_icon="▲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@300;400;600;700;900&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #080E14 !important;
    color: #C8D4DC !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0D1520 !important; }
.block-container { padding-top: 2rem !important; max-width: 1200px; }

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(65,200,232,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(65,200,232,0.03) 1px, transparent 1px);
    background-size: 32px 32px;
    pointer-events: none; z-index: 0;
}

h1, h2, h3 {
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.1em !important;
}

[data-testid="metric-container"] {
    background: #0D1520 !important;
    border: 1px solid #152030 !important;
    border-top: 2px solid #41C8E8 !important;
    border-radius: 2px !important;
    padding: 1.2rem !important;
}
[data-testid="metric-container"] label {
    color: #3A5060 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #41C8E8 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    color: #4A6070 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
}

.stButton > button {
    background: transparent !important;
    border: 1px solid #41C8E8 !important;
    color: #41C8E8 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.18em !important;
    font-size: 0.95rem !important;
    border-radius: 2px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(65,200,232,0.08) !important;
    box-shadow: 0 0 24px rgba(65,200,232,0.2) !important;
}

.stDownloadButton > button {
    background: rgba(65,200,232,0.08) !important;
    border: 1px solid #41C8E8 !important;
    color: #41C8E8 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    border-radius: 2px !important;
}

[data-testid="stFileUploader"] {
    background: #0D1520 !important;
    border: 1px dashed #152030 !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #41C8E8 !important; }

.stProgress > div > div > div { background: #41C8E8 !important; }
hr { border-color: #152030 !important; }
.stSpinner > div { border-top-color: #41C8E8 !important; }

.stTextInput > div > div > input {
    background: #0D1520 !important;
    border: 1px solid #152030 !important;
    color: #C8D4DC !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextInput > div > div > input:focus { border-color: #41C8E8 !important; }
.stSelectbox > div > div {
    background: #0D1520 !important;
    border: 1px solid #152030 !important;
    border-radius: 2px !important;
}
.stNumberInput > div > div > input {
    background: #0D1520 !important;
    border: 1px solid #152030 !important;
    color: #C8D4DC !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
}

.hud-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem; color: #2A4050;
    letter-spacing: 0.22em; text-transform: uppercase;
}
.vertex-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 5rem; font-weight: 900;
    letter-spacing: 0.25em; line-height: 1;
    color: #ffffff;
}
.vertex-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem; color: #41C8E8;
    letter-spacing: 0.3em;
}
.section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    letter-spacing: 0.22em; color: #2A4050;
    text-transform: uppercase; border-bottom: 1px solid #152030;
    padding-bottom: 6px; margin-bottom: 1rem;
}
.badge-endurance {
    display:inline-block; padding:6px 18px;
    background:rgba(65,200,232,0.1); border:1px solid #41C8E8;
    color:#41C8E8; font-family:'Barlow Condensed',sans-serif;
    font-weight:700; letter-spacing:0.12em; font-size:1.1rem; border-radius:2px;
}
.badge-explosif {
    display:inline-block; padding:6px 18px;
    background:rgba(200,168,75,0.1); border:1px solid #C8A84B;
    color:#C8A84B; font-family:'Barlow Condensed',sans-serif;
    font-weight:700; letter-spacing:0.12em; font-size:1.1rem; border-radius:2px;
}
.badge-fragile {
    display:inline-block; padding:6px 18px;
    background:rgba(200,72,80,0.1); border:1px solid #C84850;
    color:#C84850; font-family:'Barlow Condensed',sans-serif;
    font-weight:700; letter-spacing:0.12em; font-size:1.1rem; border-radius:2px;
}
.zone-bar {
    height: 8px; border-radius: 1px; margin-bottom: 4px;
}
.km-table {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    width: 100%;
    border-collapse: collapse;
}
.km-table th {
    color: #2A4050;
    letter-spacing: 0.15em;
    font-size: 0.6rem;
    padding: 6px 8px;
    border-bottom: 1px solid #152030;
    text-align: left;
}
.km-table td {
    padding: 5px 8px;
    border-bottom: 1px solid #0D1520;
    color: #7A9AAA;
}
.km-table tr:hover td { background: #0D1520; color: #C8D4DC; }
.alert-box {
    padding: 12px 16px;
    background: #0D1520;
    border-left: 2px solid rgba(65,200,232,0.3);
    margin-bottom: 10px;
    display: flex;
    gap: 16px;
    align-items: flex-start;
}
.alert-warn { border-left-color: rgba(200,168,75,0.6) !important; }
.alert-crit { border-left-color: rgba(200,72,80,0.6) !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 1 — GPX PARSER (v3.1 : correction cadence ×2)
# ══════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


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

    trkpts = root.findall('.//g:trkpt', ns)
    if not trkpts:
        trkpts = root.findall('.//trkpt')
        ns = {}

    if len(trkpts) < 10:
        raise ValueError("GPX trop court — moins de 10 points.")

    rows = []
    for pt in trkpts:
        lat = float(pt.get('lat', 0))
        lon = float(pt.get('lon', 0))

        ele_el = pt.find('g:ele', ns) if ns else pt.find('ele')
        ele = float(ele_el.text) if ele_el is not None else 0.0

        time_el = pt.find('g:time', ns) if ns else pt.find('time')
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
        for cad_el in pt.iter():
            if cad_el.tag.endswith('}cad') or cad_el.tag == 'cad':
                try:
                    v = int(cad_el.text)
                    if v > 30:
                        # FIX v3.1 : Garmin stocke la cadence unilatérale (une jambe)
                        # dans les extensions GPX. On multiplie par 2 pour obtenir
                        # les pas/min totaux (SPM), sauf si la valeur est déjà > 110
                        # (certains appareils exportent directement en SPM total).
                        cad = v * 2 if v < 110 else v
                except Exception:
                    pass

        rows.append({'lat': lat, 'lon': lon, 'elevation': ele, 'time': t,
                     'hr': hr, 'cadence': cad})

    df = pd.DataFrame(rows)

    dist_cum = [0.0]
    for i in range(1, len(df)):
        d = haversine(df.loc[i-1,'lat'], df.loc[i-1,'lon'],
                      df.loc[i,'lat'],   df.loc[i,'lon'])
        dist_cum.append(dist_cum[-1] + d)
    df['distance'] = dist_cum

    if df['time'].notna().sum() > len(df) * 0.5:
        t0 = df['time'].iloc[0]
        df['time_s'] = df['time'].apply(
            lambda t: (t - t0).total_seconds() if pd.notna(t) else None
        )
        df['time_s'] = df['time_s'].interpolate()
    else:
        df['time_s'] = df['distance'] / (10000/3600)

    df['dt'] = df['time_s'].diff().fillna(1).clip(lower=0.1)
    df['dd'] = df['distance'].diff().fillna(0)
    df['velocity_raw'] = (df['dd'] / df['dt']).clip(0, 12)
    df['velocity'] = df['velocity_raw'].rolling(7, center=True, min_periods=1).mean()

    df['dz'] = df['elevation'].diff().fillna(0)
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
    # FIX v3.1 : seuil filtre cadence relevé à 80 (post-multiplication ×2)
    has_cad = df['cadence'].notna().sum() > len(df) * 0.3

    hr_mean  = df.loc[df['hr'] > 50, 'hr'].mean() if has_hr else None
    hr_max   = df.loc[df['hr'] > 50, 'hr'].max()  if has_hr else None
    cad_mean = df.loc[df['cadence'] > 80, 'cadence'].mean() if has_cad else None

    return {
        'name': filename.replace('.gpx','').replace('_',' ').title(),
        'distance_km': total_dist / 1000,
        'total_time_s': total_time,
        'elevation_gain': elevation_gain,
        'elevation_loss': elevation_loss,
        'max_elevation': df['elevation'].max(),
        'min_elevation': df['elevation'].min(),
        'avg_velocity_ms': avg_velocity,
        'has_hr': has_hr,
        'has_cad': has_cad,
        'hr_mean': hr_mean,
        'hr_max': hr_max,
        'cad_mean': cad_mean,
    }


# ══════════════════════════════════════════════════════════════════
# 2 — ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════

def gap_correction(velocity_ms: float, grade_pct: float) -> float:
    g = grade_pct / 100.0
    energy_flat  = 3.6
    energy_slope = (155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6)
    correction   = max(0.5, min(2.5, energy_slope / energy_flat))
    return velocity_ms / correction if correction > 0 else velocity_ms

def v_to_pace(v: float) -> str:
    if not v or v <= 0.1: return "--:--"
    s = 1000 / v
    return f"{int(s//60)}:{int(s%60):02d}"

def grade_pace_profile(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['grade_abs'] = df['grade'].abs()
    bins   = [0, 5, 10, 15, 100]
    labels = ["0–5%", "5–10%", "10–15%", ">15%"]
    df['bin'] = pd.cut(df['grade_abs'], bins=bins, labels=labels, right=False)
    profile = (
        df[df['velocity'] > 0.3]
        .groupby('bin', observed=True)['velocity']
        .mean().reset_index()
    )
    profile['Allure'] = profile['velocity'].apply(v_to_pace)
    profile.columns = ['Tranche pente', 'Vitesse (m/s)', 'Allure (min/km)']
    return profile

def fatigue_index(df: pd.DataFrame) -> dict:
    df = df.copy()
    df['gap'] = df.apply(lambda r: gap_correction(r['velocity'], r['grade']), axis=1)
    total = df['time_s'].max()
    q_size = total / 4
    quartiles = {}
    for i in range(1, 5):
        mask = (df['time_s'] >= (i-1)*q_size) & (df['time_s'] < i*q_size)
        q_df = df[mask & (df['velocity'] > 0.3)]
        quartiles[f'Q{i}'] = round(q_df['gap'].mean(), 4) if len(q_df) > 5 else float('nan')
    q1 = quartiles.get('Q1', 0)
    q4 = quartiles.get('Q4', 0)
    ratio = q4 / q1 if q1 and q1 > 0 else float('nan')
    return {'quartiles': quartiles, 'decay_ratio': ratio,
            'decay_pct': (1 - ratio)*100 if not math.isnan(ratio) else float('nan')}

def flat_pace_estimate(df: pd.DataFrame) -> float:
    flat_mask = (df['grade'].abs() < 3) & (df['velocity'] > 0.3)
    fdf = df[flat_mask]
    if len(fdf) < 10:
        return df[df['velocity'] > 0.3]['velocity'].median()
    return fdf.apply(lambda r: gap_correction(r['velocity'], r['grade']), axis=1).median()

def classify_profile(decay_ratio: float, flat_v: float) -> str:
    if math.isnan(decay_ratio): return "PROFIL INCONNU"
    if decay_ratio >= 0.93: return "PROFIL ENDURANCE"
    if decay_ratio >= 0.85: return "PROFIL EXPLOSIF"
    return "PROFIL FRAGILE"


# ── Zones FC ────────────────────────────────────────────────────

ZONE_NAMES = {
    'Z1': 'Récupération',
    'Z2': 'Endurance fondamentale',
    'Z3': 'Tempo / Seuil aérobie',
    'Z4': 'Seuil lactate',
    'Z5': 'VO2max / Anaérobie',
}

def compute_hr_zones(df: pd.DataFrame, fcmax: int, custom_zones: dict = None) -> dict:
    # Mode manuel : zones en bpm absolus fournis par l'utilisateur
    if custom_zones:
        zone_bpm = {z: (int(v[0]), int(v[1])) for z, v in custom_zones.items()}
        zone_time = {z: 0.0 for z in zone_bpm}
        valid = df[df['hr'] > 50].copy()
        valid['dt'] = valid['time_s'].diff().fillna(0).clip(0, 30)
        for _, row in valid.iterrows():
            hr = row['hr']
            for z, (lo, hi) in zone_bpm.items():
                if lo <= hr < hi:
                    zone_time[z] += row['dt']
                    break
        total = sum(zone_time.values())
        zone_pct = {z: (t/total*100 if total > 0 else 0) for z, t in zone_time.items()}
        return {'time': zone_time, 'pct': zone_pct, 'bpm': zone_bpm, 'fcmax': fcmax, 'mode': 'manual'}

    # Mode auto : % FCmax (défaut)
    thresholds = {
        'Z1': (0,    0.60),
        'Z2': (0.60, 0.70),
        'Z3': (0.70, 0.80),
        'Z4': (0.80, 0.90),
        'Z5': (0.90, 1.01),
    }
    zone_time = {z: 0.0 for z in thresholds}
    valid = df[df['hr'] > 50].copy()
    valid['dt'] = valid['time_s'].diff().fillna(0).clip(0, 30)
    for _, row in valid.iterrows():
        pct = row['hr'] / fcmax
        for z, (lo, hi) in thresholds.items():
            if lo <= pct < hi:
                zone_time[z] += row['dt']
                break
    total = sum(zone_time.values())
    zone_pct = {z: (t/total*100 if total > 0 else 0) for z, t in zone_time.items()}
    zone_bpm = {z: (int(lo*fcmax), int(hi*fcmax)) for z, (lo, hi) in thresholds.items()}
    return {'time': zone_time, 'pct': zone_pct, 'bpm': zone_bpm, 'fcmax': fcmax, 'mode': 'auto'}


# ── Découplage cardiaque ────────────────────────────────────────

def cardiac_drift(df: pd.DataFrame) -> dict:
    flat = df[(df['grade'].abs() < 3) & (df['velocity'] > 0.3) & (df['hr'] > 80)].copy()
    if len(flat) < 20:
        return {'ef1': None, 'ef2': None, 'drift_pct': None, 'quartiles': {}}

    flat = flat.sort_values('distance').reset_index(drop=True)
    mid = len(flat) // 2

    def ef(sub):
        if len(sub) == 0: return None
        v  = sub['velocity'].mean()
        hr = sub['hr'].mean()
        return (v / hr) * 100 if hr > 0 else None

    ef1 = ef(flat.iloc[:mid])
    ef2 = ef(flat.iloc[mid:])
    drift = ((ef2 - ef1) / ef1 * 100) if (ef1 and ef2 and ef1 > 0) else None

    total_dist = df['distance'].iloc[-1]
    q_size = total_dist / 4
    ef_q = {}
    for i in range(1, 5):
        q = flat[(flat['distance'] >= (i-1)*q_size) & (flat['distance'] < i*q_size)]
        ef_q[f'Q{i}'] = ef(q)

    return {'ef1': ef1, 'ef2': ef2, 'drift_pct': drift, 'quartiles': ef_q}


# ── Splits par km ───────────────────────────────────────────────

def compute_km_splits(df: pd.DataFrame) -> list:
    splits = []
    total_dist = df['distance'].iloc[-1]
    n_km = int(total_dist / 1000)

    for km in range(n_km):
        lo = km * 1000
        hi = (km + 1) * 1000
        seg = df[(df['distance'] >= lo) & (df['distance'] < hi)]
        if len(seg) < 3:
            continue

        dt = seg['time_s'].iloc[-1] - seg['time_s'].iloc[0]
        dd = seg['distance'].iloc[-1] - seg['distance'].iloc[0]
        dz_pos = seg['dz'].clip(lower=0).sum()
        dz_neg = abs(seg['dz'].clip(upper=0).sum())

        pace = dt if dt > 0 else None
        v = dd / dt if dt > 0 else None
        gap_v = gap_correction(v, seg['grade'].mean()) if v else None

        hr_mean  = seg.loc[seg['hr'] > 50, 'hr'].mean() if seg['hr'].notna().any() else None
        # FIX v3.1 : seuil filtre cadence relevé à 80
        cad_mean = seg.loc[seg['cadence'] > 80, 'cadence'].mean() if seg['cadence'].notna().any() else None

        splits.append({
            'km':      km + 1,
            'pace_s':  pace,
            'pace':    v_to_pace(v) if v else '--:--',
            'gap':     v_to_pace(gap_v) if gap_v else '--:--',
            'd_pos':   int(dz_pos),
            'd_neg':   int(dz_neg),
            'hr':      round(hr_mean) if hr_mean else None,
            'cadence': round(cad_mean) if cad_mean else None,
            'velocity': v,
        })
    return splits


# ── FC selon pente ───────────────────────────────────────────────

def hr_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[(df['hr'] > 80) & (df['velocity'] > 0.3)].copy()
    bins   = list(range(-20, 25, 5))
    labels = [f"{b}%" for b in bins[:-1]]
    valid['grade_bin'] = pd.cut(valid['grade'], bins=bins, labels=labels, right=False)
    result = valid.groupby('grade_bin', observed=True).agg(
        hr_mean=('hr', 'mean'),
        n=('hr', 'count'),
        pace_mean=('velocity', 'mean')
    ).reset_index()
    result = result[result['n'] > 30]
    return result


# ── Analyse cadence ─────────────────────────────────────────────

def cadence_analysis(df: pd.DataFrame) -> dict:
    # FIX v3.1 : seuil filtre relevé à 80 (post-multiplication ×2)
    valid = df[df['cadence'] > 80]['cadence']
    if len(valid) < 10:
        return {'mean': None, 'max': None, 'dist': {}, 'optimal_pct': None}

    # FIX v3.1 : bins recalculés pour des valeurs post-×2 (150–210 ppm réaliste trail)
    bins = {'<150': 0, '150-160': 0, '160-170': 0, '170-180': 0,
            '180-190': 0, '190-200': 0, '>200': 0}
    for c in valid:
        if c < 150:     bins['<150'] += 1
        elif c < 160:   bins['150-160'] += 1
        elif c < 170:   bins['160-170'] += 1
        elif c < 180:   bins['170-180'] += 1
        elif c < 190:   bins['180-190'] += 1
        elif c <= 200:  bins['190-200'] += 1
        else:           bins['>200'] += 1

    total = len(valid)
    pct = {k: v/total*100 for k, v in bins.items()}
    # Zone optimale trail : 170-190 ppm (équivalent 85-95 ppm unilatéral Garmin)
    optimal_pct = pct.get('170-180', 0) + pct.get('180-190', 0) + pct.get('190-200', 0)

    return {
        'mean': valid.mean(),
        'max': valid.max(),
        'dist': pct,
        'optimal_pct': optimal_pct,
    }


# ══════════════════════════════════════════════════════════════════
# 3 — RECOMMANDATIONS COACH (v3.1)
# ══════════════════════════════════════════════════════════════════

def generate_coach_recommendations(
    profile: str,
    fi: dict,
    drift: dict,
    cad_analysis: dict,
    info: dict,
    fcmax: int,
) -> list:
    recs = []
    dr = fi.get('decay_ratio', float('nan'))
    dp = fi.get('decay_pct', float('nan'))
    drift_pct = drift.get('drift_pct')
    cad_mean = cad_analysis.get('mean')
    optimal_pct = cad_analysis.get('optimal_pct', 0)
    hr_mean = info.get('hr_mean')

    if hr_mean and fcmax:
        hr_pct = hr_mean / fcmax * 100
        if hr_pct > 90:
            recs.append({
                'level': 'crit',
                'title': 'Intensité maximale détectée',
                'body': f"FC moyenne à {hr_pct:.0f}% FCmax sur toute la course. "
                        "Tu as couru quasi intégralement en Z5. C'est une performance remarquable mais "
                        "la récupération doit être prioritaire : minimum 5-7 jours sans intensité. "
                        "Les 2 semaines suivantes : volume Z1/Z2 exclusivement."
            })
        elif hr_pct > 85:
            recs.append({
                'level': 'warn',
                'title': 'Course menée au seuil lactate',
                'body': f"FC moyenne à {hr_pct:.0f}% FCmax. Tu as couru majoritairement en Z4-Z5. "
                        "Pour progresser, il faut polariser l'entraînement : 80% du volume en Z1/Z2, "
                        "20% en Z4/Z5. Une FC si haute en moyenne sur un ultra suggère un manque de volume aérobie de base."
            })

    if drift_pct is not None:
        if drift_pct < -5:
            recs.append({
                'level': 'warn',
                'title': f'Découplage cardiaque : {drift_pct:.1f}%',
                'body': f"L'EF (Efficiency Factor) se dégrade de {abs(drift_pct):.1f}% entre la 1ère et 2ème moitié "
                        "sur terrain plat. Signal d'une fatigue musculaire ou glycémique. "
                        "Axe de travail : sorties longues >3h en Z2 strict + stratégie nutritionnelle "
                        "(1 gel ou 30-40g glucides/30min après 1h d'effort)."
            })
        elif drift_pct > -2:
            recs.append({
                'level': 'info',
                'title': f'Très bon découplage cardiaque : {drift_pct:.1f}%',
                'body': "L'efficacité de course est quasi-constante sur l'ensemble de l'effort. "
                        "Ton endurance aérobie est solide. Pour continuer à progresser : "
                        "introduis 1 séance/semaine de travail spécifique au seuil (2×20min à FC seuil)."
            })

    if not math.isnan(dp):
        if dp > 15:
            recs.append({
                'level': 'crit',
                'title': f'Décrochage GAP critique : -{dp:.1f}%',
                'body': f"Perte de vitesse GAP de {dp:.1f}% entre Q1 et Q4. "
                        "Le moteur s'est clairement éteint en 2ème partie de course. "
                        "Travail prioritaire : 4 semaines de volume Z2 pur (65-72% FCmax), "
                        "sorties longues progressives +200m D+/semaine. "
                        "Revoir aussi la stratégie de départ : le Q1 était probablement trop rapide."
            })
        elif dp > 7:
            recs.append({
                'level': 'warn',
                'title': f'Décrochage GAP modéré : -{dp:.1f}%',
                'body': f"Perte de {dp:.1f}% de vitesse ajustée en fin de course. "
                        "Ajoute 1 sortie longue hebdomadaire avec les 30 dernières minutes "
                        "en allure soutenue (negative split training). "
                        "Simule les conditions course : nutrition identique, même dénivelé."
            })
        elif dp < 4 and not math.isnan(dr):
            recs.append({
                'level': 'info',
                'title': f"Très bonne gestion de l'effort : -{dp:.1f}% GAP",
                'body': "Ratio Q4/Q1 excellent. Tu as géré ton allure de façon optimale. "
                        "Pour franchir un palier : travaille maintenant la vitesse de base — "
                        "2×/semaine de fractionné court (8-10×200m ou 6-8×400m à 95-100% VMA)."
            })

    if cad_mean:
        # FIX v3.1 : seuils cadence recalculés en SPM total (×2)
        if cad_mean < 168:
            recs.append({
                'level': 'warn',
                'title': f'Cadence basse : {cad_mean:.0f} spm',
                'body': f"Cadence moyenne de {cad_mean:.0f} spm, sous le seuil optimal (170-190 spm). "
                        "Une cadence basse = pas plus longs = plus de stress articulaire + frein à l'allure. "
                        "Exercice : 2×10min/semaine de 'cadence drills' à 180 spm avec métronome. "
                        "Objectif progressif : +4-6 spm par mois."
            })
        if optimal_pct < 60:
            recs.append({
                'level': 'warn',
                'title': f'Régularité cadence : {optimal_pct:.0f}% du temps en zone optimale',
                'body': "Moins de 60% du temps dans la plage cadence optimale (170-200 spm). "
                        "La variabilité de cadence est un signe de fatigue neuromusculaire ou de technique à travailler. "
                        "Priorité : montées en marchant actif + cadence rythmée en descente (180 spm)."
            })

    q_times = fi.get('quartiles', {})
    if all(not math.isnan(v) for v in q_times.values() if v):
        q1_val = q_times.get('Q1', float('nan'))
        q4_val = q_times.get('Q4', float('nan'))
        if not math.isnan(q1_val) and not math.isnan(q4_val):
            if q4_val / q1_val < 0.80:
                recs.append({
                    'level': 'warn',
                    'title': 'Endurance spécifique insuffisante en fin de course',
                    'body': "Q4/Q1 < 0.80 : tu perds plus de 20% de vitesse GAP en dernière partie. "
                            "Simulation de fin de course : inclure des blocs de 45-60 min à allure course "
                            "en fin de sortie longue (run fatigue). "
                            "Travaille aussi le ravitaillement : recalculer l'apport calorique/heure."
                })

    recs.append({
        'level': 'info',
        'title': 'Point fort à capitaliser',
        'body': _strength_advice(profile, drift_pct, cad_mean, dp)
    })

    return recs[:6]


def _strength_advice(profile, drift_pct, cad_mean, dp):
    if profile == "PROFIL ENDURANCE":
        return ("Ton moteur aérobie est ton atout principal. "
                "Capitalise dessus en ajoutant du volume D+ : vise +10% de dénivelé cumulé par cycle de 3 semaines. "
                "Introduis du travail de côtes longues (6-10min à VMA montée) pour transformer cette endurance en puissance.")
    if drift_pct and drift_pct > -3:
        return ("Ton efficacité cardiaque est excellente et stable sur l'effort. "
                "C'est le signe d'une bonne base aérobie. "
                "Tu peux désormais monter en intensité sans risque : 1 séance VMA/semaine bien dosée.")
    if cad_mean and cad_mean >= 180:
        return ("Belle cadence de course — c'est un indicateur de technique et d'économie de course solide. "
                "Maintiens ce travail technique et oriente-toi vers des chaussures plus légères "
                "pour exploiter pleinement cette économie de foulée.")
    return ("La régularité de ton allure sur terrain difficile montre une bonne lecture du parcours. "
            "Pour progresser : travaille la spécificité du profil de ta prochaine course "
            "(enchaîner montée/descente en blocs de 15-20 min sans récupération).")


# ══════════════════════════════════════════════════════════════════
# 4 — CHARTS
# ══════════════════════════════════════════════════════════════════

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono", color="#3A5060", size=10),
    margin=dict(l=0, r=0, t=20, b=0),
    xaxis=dict(gridcolor="#152030", zeroline=False, showgrid=True),
    yaxis=dict(gridcolor="#152030", zeroline=False, showgrid=True),
    showlegend=False,
)

def _layout(**overrides):
    base = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in overrides}
    base.update(overrides)
    return base


def chart_elevation(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['elevation'],
        mode='lines', line=dict(color='#41C8E8', width=1.5),
        fill='tozeroy', fillcolor='rgba(65,200,232,0.06)',
    ))
    fig.update_layout(**_layout(height=180, yaxis_title="m", xaxis_title="km"))
    return fig


def chart_pace(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    pace = df['velocity'].apply(lambda v: 1000/v/60 if v > 0.3 else None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=pace,
        mode='lines', line=dict(color='#C8A84B', width=1.5),
    ))
    fig.update_layout(**_layout(
        height=180,
        yaxis=dict(gridcolor="#152030", zeroline=False, autorange='reversed'),
        yaxis_title="min/km",
        xaxis_title="km",
    ))
    return fig


def chart_hr(df: pd.DataFrame, fcmax: int) -> go.Figure:
    dist_km = df['distance'] / 1000
    fig = go.Figure()
    zone_bounds = [(0, 0.60, '#1A3A4A'), (0.60, 0.70, '#1A5060'),
                   (0.70, 0.80, '#1A8AAA'), (0.80, 0.90, '#C8A84B'), (0.90, 1.0, '#C84850')]
    for lo, hi, color in zone_bounds:
        fig.add_hrect(y0=lo*fcmax, y1=hi*fcmax,
                      fillcolor=color, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['hr'],
        mode='lines', line=dict(color='#C84850', width=1.5),
        name='FC'
    ))
    fig.add_hline(y=fcmax, line_dash='dot', line_color='rgba(200,72,80,0.3)', line_width=1)
    fig.update_layout(**_layout(
        height=200,
        yaxis=dict(gridcolor="#152030", zeroline=False),
        yaxis_title="bpm",
        xaxis_title="km",
    ))
    return fig


def chart_hr_pace_overlay(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    pace = df['velocity'].apply(lambda v: 1000/v/60 if v > 0.3 else None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['hr'],
        mode='lines', line=dict(color='#C84850', width=1.5),
        name='FC', yaxis='y1'
    ))
    fig.add_trace(go.Scatter(
        x=dist_km, y=pace,
        mode='lines', line=dict(color='#C8A84B', width=1.5),
        name='Allure', yaxis='y2'
    ))
    fig.update_layout(**_layout(
        height=220,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4A6070', size=9)),
        yaxis=dict(title='FC (bpm)', gridcolor="#152030", zeroline=False, color='#C84850'),
        yaxis2=dict(title='Allure (min/km)', overlaying='y', side='right',
                    autorange='reversed', color='#C8A84B', gridcolor='rgba(0,0,0,0)'),
        xaxis=dict(gridcolor="#152030", zeroline=False, title='km'),
    ))
    return fig


def chart_quartiles(quartiles: dict) -> go.Figure:
    labels = list(quartiles.keys())
    values = [round(v, 4) if not math.isnan(v) else 0 for v in quartiles.values()]
    colors_bar = ['#41C8E8' if i == 0 else ('#C84850' if i == 3 else '#1A3A4A')
                  for i in range(4)]
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors_bar,
        text=[f"{v:.3f}" for v in values],
        textposition="outside", textfont=dict(color="#4A6070", size=10),
    ))
    fig.update_layout(**_layout(height=240))
    return fig


def chart_grade_dist(df: pd.DataFrame) -> go.Figure:
    bins   = [0, 5, 10, 15, 100]
    labels = ["0–5%", "5–10%", "10–15%", ">15%"]
    df2 = df.copy()
    df2['grade_abs'] = df2['grade'].abs()
    df2['bin'] = pd.cut(df2['grade_abs'], bins=bins, labels=labels, right=False)
    df2['dd'] = df2['distance'].diff().fillna(0)
    dist_by_bin = df2.groupby('bin', observed=True)['dd'].sum() / 1000
    fig = go.Figure(go.Bar(
        x=list(dist_by_bin.index), y=list(dist_by_bin.values),
        marker_color=['#41C8E8','#1A8AAA','#1A5060','#0D2A34'],
        text=[f"{v:.1f} km" for v in dist_by_bin.values],
        textposition="outside", textfont=dict(color="#4A6070", size=10),
    ))
    fig.update_layout(**_layout(height=240))
    return fig


def chart_gap_profile(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2['gap'] = df2.apply(lambda r: gap_correction(r['velocity'], r['grade']), axis=1)
    dist_km = df2['distance'] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df2['gap'],
        mode='lines', line=dict(color='#41C8E8', width=1), name='GAP'
    ))
    df2['gap_smooth'] = df2['gap'].rolling(20, center=True, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df2['gap_smooth'],
        mode='lines', line=dict(color='#C8A84B', width=2), name='Tendance'
    ))
    fig.update_layout(**_layout(
        height=200,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4A6070', size=9)),
        yaxis_title="m/s",
        xaxis_title="km",
    ))
    return fig


def chart_cadence(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    fig = go.Figure()
    # FIX v3.1 : zones optimales recalculées en SPM total
    fig.add_hrect(y0=170, y1=200, fillcolor='rgba(65,200,232,0.05)', line_width=0)
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['cadence'],
        mode='lines', line=dict(color='#41C8E8', width=1.2), name='Cadence'
    ))
    # Ligne de référence à 180 spm (standard running optimal)
    fig.add_hline(y=180, line_dash='dot', line_color='rgba(65,200,232,0.3)', line_width=1)
    fig.update_layout(**_layout(
        height=180,
        yaxis=dict(gridcolor="#152030", zeroline=False),
        yaxis_title="spm",
        xaxis_title="km",
    ))
    return fig


def chart_hr_by_grade(hr_grade_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=hr_grade_df['grade_bin'].astype(str),
        y=hr_grade_df['hr_mean'],
        marker_color='#C84850',
        text=[f"{v:.0f}" for v in hr_grade_df['hr_mean']],
        textposition="outside", textfont=dict(color="#4A6070", size=9),
    ))
    fig.update_layout(**_layout(
        height=220,
        yaxis=dict(gridcolor="#152030", zeroline=False),
        yaxis_title="FC moy (bpm)",
        xaxis_title="Pente",
    ))
    return fig


def chart_ef_quartiles(ef_q: dict) -> go.Figure:
    labels = [k for k, v in ef_q.items() if v is not None]
    values = [v for v in ef_q.values() if v is not None]
    if not values:
        return go.Figure()
    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=['#41C8E8' if i == 0 else '#C84850' if i == len(values)-1 else '#1A5060'
                      for i in range(len(values))],
        text=[f"{v:.3f}" for v in values],
        textposition="outside", textfont=dict(color="#4A6070", size=10),
    ))
    fig.update_layout(**_layout(height=220, yaxis_title="EF"))
    return fig


# ══════════════════════════════════════════════════════════════════
# 5 — PDF GENERATOR (v3.1)
# ══════════════════════════════════════════════════════════════════

def clean(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def generate_pdf(info, fi, flat_v, profile, grade_df,
                 zones, drift, cad_analysis, splits, recs,
                 fcmax, email="") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def bg():
        pdf.set_fill_color(8, 14, 20)
        pdf.rect(0, 0, 210, 297, 'F')

    def sep():
        pdf.set_draw_color(21, 32, 48)
        pdf.set_line_width(0.4)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(4)

    def section(title):
        sep()
        pdf.set_font("Courier", "", 7)
        pdf.set_text_color(42, 64, 80)
        pdf.cell(0, 5, clean(f"-- {title} --"), ln=True)
        pdf.ln(2)

    def kpi(label, value, color=(65, 200, 232)):
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(42, 64, 80)
        pdf.cell(75, 6, clean(label), border=0)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*color)
        pdf.cell(0, 6, clean(value), ln=True)

    bg()

    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(65, 200, 232)
    pdf.cell(0, 14, clean("VERTEX"), ln=True, align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(42, 64, 80)
    pdf.cell(0, 5, clean("PERFORMANCE INTELLIGENCE  |  RACE ANALYSIS v3.1"), ln=True, align="C")
    pdf.set_text_color(100, 130, 150)
    pdf.cell(0, 5, clean(f"{info['name']}  ·  {datetime.now().strftime('%d/%m/%Y')}"), ln=True, align="C")
    pdf.ln(4)

    section("METRIQUES DE COURSE")
    dist_km = info['distance_km']
    total_s = info['total_time_s']
    h, m, s = int(total_s//3600), int((total_s%3600)//60), int(total_s%60)
    kpi("Distance :", f"{dist_km:.1f} km")
    kpi("Temps total :", f"{h}h{m:02d}'{s:02d}\"")
    kpi("D+ :", f"{int(info['elevation_gain'])} m")
    kpi("Allure moyenne :", f"{v_to_pace(info['avg_velocity_ms'])} /km")
    kpi("Allure de base (plat) :", f"{v_to_pace(flat_v)} /km")
    kpi("Altitude max :", f"{int(info['max_elevation'])} m")
    if info.get('hr_mean'):
        kpi("FC moyenne :", f"{int(info['hr_mean'])} bpm")
        kpi("FC max observee :", f"{int(info['hr_max'])} bpm")
    if info.get('cad_mean'):
        kpi("Cadence moyenne :", f"{int(info['cad_mean'])} spm")
    kpi("Classification :", profile)

    if zones:
        section("ZONES DE FREQUENCE CARDIAQUE")
        for z in ['Z1','Z2','Z3','Z4','Z5']:
            bpm = zones['bpm'].get(z, (0,0))
            pct = zones['pct'].get(z, 0)
            t   = zones['time'].get(z, 0)
            mm  = int(t//60)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(42, 64, 80)
            pdf.cell(18, 5, clean(f"{z} :"), border=0)
            pdf.set_text_color(100, 130, 150)
            pdf.cell(45, 5, clean(f"{bpm[0]}-{bpm[1]} bpm"), border=0)
            pdf.set_text_color(65, 200, 232)
            pdf.cell(0, 5, clean(f"{mm} min  ({pct:.0f}%)"), ln=True)

    section("ANALYSE DE FATIGUE GAP")
    dr = fi['decay_ratio']
    dp = fi['decay_pct']
    kpi("Ratio Q4/Q1 (GAP) :", f"{dr:.3f}" if not math.isnan(dr) else "N/A")
    kpi("Perte de vitesse :", f"{dp:.1f}%" if not math.isnan(dp) else "N/A")
    for q, v in fi['quartiles'].items():
        val = f"{v:.3f} m/s  ({v_to_pace(v)} /km)" if not math.isnan(v) else "N/A"
        pdf.set_font("Helvetica", "", 8); pdf.set_text_color(42, 64, 80)
        pdf.cell(30, 5, clean(q + " :"), border=0)
        pdf.set_text_color(100, 130, 150)
        pdf.cell(0, 5, clean(val), ln=True)

    if drift.get('drift_pct') is not None:
        section("DECOUPLAGE CARDIAQUE")
        kpi("EF 1ere moitie (plat) :", f"{drift['ef1']:.3f}" if drift['ef1'] else "N/A")
        kpi("EF 2eme moitie (plat) :", f"{drift['ef2']:.3f}" if drift['ef2'] else "N/A")
        drift_val = drift['drift_pct']
        color = (200, 72, 80) if drift_val < -5 else (200, 168, 75) if drift_val < -2 else (65, 200, 232)
        kpi("Derive EF :", f"{drift_val:.1f}%", color=color)

    section("PROFIL PENTE")
    for _, row in grade_df.iterrows():
        pdf.set_font("Helvetica", "", 8); pdf.set_text_color(42, 64, 80)
        pdf.cell(40, 5, clean(str(row['Tranche pente']) + " :"), border=0)
        pdf.set_text_color(100, 130, 150)
        pdf.cell(0, 5, clean(f"{row['Allure (min/km)']} /km"), ln=True)

    if cad_analysis.get('mean'):
        section("ANALYSE CADENCE")
        kpi("Cadence moyenne :", f"{cad_analysis['mean']:.0f} spm")
        kpi("Zone optimale (170-200spm) :", f"{cad_analysis['optimal_pct']:.0f}% du temps")
        for k, v in cad_analysis['dist'].items():
            if v > 1:
                pdf.set_font("Helvetica", "", 8); pdf.set_text_color(42, 64, 80)
                pdf.cell(40, 5, clean(k + " spm :"), border=0)
                pdf.set_text_color(100, 130, 150)
                pdf.cell(0, 5, clean(f"{v:.0f}%"), ln=True)

    if splits:
        section("SPLITS PAR KM (resume)")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(42, 64, 80)
        cols = ["Km", "Allure", "GAP", "D+", "FC", "Cad"]
        widths = [12, 22, 22, 15, 18, 18]
        for col, w in zip(cols, widths):
            pdf.cell(w, 5, clean(col), border=0)
        pdf.ln()
        pdf.set_font("Helvetica", "", 7)
        for sp in splits[::2]:
            pdf.set_text_color(100, 130, 150)
            pdf.cell(12, 4, clean(str(sp['km'])), border=0)
            pdf.cell(22, 4, clean(sp['pace']), border=0)
            pdf.cell(22, 4, clean(sp['gap']), border=0)
            pdf.cell(15, 4, clean(f"+{sp['d_pos']}m"), border=0)
            pdf.cell(18, 4, clean(str(sp['hr']) if sp['hr'] else "--"), border=0)
            pdf.cell(18, 4, clean(str(sp['cadence']) if sp['cadence'] else "--"), border=0)
            pdf.ln()

    section("RECOMMANDATIONS COACH")
    level_colors = {'info': (65,200,232), 'warn': (200,168,75), 'crit': (200,72,80)}
    for i, rec in enumerate(recs, 1):
        color = level_colors.get(rec['level'], (100,130,150))
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*color)
        pdf.cell(8, 5, clean(f"{i}."), border=0)
        pdf.cell(0, 5, clean(rec['title']), ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 130, 150)
        pdf.multi_cell(0, 5, clean(rec['body']))
        pdf.ln(1)

    sep()
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(30, 50, 60)
    if email:
        pdf.cell(0, 4, clean(f"Plans envoyes a : {email}"), ln=True, align="C")
    pdf.cell(0, 4,
        clean(f"VERTEX v3.2 — GAP Minetti (2002) — FCmax: {fcmax} bpm — {datetime.now().strftime('%d/%m/%Y')}"),
        ln=True, align="C")

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════
# 6 — UI
# ══════════════════════════════════════════════════════════════════

def render_landing():
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="hud-label">// SYSTEM ONLINE — v3.2 //</div>', unsafe_allow_html=True)
        st.markdown('<div class="vertex-title">VERTEX</div>', unsafe_allow_html=True)
        st.markdown('<div class="vertex-sub">PERFORMANCE INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#2A4050;line-height:2;">
        > ENGINE &nbsp;: GAP Analysis · Minetti 2002<br>
        > SENSORS : FC · Cadence · Altitude baro<br>
        > OUTPUT &nbsp;: Profil fatigue · Zones · Découplage · PDF coach
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── PROFIL ATHLÈTE ──────────────────────────────────────
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#2A4050;
        letter-spacing:0.22em;text-transform:uppercase;border-bottom:1px solid #152030;
        padding-bottom:6px;margin-bottom:1rem;">// PROFIL ATHLÈTE //</div>
        """, unsafe_allow_html=True)

        col_fc, col_mode = st.columns([1, 1])
        with col_fc:
            fcmax_input = st.number_input(
                "FCmax (bpm)",
                min_value=150, max_value=220,
                value=st.session_state.get('fcmax', 190),
                step=1,
                help="Ta fréquence cardiaque maximale réelle",
            )
            st.session_state['fcmax'] = fcmax_input

        with col_mode:
            zone_mode = st.selectbox(
                "Mode zones FC",
                options=["Auto (% FCmax)", "Manuel (je connais mes zones)"],
                index=0 if st.session_state.get('zone_mode', 'auto') == 'auto' else 1,
            )
            st.session_state['zone_mode'] = 'auto' if zone_mode.startswith('Auto') else 'manual'

        if st.session_state['zone_mode'] == 'manual':
            st.markdown("""
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;
            letter-spacing:0.15em;margin:8px 0 6px;">SEUILS DE ZONES (bpm)</div>
            """, unsafe_allow_html=True)
            zc1, zc2, zc3, zc4, zc5 = st.columns(5)
            default_zones = st.session_state.get('custom_zones', {
                'Z1': (0,   int(fcmax_input * 0.60)),
                'Z2': (int(fcmax_input * 0.60), int(fcmax_input * 0.70)),
                'Z3': (int(fcmax_input * 0.70), int(fcmax_input * 0.80)),
                'Z4': (int(fcmax_input * 0.80), int(fcmax_input * 0.90)),
                'Z5': (int(fcmax_input * 0.90), fcmax_input),
            })
            z1_hi = zc1.number_input("Z1 max", min_value=80,  max_value=220, value=default_zones['Z1'][1], step=1)
            z2_hi = zc2.number_input("Z2 max", min_value=80,  max_value=220, value=default_zones['Z2'][1], step=1)
            z3_hi = zc3.number_input("Z3 max", min_value=80,  max_value=220, value=default_zones['Z3'][1], step=1)
            z4_hi = zc4.number_input("Z4 max", min_value=80,  max_value=220, value=default_zones['Z4'][1], step=1)
            z5_hi = zc5.number_input("Z5 max", min_value=80,  max_value=220, value=default_zones['Z5'][1], step=1)
            st.session_state['custom_zones'] = {
                'Z1': (0,    z1_hi),
                'Z2': (z1_hi, z2_hi),
                'Z3': (z2_hi, z3_hi),
                'Z4': (z3_hi, z4_hi),
                'Z5': (z4_hi, z5_hi),
            }
            st.markdown("""
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;margin-top:4px;">
            Saisis le bpm maximum de chaque zone
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "IMPORTER UN FICHIER GPX",
            type=["gpx"],
            help="Garmin Connect → Exporter l'original (avec FC et cadence)",
            label_visibility="visible",
        )
        if uploaded:
            st.session_state['gpx_bytes']    = uploaded.read()
            st.session_state['gpx_filename'] = uploaded.name
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, desc in [
        (c1, "GAP ENGINE",     "Grade-Adjusted Pace · Minetti 2002 · Profil fatigue 4 quartiles"),
        (c2, "ZONES FC",       "Distribution Z1→Z5 · Découplage cardiaque · EF par segment"),
        (c3, "CADENCE",        "Distribution · Régularité · Zones optimales · Évolution"),
        (c4, "COACH REPORT",   "6 recommandations personnalisées · Export PDF complet"),
    ]:
        with col:
            st.markdown(f"""
            <div style="border:1px solid #152030;padding:1.2rem;border-top:2px solid rgba(65,200,232,0.3);">
            <div class="hud-label" style="color:#41C8E8">{label}</div>
            <div style="color:#3A5060;font-size:0.82rem;margin-top:6px;font-family:'DM Sans',sans-serif">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1A2A35;text-align:center;letter-spacing:0.1em;">
    EXPORT GPX AVEC FC : Garmin Connect → Activité → ··· → Exporter l'original &nbsp;|&nbsp;
    Strava → utiliser Garmin Connect directement (Strava supprime la FC à l'export GPX)
    </div>
    """, unsafe_allow_html=True)


def render_dashboard(gpx_bytes: bytes, filename: str):
    with st.spinner("Analyse du fichier GPX..."):
        try:
            df = parse_gpx(gpx_bytes)
        except ValueError as e:
            st.error(f"Erreur de lecture GPX : {e}")
            if st.button("↺ Recommencer"):
                for k in ['gpx_bytes','gpx_filename']:
                    st.session_state.pop(k, None)
                st.rerun()
            return

    info     = extract_race_info(df, filename)
    fi       = fatigue_index(df)
    flat_v   = flat_pace_estimate(df)
    grade_df = grade_pace_profile(df)
    profile  = classify_profile(fi['decay_ratio'], flat_v)
    splits   = compute_km_splits(df)
    cad_an   = cadence_analysis(df)

    # Récupération FCmax et zones depuis session_state (saisis sur la landing)
    fcmax = int(st.session_state.get('fcmax', int(info['hr_max']) if info.get('hr_max') else 190))
    zone_mode = st.session_state.get('zone_mode', 'auto')
    custom_zones = st.session_state.get('custom_zones', None)

    with st.sidebar:
        st.markdown('<div class="hud-label">// PARAMETRES //</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#41C8E8;margin-bottom:4px;">
        FCmax : {fcmax} bpm
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;">
        Zones : {'Manuel' if zone_mode == 'manual' else '% FCmax auto'}
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ NOUVELLE ANALYSE"):
            for k in ['gpx_bytes', 'gpx_filename', 'fcmax', 'zone_mode', 'custom_zones']:
                st.session_state.pop(k, None)
            st.rerun()

    zones    = compute_hr_zones(df, fcmax, custom_zones if zone_mode == 'manual' else None) if info['has_hr'] else None
    drift    = cardiac_drift(df)           if info['has_hr'] else {'ef1':None,'ef2':None,'drift_pct':None,'quartiles':{}}
    hr_grade = hr_by_grade(df)             if info['has_hr'] else None
    recs     = generate_coach_recommendations(profile, fi, drift, cad_an, info, fcmax)

    # ── Header ──────────────────────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown(
            f'<div class="hud-label">// ANALYSE COMPLETE — v3.2 //</div>'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:1.8rem;'
            f'font-weight:700;letter-spacing:0.15em;color:#ffffff">'
            f'{info["name"].upper()}</div>',
            unsafe_allow_html=True,
        )
    with col_badge:
        badge_class = {
            "PROFIL ENDURANCE": "badge-endurance",
            "PROFIL EXPLOSIF":  "badge-explosif",
            "PROFIL FRAGILE":   "badge-fragile",
        }.get(profile, "badge-fragile")
        st.markdown(f'<br><span class="{badge_class}">{profile}</span>', unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#152030;margin:0.5rem 0 1.5rem;">', unsafe_allow_html=True)

    # ── KPIs ────────────────────────────────────────────────────
    total_s = info['total_time_s']
    h_t = int(total_s//3600); m_t = int((total_s%3600)//60)

    cols_kpi = st.columns(6) if info['has_hr'] else st.columns(4)
    cols_kpi[0].metric("DISTANCE", f"{info['distance_km']:.1f} km")
    cols_kpi[1].metric("TEMPS", f"{h_t}h{m_t:02d}'")
    cols_kpi[2].metric("D+", f"{int(info['elevation_gain'])} m")
    cols_kpi[3].metric("ALLURE MOY.", v_to_pace(info['avg_velocity_ms']) + "/km")
    if info['has_hr']:
        cols_kpi[4].metric("FC MOYENNE", f"{int(info['hr_mean'])} bpm")
        cols_kpi[5].metric("FC MAX", f"{int(info['hr_max'])} bpm")

    if info['has_cad']:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CADENCE MOY.", f"{int(info['cad_mean'])} spm")
        dr = fi['decay_ratio']
        dp = fi['decay_pct']
        c2.metric("RATIO Q4/Q1", f"{dr:.3f}" if not math.isnan(dr) else "N/A")
        c3.metric("PERTE GAP", f"{dp:.1f}%" if not math.isnan(dp) else "N/A")
        if drift.get('drift_pct') is not None:
            c4.metric("DÉRIVE EF", f"{drift['drift_pct']:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══ SECTION 1 : PROFIL DE COURSE ════════════════════════════
    st.markdown('<div class="section-title">PROFIL DE COURSE</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="hud-label" style="margin-bottom:4px;">Profil altimétrique</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_elevation(df), use_container_width=True)
    with p2:
        st.markdown('<div class="hud-label" style="margin-bottom:4px;">Allure au fil des km</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_pace(df), use_container_width=True)

    # ══ SECTION 2 : FC ══════════════════════════════════════════
    if info['has_hr']:
        st.markdown('<div class="section-title">FRÉQUENCE CARDIAQUE</div>', unsafe_allow_html=True)

        h1, h2 = st.columns(2)
        with h1:
            st.markdown('<div class="hud-label" style="margin-bottom:4px;">FC sur la course</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_hr(df, fcmax), use_container_width=True)
        with h2:
            st.markdown('<div class="hud-label" style="margin-bottom:4px;">FC × Allure superposées</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_hr_pace_overlay(df), use_container_width=True)

        if zones:
            zone_mode_label = '· ZONES MANUELLES' if zone_mode == 'manual' else f'· % FCMAX AUTO ({fcmax} bpm)'
            st.markdown(f'<div class="hud-label" style="margin-top:1rem;margin-bottom:12px;">Distribution des zones FC <span style="color:#41C8E8">{zone_mode_label}</span></div>', unsafe_allow_html=True)
            zone_cols = st.columns(5)
            zone_labels = ['Z1','Z2','Z3','Z4','Z5']
            zone_colors_hex = ['#1A3A4A','#1A5060','#1A8AAA','#C8A84B','#C84850']
            for i, z in enumerate(zone_labels):
                with zone_cols[i]:
                    pct = zones['pct'][z]
                    t   = zones['time'][z]
                    mm  = int(t//60)
                    bpm = zones['bpm'][z]
                    st.markdown(f"""
                    <div style="background:#0D1520;border:1px solid #152030;border-top:2px solid {zone_colors_hex[i]};padding:12px;text-align:center;">
                        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;letter-spacing:0.2em">{z}</div>
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:700;color:{zone_colors_hex[i]}">{pct:.0f}%</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3A5060">{mm} min</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;margin-top:4px">{bpm[0]}–{bpm[1]} bpm</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;color:#2A4050;margin-top:4px">{ZONE_NAMES[z]}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if hr_grade is not None and len(hr_grade) > 3:
            hz1, hz2 = st.columns(2)
            with hz1:
                st.markdown('<div class="hud-label" style="margin-bottom:4px;">FC moyenne par pente</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_hr_by_grade(hr_grade), use_container_width=True)
            with hz2:
                st.markdown('<div class="hud-label" style="margin-bottom:4px;">Découplage cardiaque (EF par quartile)</div>', unsafe_allow_html=True)
                if drift['quartiles']:
                    st.plotly_chart(chart_ef_quartiles(drift['quartiles']), use_container_width=True)
                if drift.get('drift_pct') is not None:
                    color = '#C84850' if drift['drift_pct'] < -5 else '#C8A84B' if drift['drift_pct'] < -2 else '#41C8E8'
                    st.markdown(f"""
                    <div style="background:#0D1520;border:1px solid #152030;padding:12px;margin-top:8px;">
                        <div class="hud-label">Dérive EF (plat)</div>
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:700;color:{color}">
                            {drift['drift_pct']:.1f}%
                        </div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3A5060">
                            EF1={drift['ef1']:.3f} → EF2={drift['ef2']:.3f}
                        </div>
                    </div>""", unsafe_allow_html=True)

    # ══ SECTION 3 : CADENCE ═════════════════════════════════════
    if info['has_cad']:
        st.markdown('<div class="section-title">CADENCE</div>', unsafe_allow_html=True)
        cd1, cd2 = st.columns([2, 1])
        with cd1:
            st.markdown('<div class="hud-label" style="margin-bottom:4px;">Évolution cadence (spm)</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_cadence(df), use_container_width=True)
        with cd2:
            st.markdown('<div class="hud-label" style="margin-bottom:12px;">Distribution</div>', unsafe_allow_html=True)
            optimal_color = '#41C8E8' if cad_an['optimal_pct'] and cad_an['optimal_pct'] > 65 else '#C8A84B'
            st.markdown(f"""
            <div style="background:#0D1520;border:1px solid #152030;padding:16px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#2A4050;letter-spacing:0.2em">ZONE OPTIMALE (170-200spm)</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.5rem;font-weight:700;color:{optimal_color}">
                    {cad_an['optimal_pct']:.0f}%
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3A5060">du temps de course</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for k, v in cad_an['dist'].items():
                if v > 0.5:
                    bar_w = int(v * 1.4)
                    is_opt = k in ['170-180', '180-190', '190-200']
                    color = '#41C8E8' if is_opt else '#1A3A4A'
                    st.markdown(f"""
                    <div style="margin-bottom:5px;">
                        <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:0.6rem;color:#3A5060;margin-bottom:2px;">
                            <span>{k} spm</span><span>{v:.0f}%</span>
                        </div>
                        <div style="background:{color};height:4px;width:{min(bar_w,100)}%;border-radius:1px;"></div>
                    </div>""", unsafe_allow_html=True)

    # ══ SECTION 4 : GAP ═════════════════════════════════════════
    st.markdown('<div class="section-title">ANALYSE DE FATIGUE GAP</div>', unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown('<div class="hud-label" style="margin-bottom:4px;">Vitesse GAP par quartile</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_quartiles(fi['quartiles']), use_container_width=True)
    with g2:
        st.markdown('<div class="hud-label" style="margin-bottom:4px;">Courbe GAP sur la course</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_gap_profile(df), use_container_width=True)
    with g3:
        st.markdown('<div class="hud-label" style="margin-bottom:4px;">Distribution par pente</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_grade_dist(df), use_container_width=True)

    st.markdown('<div class="section-title">ALLURE PAR TRANCHE DE PENTE</div>', unsafe_allow_html=True)
    st.dataframe(
        grade_df.style.set_properties(**{
            "background-color": "#0D1520",
            "color": "#C8D4DC",
            "border": "1px solid #152030",
        }),
        use_container_width=True, hide_index=True,
    )

    # ══ SECTION 5 : SPLITS PAR KM ═══════════════════════════════
    st.markdown('<div class="section-title">SPLITS PAR KM</div>', unsafe_allow_html=True)
    if splits:
        has_hr_sp  = any(s['hr'] for s in splits)
        has_cad_sp = any(s['cadence'] for s in splits)

        headers = ['KM', 'ALLURE', 'GAP', 'D+', 'D-']
        if has_hr_sp:  headers.append('FC')
        if has_cad_sp: headers.append('CAD')

        header_row = ''.join(f'<th>{h}</th>' for h in headers)
        rows_html = ''
        valid_paces = [s['pace_s'] for s in splits if s.get('pace_s')]
        med_pace = sum(valid_paces) / len(valid_paces) if valid_paces else None
        for sp in splits:
            pace_s = sp.get('pace_s')
            color = '#C8D4DC'
            if pace_s and med_pace:
                if pace_s < med_pace * 0.92:
                    color = '#41C8E8'
                elif pace_s > med_pace * 1.08:
                    color = '#C84850'

            row = f'<td>{sp["km"]}</td>'
            row += f'<td style="color:{color}">{sp["pace"]}</td>'
            row += f'<td>{sp["gap"]}</td>'
            row += f'<td style="color:#41C8E8">+{sp["d_pos"]}m</td>'
            row += f'<td style="color:#C84850">-{sp["d_neg"]}m</td>'
            if has_hr_sp:
                hr_color = '#C84850' if sp['hr'] and sp['hr'] > fcmax * 0.92 else '#C8D4DC'
                row += f'<td style="color:{hr_color}">{sp["hr"] or "--"}</td>'
            if has_cad_sp:
                # FIX v3.1 : zone optimale recalculée en SPM total
                cad_color = '#41C8E8' if sp['cadence'] and 170 <= sp['cadence'] <= 200 else '#C8D4DC'
                row += f'<td style="color:{cad_color}">{sp["cadence"] or "--"}</td>'
            rows_html += f'<tr>{row}</tr>'

        st.markdown(f"""
        <div style="overflow-x:auto;">
        <table class="km-table">
            <thead><tr>{header_row}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>""", unsafe_allow_html=True)

    # ══ SECTION 6 : RECOMMANDATIONS COACH ═══════════════════════
    st.markdown('<div class="section-title">RECOMMANDATIONS COACH</div>', unsafe_allow_html=True)
    level_styles = {
        'info': ('rgba(65,200,232,0.3)',  '#41C8E8', '◆ INFO'),
        'warn': ('rgba(200,168,75,0.6)',  '#C8A84B', '▲ ATTENTION'),
        'crit': ('rgba(200,72,80,0.6)',   '#C84850', '● PRIORITAIRE'),
    }
    for rec in recs:
        border_color, text_color, label = level_styles.get(rec['level'], level_styles['info'])
        st.markdown(f"""
        <div style="padding:14px 18px;background:#0D1520;border-left:3px solid {border_color};margin-bottom:10px;">
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:{text_color};letter-spacing:0.2em;margin-bottom:6px">{label}</div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;color:#ffffff;margin-bottom:6px">{rec['title']}</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#4A6070;line-height:1.6">{rec['body']}</div>
        </div>""", unsafe_allow_html=True)

    # ══ SECTION 7 : PDF ═════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">FEUILLE DE ROUTE TACTIQUE</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns([3, 1])
    with lc1:
        email = st.text_input(
            "Recevoir mes futurs plans par email (optionnel)",
            placeholder="ton@email.com",
            key="email_input",
        )
    with lc2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▲  GENERER LE PDF"):
            with st.spinner("Génération du rapport..."):
                pdf_bytes = generate_pdf(
                    info, fi, flat_v, profile, grade_df,
                    zones, drift, cad_an, splits, recs, fcmax,
                    st.session_state.get("email_input", "")
                )
            fname = f"VERTEX_{info['name'].replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
            st.download_button(
                "⬇  TELECHARGER LE RAPPORT", data=pdf_bytes,
                file_name=fname, mime="application/pdf",
            )


# ══════════════════════════════════════════════════════════════════
# 7 — MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    if 'gpx_bytes' not in st.session_state:
        render_landing()
    else:
        render_dashboard(
            st.session_state['gpx_bytes'],
            st.session_state.get('gpx_filename', 'course.gpx'),
        )

if __name__ == "__main__":
    main()
