"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — GPX Performance Analyzer  |  app.py            ║
║         No API · No OAuth · Pure GPX Analysis · v2.0            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import io
import math
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
.block-container { padding-top: 2rem !important; max-width: 1140px; }

/* Grid background */
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

/* Metrics */
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

/* Buttons */
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

/* Download button */
.stDownloadButton > button {
    background: rgba(65,200,232,0.08) !important;
    border: 1px solid #41C8E8 !important;
    color: #41C8E8 !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    border-radius: 2px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #0D1520 !important;
    border: 1px dashed #152030 !important;
    border-radius: 2px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #41C8E8 !important;
}

/* Progress */
.stProgress > div > div > div { background: #41C8E8 !important; }

/* Divider */
hr { border-color: #152030 !important; }

/* Spinner */
.stSpinner > div { border-top-color: #41C8E8 !important; }

/* Text input */
.stTextInput > div > div > input {
    background: #0D1520 !important;
    border: 1px solid #152030 !important;
    color: #C8D4DC !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: #41C8E8 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #0D1520 !important;
    border: 1px solid #152030 !important;
    border-radius: 2px !important;
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
.stat-row {
    display:flex; gap:8px; align-items:baseline;
    border-left:2px solid #41C8E830; padding-left:12px;
    margin-bottom:8px;
}
.stat-val {
    font-family:'Barlow Condensed',sans-serif;
    font-size:1.5rem; font-weight:700; color:#41C8E8;
}
.stat-unit {
    font-family:'DM Mono',monospace;
    font-size:0.65rem; color:#3A5060; letter-spacing:0.1em;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 1 — GPX PARSER
# ══════════════════════════════════════════════════════════════════

NS = {
    'gpx':  'http://www.topografix.com/GPX/1/1',
    'gpx10':'http://www.topografix.com/GPX/1/0',
    'gpx11':'http://www.topografix.com/GPX/1/1',
}

def haversine(lat1, lon1, lat2, lon2) -> float:
    """Distance en mètres entre deux points GPS."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def parse_gpx(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse un fichier GPX et retourne un DataFrame avec :
    time, lat, lon, elevation, distance_cum, velocity, grade
    """
    try:
        root = ET.fromstring(file_bytes)
    except ET.ParseError as e:
        raise ValueError(f"Fichier GPX invalide : {e}")

    # Detect namespace
    tag = root.tag
    if 'gpx' in tag.lower():
        ns_uri = tag[1:tag.index('}')]
        ns = {'g': ns_uri}
    else:
        ns = {'g': 'http://www.topografix.com/GPX/1/1'}

    # Collect trackpoints
    trkpts = root.findall('.//g:trkpt', ns)
    if not trkpts:
        trkpts = root.findall('.//trkpt')  # fallback no namespace
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
                ts = time_el.text.replace('Z','').replace('z','')
                if '.' in ts:
                    t = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S.%f')
                else:
                    t = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%S')
            except Exception:
                pass
        rows.append({'lat': lat, 'lon': lon, 'elevation': ele, 'time': t})

    df = pd.DataFrame(rows)

    # Compute cumulative distance
    dist_cum = [0.0]
    for i in range(1, len(df)):
        d = haversine(df.loc[i-1,'lat'], df.loc[i-1,'lon'],
                      df.loc[i,'lat'],   df.loc[i,'lon'])
        dist_cum.append(dist_cum[-1] + d)
    df['distance'] = dist_cum

    # Compute time in seconds
    if df['time'].notna().sum() > len(df) * 0.5:
        t0 = df['time'].iloc[0]
        df['time_s'] = df['time'].apply(
            lambda t: (t - t0).total_seconds() if pd.notna(t) else None
        )
        df['time_s'] = df['time_s'].interpolate()
    else:
        # Estimate time from distance at 10 km/h
        df['time_s'] = df['distance'] / (10000/3600)

    # Compute velocity (m/s) — smooth over 5 points
    df['dt'] = df['time_s'].diff().fillna(1).clip(lower=0.1)
    df['dd'] = df['distance'].diff().fillna(0)
    df['velocity_raw'] = (df['dd'] / df['dt']).clip(0, 12)
    df['velocity'] = df['velocity_raw'].rolling(7, center=True, min_periods=1).mean()

    # Compute grade (%)
    df['dz'] = df['elevation'].diff().fillna(0)
    df['grade'] = (df['dz'] / df['dd'].replace(0, float('nan')) * 100).fillna(0).clip(-40, 40)
    df['grade'] = df['grade'].rolling(5, center=True, min_periods=1).mean()

    return df.reset_index(drop=True)


def extract_race_info(df: pd.DataFrame, filename: str) -> dict:
    """Extrait les infos de course depuis le DataFrame."""
    total_dist = df['distance'].iloc[-1]
    total_time = df['time_s'].iloc[-1]
    elevation_gain = df['dz'].clip(lower=0).sum()
    elevation_loss = abs(df['dz'].clip(upper=0).sum())
    max_elevation = df['elevation'].max()
    min_elevation = df['elevation'].min()
    avg_velocity = df[df['velocity'] > 0.3]['velocity'].mean()

    return {
        'name': filename.replace('.gpx','').replace('_',' ').title(),
        'distance_km': total_dist / 1000,
        'total_time_s': total_time,
        'elevation_gain': elevation_gain,
        'elevation_loss': elevation_loss,
        'max_elevation': max_elevation,
        'min_elevation': min_elevation,
        'avg_velocity_ms': avg_velocity,
    }


# ══════════════════════════════════════════════════════════════════
# 2 — ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════

def gap_correction(velocity_ms: float, grade_pct: float) -> float:
    """Grade-Adjusted Pace — modèle Minetti (2002)."""
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

ADVICE = {
    "PROFIL ENDURANCE": [
        "Ton moteur aerobique est solide. Integre 2x/semaine des blocs de cote specifiques (6-8 min a 90% VMA sur 8-12%) pour gagner en puissance verticale.",
        "Tes sorties longues peuvent monter en D+. Vise +200m de D+ supplementaire par semaine jusqu'a la prochaine competition.",
        "Travaille la cadence en descente : 180 pas/min minimum. C'est la principale source de gain de temps sur un profil Endurance.",
    ],
    "PROFIL EXPLOSIF": [
        "Bonne vitesse de base mais erosion au long cours detectee. Priorise les sorties > 2h30 en Zone 2 strict (< 75% FCmax) pour consolider la base.",
        "Ajoute des blocs de montee prolongee (20-30 min continus) pour habituer ton metabolisme a la production d'energie soutenue.",
        "Nutrition : prise de glucides toutes les 30-40 min apres 1h d'effort. Le decrochage GAP est souvent d'origine metabolique.",
    ],
    "PROFIL FRAGILE": [
        "Decrochage significatif detecte. Stop au travail de vitesse — 4 semaines de volume Z2 pur (60-70% FCmax) sont prioritaires.",
        "Analyse ta nutrition a l'effort : le ratio Q4/Q1 suggere un deficit glucidique ou une mauvaise gestion de l'effort en debut de course.",
        "Apres 3 semaines de Z2, reintegre 1 seance de cotes courtes (30s x 10 repetitions) pour retrouver la puissance sans fatigue systemique.",
    ],
    "PROFIL INCONNU": [
        "Donnees insuffisantes pour une classification precise.",
        "Verifie que ton fichier GPX contient des donnees de temps et d'altitude completes.",
        "Relance l'analyse avec un fichier GPX plus complet.",
    ],
}


# ══════════════════════════════════════════════════════════════════
# 3 — CHARTS
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

def chart_elevation(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['elevation'],
        mode='lines', line=dict(color='#41C8E8', width=1.5),
        fill='tozeroy', fillcolor='rgba(65,200,232,0.06)',
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=180,
        yaxis_title="m", xaxis_title="km")
    return fig

def chart_pace(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    pace = df['velocity'].apply(lambda v: 1000/v/60 if v > 0.3 else None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=pace,
        mode='lines', line=dict(color='#C8A84B', width=1.5),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=180,
        yaxis=dict(gridcolor="#152030", zeroline=False, autorange='reversed'),
        yaxis_title="min/km", xaxis_title="km")
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
    fig.update_layout(**PLOTLY_LAYOUT, height=240)
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
    fig.update_layout(**PLOTLY_LAYOUT, height=240)
    return fig

def chart_gap_profile(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2['gap'] = df2.apply(lambda r: gap_correction(r['velocity'], r['grade']), axis=1)
    dist_km = df2['distance'] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df2['gap'],
        mode='lines', line=dict(color='#41C8E8', width=1),
        name='GAP'
    ))
    # Rolling mean
    df2['gap_smooth'] = df2['gap'].rolling(20, center=True, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df2['gap_smooth'],
        mode='lines', line=dict(color='#C8A84B', width=2),
        name='Tendance'
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200,
        showlegend=True,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4A6070', size=9)),
        yaxis_title="m/s", xaxis_title="km")
    return fig


# ══════════════════════════════════════════════════════════════════
# 4 — PDF GENERATOR
# ══════════════════════════════════════════════════════════════════

def clean(text: str) -> str:
    import unicodedata
    text = unicodedata.normalize("NFKD", str(text))
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def generate_pdf(info: dict, fi: dict, flat_v: float, profile: str,
                 grade_profile_df: pd.DataFrame, email: str = "") -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Dark background
    pdf.set_fill_color(8, 14, 20)
    pdf.rect(0, 0, 210, 297, 'F')

    # Header
    pdf.set_font("Helvetica", "B", 32)
    pdf.set_text_color(65, 200, 232)
    pdf.cell(0, 14, clean("VERTEX"), ln=True, align="C")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(42, 64, 80)
    pdf.cell(0, 5, clean("PERFORMANCE INTELLIGENCE  |  RACE ANALYSIS"), ln=True, align="C")
    pdf.set_text_color(100, 130, 150)
    pdf.cell(0, 5, clean(f"{info['name']}  ·  {datetime.now().strftime('%d/%m/%Y')}"), ln=True, align="C")
    pdf.ln(4)

    # Separator
    pdf.set_draw_color(21, 32, 48)
    pdf.set_line_width(0.4)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(5)

    # KPIs
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(42, 64, 80)
    pdf.cell(0, 5, clean("-- METRIQUES DE COURSE --"), ln=True)
    pdf.ln(2)

    def kpi(label, value):
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(42, 64, 80)
        pdf.cell(65, 6, clean(label), border=0)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(65, 200, 232)
        pdf.cell(0, 6, clean(value), ln=True)

    dist_km = info['distance_km']
    total_s = info['total_time_s']
    h, m, s = int(total_s//3600), int((total_s%3600)//60), int(total_s%60)
    avg_pace = v_to_pace(info['avg_velocity_ms'])

    kpi("Distance :", f"{dist_km:.1f} km")
    kpi("Temps total :", f"{h}h{m:02d}'{s:02d}\"")
    kpi("D+ :", f"{int(info['elevation_gain'])} m")
    kpi("Allure moyenne :", f"{avg_pace} /km")
    kpi("Altitude max :", f"{int(info['max_elevation'])} m")
    kpi("Allure de base (plat) :", f"{v_to_pace(flat_v)} /km")
    pdf.ln(3)

    # GAP Analysis
    pdf.set_draw_color(21, 32, 48)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(42, 64, 80)
    pdf.cell(0, 5, clean("-- ANALYSE DE FATIGUE GAP --"), ln=True)
    pdf.ln(2)

    dr = fi['decay_ratio']
    dp = fi['decay_pct']
    kpi("Ratio Q4/Q1 (GAP) :", f"{dr:.3f}" if not math.isnan(dr) else "N/A")
    kpi("Perte de vitesse :", f"{dp:.1f}%" if not math.isnan(dp) else "N/A")
    kpi("Classification :", profile)
    pdf.ln(3)

    # Quartiles table
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(65, 200, 232)
    for q, v in fi['quartiles'].items():
        val = f"{v:.3f} m/s  ({v_to_pace(v)} /km)" if not math.isnan(v) else "N/A"
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(42, 64, 80)
        pdf.cell(30, 5, clean(q + " :"), border=0)
        pdf.set_text_color(100, 130, 150)
        pdf.cell(0, 5, clean(val), ln=True)
    pdf.ln(3)

    # Grade profile
    pdf.set_draw_color(21, 32, 48)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(42, 64, 80)
    pdf.cell(0, 5, clean("-- PROFIL PENTE --"), ln=True)
    pdf.ln(2)
    for _, row in grade_profile_df.iterrows():
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(42, 64, 80)
        pdf.cell(40, 5, clean(str(row['Tranche pente']) + " :"), border=0)
        pdf.set_text_color(100, 130, 150)
        pdf.cell(0, 5, clean(f"{row['Allure (min/km)']} /km"), ln=True)
    pdf.ln(3)

    # Advice
    pdf.set_draw_color(21, 32, 48)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Courier", "", 7)
    pdf.set_text_color(42, 64, 80)
    pdf.cell(0, 5, clean("-- 3 RECOMMANDATIONS STRATEGIQUES --"), ln=True)
    pdf.ln(2)
    advices = ADVICE.get(profile, ADVICE["PROFIL INCONNU"])
    for i, adv in enumerate(advices, 1):
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(65, 200, 232)
        pdf.cell(8, 5, clean(f"{i}."), border=0)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 130, 150)
        pdf.multi_cell(0, 5, clean(adv))
        pdf.ln(1)
    pdf.ln(3)

    # Footer
    pdf.set_draw_color(21, 32, 48)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(30, 50, 60)
    if email:
        pdf.cell(0, 4, clean(f"Plans envoyes a : {email}"), ln=True, align="C")
    pdf.cell(0, 4, clean("VERTEX Performance Intelligence - Powered by GAP Minetti (2002)"), ln=True, align="C")

    return bytes(pdf.output())


# ══════════════════════════════════════════════════════════════════
# 5 — UI
# ══════════════════════════════════════════════════════════════════

def render_landing():
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="hud-label">// SYSTEM ONLINE //</div>', unsafe_allow_html=True)
        st.markdown('<div class="vertex-title">VERTEX</div>', unsafe_allow_html=True)
        st.markdown('<div class="vertex-sub">PERFORMANCE INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.72rem;color:#2A4050;line-height:2;">
        > ENGINE : GAP Analysis · Minetti 2002<br>
        > INPUT &nbsp;: GPX File (Garmin / Suunto / Polar / Apple)<br>
        > OUTPUT : Tactical Fatigue Profile + PDF
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "IMPORTER UN FICHIER GPX",
            type=["gpx"],
            help="Exporte ton activité depuis Garmin Connect, Suunto, Strava ou toute montre GPS",
            label_visibility="visible",
        )

        if uploaded:
            st.session_state['gpx_bytes']    = uploaded.read()
            st.session_state['gpx_filename'] = uploaded.name
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, label, desc in [
        (c1, "UNIVERSEL", "Compatible Garmin, Suunto, Polar, Apple Watch, Coros, Wahoo"),
        (c2, "GAP ENGINE", "Grade-Adjusted Pace · Modele Minetti 2002 · Profil de fatigue"),
        (c3, "RAPPORT PDF", "Feuille de route tactique avec 3 recommandations personnalisees"),
    ]:
        with col:
            st.markdown(f"""
            <div style="border:1px solid #152030;padding:1.2rem;border-top:2px solid rgba(65,200,232,0.3);">
            <div class="hud-label" style="color:#41C8E8">{label}</div>
            <div style="color:#3A5060;font-size:0.85rem;margin-top:6px;font-family:'DM Sans',sans-serif">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1A2A35;text-align:center;letter-spacing:0.1em;">
    COMMENT EXPORTER TON GPX : Garmin Connect → Activité → ··· → Exporter vers GPX &nbsp;|&nbsp;
    Strava → Activité → ··· → Exporter le fichier GPX &nbsp;|&nbsp; Suunto → Training → Export GPX
    </div>
    """, unsafe_allow_html=True)


def render_dashboard(gpx_bytes: bytes, filename: str):
    # Parse
    with st.spinner("Analyse du fichier GPX..."):
        try:
            df = parse_gpx(gpx_bytes)
        except ValueError as e:
            st.error(f"Erreur de lecture GPX : {e}")
            if st.button("↺ Recommencer"):
                for k in ['gpx_bytes','gpx_filename','analysis']:
                    st.session_state.pop(k, None)
                st.rerun()
            return

    info      = extract_race_info(df, filename)
    fi        = fatigue_index(df)
    flat_v    = flat_pace_estimate(df)
    grade_df  = grade_pace_profile(df)
    profile   = classify_profile(fi['decay_ratio'], flat_v)

    # ── Header ──────────────────────────────────────────────────
    col_title, col_reset = st.columns([5, 1])
    with col_title:
        st.markdown(
            f'<div class="hud-label">// ANALYSE COMPLETE //</div>'
            f'<div style="font-family:Barlow Condensed,sans-serif;font-size:1.8rem;'
            f'font-weight:700;letter-spacing:0.15em;color:#ffffff">'
            f'{info["name"].upper()}</div>',
            unsafe_allow_html=True,
        )
    with col_reset:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺ NOUVELLE ANALYSE"):
            for k in ['gpx_bytes','gpx_filename']:
                st.session_state.pop(k, None)
            st.rerun()

    st.markdown('<hr style="border-color:#152030;margin:0.5rem 0 1.5rem;">', unsafe_allow_html=True)

    # ── KPIs ────────────────────────────────────────────────────
    total_s = info['total_time_s']
    h_t = int(total_s//3600)
    m_t = int((total_s%3600)//60)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("DISTANCE", f"{info['distance_km']:.1f} km")
    k2.metric("TEMPS", f"{h_t}h{m_t:02d}'")
    k3.metric("D+", f"{int(info['elevation_gain'])} m")
    k4.metric("ALLURE MOY.", v_to_pace(info['avg_velocity_ms']) + "/km")
    with k5:
        badge_class = {
            "PROFIL ENDURANCE": "badge-endurance",
            "PROFIL EXPLOSIF":  "badge-explosif",
            "PROFIL FRAGILE":   "badge-fragile",
        }.get(profile, "badge-fragile")
        st.markdown(
            f'<div class="hud-label" style="margin-bottom:6px;">CLASSIFICATION</div>'
            f'<span class="{badge_class}">{profile}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Elevation + Pace profiles ────────────────────────────────
    st.markdown('<div class="section-title">PROFIL DE COURSE</div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown('<div class="hud-label" style="margin-bottom:4px;">Profil altimetrique</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_elevation(df), use_container_width=True)
    with p2:
        st.markdown('<div class="hud-label" style="margin-bottom:4px;">Allure au fil des km</div>', unsafe_allow_html=True)
        st.plotly_chart(chart_pace(df), use_container_width=True)

    # ── GAP + Quartiles ──────────────────────────────────────────
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

    # ── Metrics row ──────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    dr = fi['decay_ratio']
    dp = fi['decay_pct']
    m1.metric("ALLURE DE BASE (PLAT)", v_to_pace(flat_v) + "/km")
    m2.metric("RATIO Q4/Q1", f"{dr:.3f}" if not math.isnan(dr) else "N/A")
    m3.metric("PERTE DE VITESSE GAP", f"{dp:.1f}%" if not math.isnan(dp) else "N/A")
    m4.metric("ALT. MAX", f"{int(info['max_elevation'])} m")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Grade pace table ─────────────────────────────────────────
    st.markdown('<div class="section-title">ALLURE PAR TRANCHE DE PENTE</div>', unsafe_allow_html=True)
    st.dataframe(
        grade_df.style.set_properties(**{
            "background-color": "#0D1520",
            "color": "#C8D4DC",
            "border": "1px solid #152030",
        }),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendations ──────────────────────────────────────────
    st.markdown('<div class="section-title">3 RECOMMANDATIONS TACTIQUES</div>', unsafe_allow_html=True)
    advices = ADVICE.get(profile, ADVICE["PROFIL INCONNU"])
    for i, adv in enumerate(advices, 1):
        st.markdown(
            f'<div style="display:flex;gap:16px;margin-bottom:10px;padding:12px 16px;'
            f'background:#0D1520;border-left:2px solid rgba(65,200,232,0.3);">'
            f'<span style="font-family:DM Mono,monospace;color:#41C8E8;font-size:0.8rem;min-width:20px">{i:02d}</span>'
            f'<span style="color:#4A6070;font-size:0.9rem;font-family:DM Sans,sans-serif">{adv}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Lead capture + PDF ───────────────────────────────────────
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
            with st.spinner("Génération..."):
                pdf_bytes = generate_pdf(
                    info, fi, flat_v, profile, grade_df,
                    st.session_state.get("email_input", "")
                )
            fname = f"VERTEX_{info['name'].replace(' ','_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
            st.download_button(
                "⬇  TELECHARGER", data=pdf_bytes,
                file_name=fname, mime="application/pdf",
            )


# ══════════════════════════════════════════════════════════════════
# 6 — MAIN
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
