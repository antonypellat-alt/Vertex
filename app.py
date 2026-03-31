"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — app.py  (UI uniquement)                        ║
║         FC · Cadence · GAP · Zones · Découplage                 ║
╚══════════════════════════════════════════════════════════════════╝

Architecture modulaire v3.5 :
  gpx_parser.py — GPX parsing, Haversine, extract_race_info
  engine.py     — GAP, fatigue, FC, cadence, recommandations
  charts.py     — Plotly charts, PDF generator
  app.py        — UI Streamlit uniquement (ce fichier)
"""

import math
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as components

# ── Imports modules VERTEX ───────────────────────────────────────
from gpx_parser import parse_gpx as _parse_gpx, extract_race_info

@st.cache_data(show_spinner=False)
def parse_gpx(file_bytes: bytes):
    return _parse_gpx(file_bytes)

from tcx_parser import parse_tcx as _parse_tcx
@st.cache_data(show_spinner=False)
def parse_tcx_cached(file_bytes: bytes):
    return _parse_tcx(file_bytes)

try:
    from fit_parser import parse_fit as _parse_fit
    @st.cache_data(show_spinner=False)
    def parse_fit_cached(file_bytes: bytes):
        return _parse_fit(file_bytes)
    _FIT_AVAILABLE = True
except ImportError:
    _FIT_AVAILABLE = False
    def parse_fit_cached(file_bytes: bytes):
        raise ValueError(
            "Le support des fichiers FIT n'est pas disponible "
            "sur cette instance."
        )

from engine import (
    fatigue_index, flat_pace_estimate, grade_pace_profile,
    classify_profile, compute_hr_zones, cardiac_drift,
    compute_km_splits, hr_by_grade, cadence_analysis,
    generate_coach_recommendations, v_to_pace, ZONE_NAMES,
    detect_walk_segments, walk_stats, compute_performance_score,
    compute_verdict, detect_elevation_profile, apply_decay_correction,
)
from charts import (
    chart_elevation, chart_pace, chart_hr, chart_hr_pace_overlay,
    chart_quartiles, chart_grade_dist, chart_gap_profile,
    chart_cadence, chart_hr_by_grade, chart_ef_quartiles,
    generate_pdf,
)

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

/* ══ MOBILE — 640px et moins ══════════════════════════════════ */
@media (max-width: 640px) {
    /* Padding réduit */
    .block-container { padding: 0.8rem 0.6rem !important; }

    /* Titre VERTEX landing */
    .vertex-title { font-size: 3.2rem !important; letter-spacing: 0.12em !important; }
    .vertex-sub   { font-size: 0.58rem !important; letter-spacing: 0.18em !important; }

    /* Nom de la course */
    .race-name-block { font-size: 1.3rem !important; }

    /* Badges profil — pleine largeur */
    .badge-endurance, .badge-explosif, .badge-fragile {
        display: block !important;
        text-align: center !important;
        margin-top: 8px !important;
        font-size: 0.9rem !important;
        padding: 5px 10px !important;
    }

    /* Score cards — empilées via flex-wrap natif Streamlit */

    /* Verdict bar — passage en colonne */
    .verdict-mobile-stack {
        flex-direction: column !important;
        gap: 8px !important;
    }
    .verdict-mobile-stack > div:last-child {
        border-left: none !important;
        border-top: 1px solid #152030 !important;
        padding-left: 0 !important;
        padding-top: 8px !important;
    }

    /* km-table — scroll horizontal */
    .km-table-wrapper { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .km-table { font-size: 0.62rem !important; min-width: 420px; }
    .km-table th, .km-table td { padding: 4px 6px !important; }

    /* Section titles */
    .section-title { font-size: 0.58rem !important; }

    /* Recommandations — padding réduit */
    .rec-mobile { padding: 10px 12px !important; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 1 — LANDING PAGE
# ══════════════════════════════════════════════════════════════════

def render_landing():
    st.markdown("<br>", unsafe_allow_html=True)

    # ══ HERO ════════════════════════════════════════════════════════
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown('<div class="hud-label">// SYSTEM ONLINE — v3.5 //</div>', unsafe_allow_html=True)
        st.markdown('<div class="vertex-title">VERTEX</div>', unsafe_allow_html=True)
        st.markdown('<div class="vertex-sub">PERFORMANCE INTELLIGENCE</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.7rem;
                    font-weight:700;color:#ffffff;line-height:1.3;letter-spacing:0.04em;">
            En 30 secondes, tu sais<br>
            <span style="color:#41C8E8;">
                si tu as bien géré ton effort.
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3A5060;line-height:1.8;">
            Strava montre ce que tu as fait.<br>
            VERTEX explique ce que ça t'a coûté physiquement.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ══ 3 BLOCS VALEUR ══════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    v1, v2, v3 = st.columns(3)
    _VALUE_STYLE_TPL = (
        "border:1px solid #152030;padding:1.4rem 1.2rem;"
        "border-top:2px solid {color};background:#0D1520;"
    )
    with v1:
        st.markdown(f"""
        <div style="{_VALUE_STYLE_TPL.format(color='#41C8E8')}">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                        font-weight:700;color:#ffffff;margin-bottom:8px;line-height:1.3;">
                Sais si ton départ était trop rapide
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                        color:#41C8E8;letter-spacing:0.15em;margin-bottom:10px;">
                SCORE DE PERFORMANCE
            </div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
                        color:#3A5060;line-height:1.6;">
                Le score VERTEX mesure ta tenue d'allure sur les 4 quarts de course.
                Si tu as explosé en Q3, il le détecte.
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                        color:#1A3040;margin-top:10px;letter-spacing:0.1em;">
                GAP · Minetti 2002 · Decay ratio Q4/Q1
            </div>
        </div>""", unsafe_allow_html=True)
    with v2:
        st.markdown(f"""
        <div style="{_VALUE_STYLE_TPL.format(color='#C8A84B')}">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                        font-weight:700;color:#ffffff;margin-bottom:8px;line-height:1.3;">
                Comprends pourquoi ton cœur a décroché
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                        color:#C8A84B;letter-spacing:0.15em;margin-bottom:10px;">
                DÉCOUPLAGE CARDIAQUE
            </div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
                        color:#3A5060;line-height:1.6;">
                La FC qui monte pour maintenir la même allure, c'est mesurable.
                VERTEX te dit quand ça a basculé et à quelle intensité.
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                        color:#1A3040;margin-top:10px;letter-spacing:0.1em;">
                Efficacité cardiaque · Dérive · Zones FC Z1→Z5
            </div>
        </div>""", unsafe_allow_html=True)
    with v3:
        st.markdown(f"""
        <div style="{_VALUE_STYLE_TPL.format(color='#C84850')}">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.2rem;
                        font-weight:700;color:#ffffff;margin-bottom:8px;line-height:1.3;">
                Reçois les ajustements concrets pour ta prochaine course
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                        color:#C84850;letter-spacing:0.15em;margin-bottom:10px;">
                RECOMMANDATIONS COACH
            </div>
            <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;
                        color:#3A5060;line-height:1.6;">
                Cadence, nutrition, gestion des montées — les recommandations
                sont générées depuis tes données, pas depuis un template générique.
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                        color:#1A3040;margin-top:10px;letter-spacing:0.1em;">
                6 recommandations · Export PDF coach
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<hr style="border-color:#152030;">', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ══ FORMULAIRE + UPLOAD ════════════════════════════════════════
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        # ── PROFIL ATHLÈTE ──────────────────────────────────────
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#2A4050;
        letter-spacing:0.22em;text-transform:uppercase;border-bottom:1px solid #152030;
        padding-bottom:6px;margin-bottom:1rem;">// PROFIL ATHLÈTE //</div>
        """, unsafe_allow_html=True)

        # ── Checkbox mode zones (toujours visible) ───────────────────
        manual_zones = st.checkbox(
            "Je connais mes zones FC",
            value=st.session_state.get('zone_mode', 'auto') == 'manual',
            help="Cocher pour saisir tes seuils directement en bpm · La FCmax sera déduite du plafond Z5",
        )
        st.session_state['zone_mode'] = 'manual' if manual_zones else 'auto'

        # ── Champ FCmax : visible uniquement en mode auto ─────────────
        if st.session_state['zone_mode'] == 'auto':
            if 'fcmax' not in st.session_state:
                st.session_state['fcmax'] = 190
            fcmax_input = st.number_input(
                "FCmax (bpm)",
                min_value=150, max_value=220,
                step=1,
                key='fcmax',
                help="Ta fréquence cardiaque maximale réelle (défaut : 190 bpm)",
            )
            if fcmax_input == 190:
                st.markdown(
                    '<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;'
                    'color:#C8A84B;letter-spacing:0.12em;margin-top:4px;">'
                    '&#9650; Valeur par défaut — saisis ta FCmax réelle pour des zones précises'
                    '</div>',
                    unsafe_allow_html=True,
                )
        else:
            # Mode manuel : FCmax déduite de Z5 — on garde une valeur de référence pour les defaults
            fcmax_input = st.session_state.get('fcmax', 190)

        if st.session_state['zone_mode'] == 'manual':
            st.markdown("""
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;
            letter-spacing:0.15em;margin:8px 0 6px;">SEUILS DE ZONES (bpm)</div>
            """, unsafe_allow_html=True)
            zc1, zc2, zc3, zc4, zc5 = st.columns(5)
            default_zones = st.session_state.get('custom_zones', {
                'Z1': (0,                      int(fcmax_input * 0.60)),
                'Z2': (int(fcmax_input * 0.60), int(fcmax_input * 0.70)),
                'Z3': (int(fcmax_input * 0.70), int(fcmax_input * 0.80)),
                'Z4': (int(fcmax_input * 0.80), int(fcmax_input * 0.90)),
                'Z5': (int(fcmax_input * 0.90), fcmax_input),
            })
            z1_hi = zc1.number_input("Z1 max", min_value=80, max_value=220, value=default_zones['Z1'][1], step=1)
            z2_hi = zc2.number_input("Z2 max", min_value=80, max_value=220, value=default_zones['Z2'][1], step=1)
            z3_hi = zc3.number_input("Z3 max", min_value=80, max_value=220, value=default_zones['Z3'][1], step=1)
            z4_hi = zc4.number_input("Z4 max", min_value=80, max_value=220, value=default_zones['Z4'][1], step=1)
            z5_hi = zc5.number_input("Z5 max", min_value=80, max_value=220, value=default_zones['Z5'][1], step=1)
            st.session_state['custom_zones'] = {
                'Z1': (0,     z1_hi),
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
            "IMPORTER UN FICHIER GPX / TCX / FIT",
            type=["gpx", "tcx", "fit"],
            help="Garmin Connect → Exporter l'original · Polar → Export TCX · Suunto → Export FIT",
            label_visibility="visible",
        )
        if uploaded:
            st.session_state['gpx_bytes_pending'] = uploaded.read()
            st.session_state['gpx_filename']      = uploaded.name

        if st.session_state.get('gpx_bytes_pending'):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("▲  VALIDER ET ANALYSER", use_container_width=True):
                st.session_state['gpx_bytes'] = st.session_state.pop('gpx_bytes_pending')
                _cz = st.session_state.get('custom_zones')
                _zm = st.session_state.get('zone_mode', 'auto')
                if _zm == 'manual' and _cz and 'Z5' in _cz:
                    st.session_state['fcmax_confirmed'] = int(_cz['Z5'][1])
                else:
                    st.session_state['fcmax_confirmed'] = int(st.session_state.get('fcmax', 190))
                st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)
    features = [
        ("GAP ENGINE",   "Ta vraie allure normalisée sur le dénivelé",        "Grade-Adjusted Pace · Minetti 2002 · 4 quartiles"),
        ("ZONES FC",     "Où tu as couru — et combien ça t'a coûté",          "Distribution Z1→Z5 · Réponse cardiaque · Effort relatif"),
        ("CADENCE",      "Ta foulée : régulière, efficace, ou à corriger ?",  "Distribution · Zones optimales · Évolution"),
        ("COACH REPORT", "Un PDF que tu peux donner directement à ton coach", "6 recommandations · Export PDF complet"),
    ]
    cards_html = '<div style="display:flex;flex-wrap:wrap;gap:8px;">'
    for label, benefit, tech in features:
        cards_html += f"""
        <div style="flex:1;min-width:160px;border:1px solid #152030;padding:1.2rem;
                    border-top:2px solid rgba(65,200,232,0.3);">
            <div class="hud-label" style="color:#41C8E8">{label}</div>
            <div style="color:#C8D4DC;font-size:0.88rem;margin-top:8px;
                        font-family:'Barlow Condensed',sans-serif;font-weight:600;
                        letter-spacing:0.03em;line-height:1.3;">{benefit}</div>
            <div style="color:#2A4050;font-size:0.62rem;margin-top:6px;
                        font-family:'DM Mono',monospace;letter-spacing:0.08em;">{tech}</div>
        </div>"""
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#1A2A35;text-align:center;letter-spacing:0.1em;">
    EXPORT GPX AVEC FC : Garmin Connect → Activité → ··· → Exporter l'original &nbsp;|&nbsp;
    Polar → Flow → Activité → Exporter TCX &nbsp;|&nbsp;
    Suunto → App Suunto → Activité → Exporter FIT  |  Strava → utiliser Garmin Connect directement
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════

def render_dashboard(gpx_bytes: bytes, filename: str):
    # Guard taille fichier — OOM protection
    _MAX_FILE_MB = 15
    if len(gpx_bytes) > _MAX_FILE_MB * 1024 * 1024:
        st.error(
            f"Fichier trop volumineux ({len(gpx_bytes)//1024//1024} MB). "
            f"Maximum autorisé : {_MAX_FILE_MB} MB. "
            "Réduis la fréquence d'enregistrement GPS sur ta montre "
            "(1 point/sec → 1 point/5sec)."
        )
        if st.button("↺ Recommencer"):
            for k in ['gpx_bytes', 'gpx_filename', 'fcmax_confirmed',
                      'zone_mode', 'custom_zones']:
                st.session_state.pop(k, None)
            st.rerun()
        return

    # UX-KAI : ancre HTML native — fiable sur Streamlit Community Cloud (pas de cross-iframe)
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    components.html(
        """<script>
        setTimeout(function(){
            var el = window.parent.document.getElementById('top');
            if(el){ el.scrollIntoView(); }
        }, 200);
        </script>""",
        height=1,
    )

    ext = filename.lower().split('.')[-1]
    spinner_labels = {'tcx': "Analyse du fichier TCX...", 'fit': "Analyse du fichier FIT..."}
    spinner_label = spinner_labels.get(ext, "Analyse du fichier GPX...")
    with st.spinner(spinner_label):
        try:
            if ext == 'tcx':
                df = parse_tcx_cached(gpx_bytes)
            elif ext == 'fit':
                if not _FIT_AVAILABLE:
                    st.error(
                        "Les fichiers FIT ne sont pas supportés sur cette instance. "
                        "Exporte ton activité en GPX depuis Garmin Connect."
                    )
                    return
                df = parse_fit_cached(gpx_bytes)
            else:
                df = parse_gpx(gpx_bytes)
        except (ValueError, KeyError, AttributeError) as e:
            st.error(f"Erreur de lecture ({ext.upper()}) : {e}")
            if st.button("↺ Recommencer"):
                for k in ['gpx_bytes', 'gpx_filename', 'fcmax_confirmed',
                          'zone_mode', 'custom_zones']:
                    st.session_state.pop(k, None)
                st.rerun()
            return
        except MemoryError:
            st.error(
                "Fichier trop volumineux pour être traité. "
                "Réduis la fréquence d'enregistrement GPS sur ta montre."
            )
            if st.button("↺ Recommencer"):
                for k in ['gpx_bytes', 'gpx_filename', 'fcmax_confirmed',
                          'zone_mode', 'custom_zones']:
                    st.session_state.pop(k, None)
                st.rerun()
            return

        info     = extract_race_info(df, filename)
        fi           = fatigue_index(df)
        elev_profile = detect_elevation_profile(df)
        fi           = apply_decay_correction(fi, elev_profile, df)
        flat_v   = flat_pace_estimate(df)
        grade_df = grade_pace_profile(df)
        # classify_profile sur decay corrigé si disponible
        _decay_for_profile = fi.get('decay_ratio_corrected', fi['decay_ratio'])
        if _decay_for_profile is None or (isinstance(_decay_for_profile, float) and math.isnan(_decay_for_profile)):
            _decay_for_profile = fi['decay_ratio']
        profile  = classify_profile(_decay_for_profile)

    # Sprint 2 ④ : détection marche — enrichit df avec colonne is_walk
    df       = detect_walk_segments(df)
    wstats   = walk_stats(df)

    splits   = compute_km_splits(df)
    cad_an   = cadence_analysis(df)

    zone_mode    = st.session_state.get('zone_mode', 'auto')
    custom_zones = st.session_state.get('custom_zones', None)
    if zone_mode == 'manual' and custom_zones and 'Z5' in custom_zones:
        fcmax = int(custom_zones['Z5'][1])
    else:
        fcmax = int(st.session_state.get('fcmax_confirmed',
                    st.session_state.get('fcmax', 190)))

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
            for k in ['gpx_bytes', 'gpx_filename', 'fcmax', 'fcmax_confirmed', 'zone_mode', 'custom_zones']:
                st.session_state.pop(k, None)
            st.rerun()

    zones    = compute_hr_zones(df, fcmax, custom_zones if zone_mode == 'manual' else None) if info['has_hr'] else None
    _dp_per_km = info['elevation_gain'] / info['distance_km'] if info.get('distance_km', 0) > 0 else 0.0
    # SCI-3 : construire fi_score (ratio corrigé) AVANT cardiac_drift
    # pour que decay_v reflète le ratio corrigé et non les quartiles bruts
    fi_score = dict(fi)
    _corr = fi.get('decay_ratio_corrected', float('nan'))
    if not (isinstance(_corr, float) and math.isnan(_corr)) and _corr is not None:
        fi_score['decay_ratio'] = fi['decay_ratio_corrected']
        fi_score['decay_pct']   = fi['decay_pct_corrected']

    # [20] decay_v depuis ratio corrigé SCI-3 (pas depuis Q1/Q4 bruts)
    # Évite le faux NEGATIVE_SPLIT sur profils BVT/TDS (descente finale gonfle Q4)
    _decay_ratio_for_drift = fi_score.get('decay_ratio', 1.0)
    if _decay_ratio_for_drift is None or (
        isinstance(_decay_ratio_for_drift, float) and
        math.isnan(_decay_ratio_for_drift)
    ):
        _decay_ratio_for_drift = 1.0
    _decay_v_app = _decay_ratio_for_drift - 1.0

    drift    = cardiac_drift(df,
                   duration_s=info['total_time_s'],
                   dp_per_km=_dp_per_km,
                   decay_v=_decay_v_app) if info['has_hr'] else {
        'ef1': None, 'ef2': None, 'drift_pct': None, 'quartiles': {},
        'pattern': None, 'collapse_pct': None, 'fc_slope_bph': None,
        'fc_q1_mean': None, 'fc_q4_mean': None, 'insufficient_data': True,
        'decay_v': None,
    }
    # SCI-5 : flag Q1 D+ surchargé — angle mort decay_ratio sur profil front-loaded
    _ep = elev_profile  # déjà calculé plus haut
    _dplus_by_q = _ep.get('dplus_by_q', {})
    _total_dplus = sum(_dplus_by_q.values())
    if _total_dplus > 0:
        _q1_frac = _dplus_by_q.get('Q1', 0.0) / _total_dplus
        if _q1_frac > 0.35:
            drift['q1_dplus_overloaded'] = True
    hr_grade = hr_by_grade(df) if info['has_hr'] else None
    recs     = generate_coach_recommendations(profile, fi_score, drift, cad_an, info, fcmax)
    perf     = compute_performance_score(fi_score, drift, dp_per_km=_dp_per_km)
    verdict  = compute_verdict(fi_score, drift, perf)

    # Mode debug — accessible via ?debug=1 dans l'URL
    # Invisible pour les athlètes en usage normal
    if st.query_params.get("debug") == "1":
        with st.expander("🔧 DEBUG — valeurs intermédiaires", expanded=True):
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;'
                'font-size:0.65rem;color:#41C8E8;">// DEBUG MODE //</div>',
                unsafe_allow_html=True
            )
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.json({
                    'decay_ratio':           fi.get('decay_ratio'),
                    'decay_ratio_corrected': fi.get('decay_ratio_corrected'),
                    'decay_pct':             fi.get('decay_pct'),
                    'correction_applied':    fi.get('correction_applied'),
                    'elev_profile':          fi.get('elev_profile', {}).get('profile'),
                    '_decay_v_app':          round(_decay_v_app, 4),
                })
            with col_d2:
                st.json({
                    'drift_pattern':   drift.get('pattern'),
                    'fc_slope_bph':    drift.get('fc_slope_bph'),
                    'drift_pct':       drift.get('drift_pct'),
                    'score':           perf.get('score'),
                    'score_gap':       perf.get('score_gap'),
                    'score_ef':        perf.get('score_ef'),
                    'score_var':       perf.get('score_var'),
                    'verdict_code':    verdict.get('code'),
                    'zone':            perf.get('weights_meta', {}).get('zone'),
                })

    # ══ KAI UX — SÉQUENCE ABOVE THE FOLD ═══════════════════════════
    # Ordre : 1) Titre + badge  2) Verdict + Score fusionnés
    #         3) Warnings en expander replié  4) KPIs + détail

    # ── 1. Header : titre + badge ────────────────────────────────
    col_title, col_badge = st.columns([4, 1])
    with col_title:
        st.markdown(
            f'<div class="hud-label">// ANALYSE COMPLETE — v3.5 //</div>'
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

    st.markdown('<hr style="border-color:#152030;margin:0.5rem 0 1rem;">', unsafe_allow_html=True)

    # ── 2. Bloc VERDICT + SCORE fusionnés ───────────────────────
    _v      = verdict
    _vcode  = _v['code']
    _vlabel = _v['label']
    _vsub   = _v['sub']
    _vcolor = _v['color']
    _vicon  = _v['icon']
    _share  = _v.get('share_line', '')

    # Couleur du score global (calculé ici pour le bloc fusionné)
    _score     = perf['score']
    _s_gap     = perf['score_gap']
    _s_ef      = perf['score_ef']
    _s_var     = perf['score_var']
    _partial   = perf['partial']
    _p_reason  = perf['partial_reason'] or ''
    _w         = perf['weights']
    _wmeta     = perf.get('weights_meta', {})
    _zone_validated = _wmeta.get('zone_validated', True)
    _zone           = _wmeta.get('zone', 'Z3')

    if _score >= 80:   _score_color = '#41C8E8'
    elif _score >= 60: _score_color = '#C8A84B'
    else:              _score_color = '#C84850'

    _share_html = (
        f'<div style="width:100%;margin-top:12px;padding-top:10px;'
        f'border-top:1px solid #152030;font-family:\'DM Mono\',monospace;'
        f'font-size:0.62rem;color:#4A6070;letter-spacing:0.08em;font-style:italic;">'
        f'{_share}</div>'
    ) if _share else ''

    st.markdown(f"""
    <div class="verdict-mobile-stack"
         style="padding:20px 24px;background:#0D1620;
                border:1px solid #152030;border-left:3px solid {_vcolor};
                margin-bottom:16px;display:flex;flex-wrap:wrap;
                align-items:center;gap:20px;">
        <div style="flex:1;min-width:160px;">
            <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                        color:#2A4050;letter-spacing:0.22em;margin-bottom:4px;">
                VERDICT · {_vcode}
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;
                        font-weight:700;letter-spacing:0.08em;color:{_vcolor};
                        line-height:1.1;">
                {_vicon}&nbsp; {_vlabel}
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                        color:#8899AA;line-height:1.6;margin-top:8px;
                        word-break:break-word;">
                {_vsub}
            </div>
        </div>
        <div style="text-align:center;min-width:100px;padding-left:20px;
                    border-left:1px solid #152030;">
            <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                        color:#2A4050;letter-spacing:0.22em;margin-bottom:4px;">
                SCORE VERTEX
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:3.6rem;
                        font-weight:900;line-height:1;color:{_score_color};">{_score}</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                        color:#3A5060;margin-top:4px;">
                {"⚠ " + _p_reason if _partial else "/ 100"}
            </div>
        </div>
        {_share_html}
    </div>""", unsafe_allow_html=True)

    # ── 2b. Ligne action immédiate — B1 Sprint 5 ────────────────
    _action = _v.get('action_line', '')
    if _action:
        st.markdown(f"""
        <div style="padding:10px 20px;background:rgba(65,200,232,0.05);
                    border-left:3px solid {_vcolor};margin-bottom:16px;margin-top:-10px;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.05rem;
                        font-weight:600;color:{_vcolor};letter-spacing:0.04em;">
                {_action}
            </div>
        </div>""", unsafe_allow_html=True)

    # ── 3. Warnings — repliés dans expander ─────────────────────
    _vel_std = float(df['velocity'].std()) if 'velocity' in df.columns else 0.0
    _cad_amb = bool(df['cad_ambiguous'].any()) if 'cad_ambiguous' in df.columns else False
    n_gaps   = int(df['gap_flag'].sum()) if 'gap_flag' in df.columns else 0

    _warnings = []
    if not info['has_hr']:
        _hr_cov = info.get('hr_coverage_pct', 0)
        if _hr_cov > 0:
            _warnings.append(('warn', '▲ FC INSUFFISANTE',
                f'Fréquence cardiaque présente sur {_hr_cov}% des points '
                f'(seuil minimum : 30%) — analyse cardiaque non calculée. '
                "Exporte depuis Garmin Connect ou Polar Flow "
                "(Strava supprime la FC à l'export)."))
        else:
            _warnings.append(('crit', '⚠ FC ABSENTE',
                'Fréquence cardiaque non disponible — score et '
                'découplage cardiaque non calculés.'))
    if not info['has_cad']:
        _warnings.append(('info', '◆ CADENCE ABSENTE',
            'Cadence non enregistrée — les recommandations liées à la foulée ne sont pas disponibles.'))
    elif _cad_amb:
        _warnings.append(('info', '◆ CADENCE À VÉRIFIER',
            'Cadence potentiellement doublée automatiquement (Garmin) — vérifier la cohérence avec ta montre.'))
    if _vel_std > 2.5:
        _warnings.append(('warn', '▲ GPS INSTABLE',
            'Variance de vitesse élevée détectée — prudence sur la lecture de la vitesse et du GAP.'))
    if n_gaps > 0:
        _warnings.append(('warn', f'▲ GPS — {n_gaps} SEGMENT(S) INTERPOLÉ(S)',
            'Données reconstituées sur ces zones — prudence sur la lecture locale.'))
    if drift.get('insufficient_data') and info['has_hr']:
        _warnings.append(('info', '◆ TERRAIN INSUFFISANT',
            'Découplage cardiaque non calculable — moins de 10 min de terrain plat détecté sur ce parcours.'))
    if len(df) < 10 or info.get('distance_km', 1) < 1:
        _warnings.append(('warn', '▲ FICHIER COURT',
            'Fichier trop court pour une analyse fiable — moins de 10 points GPS ou moins de 1 km.'))
    if info.get('timestamps_estimated'):
        _warnings.append(('crit', '⚠ TIMESTAMPS ESTIMÉS',
            "Les horodatages GPS sont absents ou incomplets — "
            "les données temporelles (allure, dérive cardiaque) "
            "sont estimées et peuvent être inexactes. "
            "Vérifie les paramètres d'enregistrement de ta montre."))
    if info.get('elevation_degraded'):
        _warnings.append(('warn', '▲ LISSAGE GPS DÉGRADÉ',
            "Lissage d'élévation incomplet — D+/D- et GAP peuvent "
            'être approximatifs sur ce fichier.'))

    if _warnings:
        _warn_colors = {
            'crit': ('#C84850', 'rgba(200,72,80,0.15)'),
            'warn': ('#C8A84B', 'rgba(200,168,75,0.1)'),
            'info': ('#41C8E8', 'rgba(65,200,232,0.08)'),
        }
        # Niveau le plus élevé pour le label de l'expander
        _max_level = 'crit' if any(w[0] == 'crit' for w in _warnings) \
                     else 'warn' if any(w[0] == 'warn' for w in _warnings) \
                     else 'info'
        _exp_icon  = '⚠' if _max_level == 'crit' else '▲' if _max_level == 'warn' else '◆'
        _exp_color = _warn_colors[_max_level][0]

        with st.expander(f"{_exp_icon}  AVERTISSEMENTS TECHNIQUES ({len(_warnings)})", expanded=False):
            for _wlevel, _wtitle, _wmsg in _warnings:
                _wc, _wbg = _warn_colors[_wlevel]
                if _wlevel == "crit":
                    _pad, _bwidth, _title_w = "14px 18px", "4px", "font-weight:700;"
                elif _wlevel == "warn":
                    _pad, _bwidth, _title_w = "10px 16px", "3px", "font-weight:500;"
                else:
                    _pad, _bwidth, _title_w = "8px 16px",  "2px", "font-weight:400;"
                st.markdown(f"""
                <div style="padding:{_pad};background:{_wbg};border-left:{_bwidth} solid {_wc};
                            margin-bottom:6px;display:flex;align-items:center;gap:12px;">
                    <div>
                        <span style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                     color:{_wc};letter-spacing:0.2em;{_title_w}">{_wtitle} &nbsp;·&nbsp; </span>
                        <span style="font-family:'DM Mono',monospace;font-size:0.58rem;
                                     color:#4A6070;">{_wmsg}</span>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ── K2 : textes d'interprétation ────────────────────────────
    # Score global — ligne 1 : lecture immédiate / ligne 2 : lecture pro
    if _score >= 80:
        _score_interp1 = "Effort bien géré de bout en bout."
        _score_interp2 = "Allure normalisée stable · couplage cardio-vitesse maintenu."
    elif _score >= 60:
        _score_interp1 = "Gestion correcte, avec des signes de fatigue."
        _score_interp2 = "Ralentissement GAP ou réponse cardiaque dégradée sur la 2e moitié."
    else:
        _score_interp1 = "Effort mal dosé — faiblesse significative détectée."
        _score_interp2 = "Chute d'allure normalisée et/ou fatigue cardio-musculaire marquée."

    # GAP Q4/Q1
    if _s_gap >= 80:
        _gap_interp = "Tenue d'allure solide · <8% de perte sur le dernier quart."
    elif _s_gap >= 60:
        _gap_interp = "Ralentissement notable en fin de course."
    else:
        _gap_interp = "Effondrement d'allure · départ trop rapide ou nutrition insuffisante."

    # Dérive EF
    if _s_ef is None:
        _ef_interp = "FC absente — réponse cardiaque non mesurable."
    elif _s_ef >= 80:
        _ef_interp = "Couplage FC/vitesse maintenu · fatigue aérobie faible."
    elif _s_ef >= 60:
        _ef_interp = "Découplage progressif · la FC monte pour maintenir l'allure."
    else:
        _ef_interp = "Découplage sévère · stress cardiovasculaire élevé sur la fin."

    # Régularité
    if _s_var >= 80:
        _var_interp = "Course linéaire · gestion d'allure maîtrisée."
    elif _s_var >= 60:
        _var_interp = "Variations modérées · terrain ou effort irrégulier."
    else:
        _var_interp = "Forte irrégularité · à croiser avec le profil altimétrique."

    # Score — HTML flex-wrap : 1 grande carte + 3 petites
    _gap_color = '#41C8E8' if _s_gap >= 80 else '#C8A84B' if _s_gap >= 60 else '#C84850'
    _ef_color  = ('#41C8E8' if _s_ef >= 80 else '#C8A84B' if _s_ef >= 60 else '#C84850') if _s_ef is not None else '#2A4050'
    _var_color = '#41C8E8' if _s_var >= 80 else '#C8A84B' if _s_var >= 60 else '#C84850'
    ef_val     = str(_s_ef) if _s_ef is not None else 'N/A'
    ef_bar     = _s_ef if _s_ef is not None else 0

    # ── Score interprétation — ligne sous le bloc fusionné ───────
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                color:{_score_color};margin-bottom:8px;margin-top:-8px;line-height:1.5;
                padding:0 4px;">
        {_score_interp1} &nbsp;·&nbsp;
        <span style="color:#3A5060">{_score_interp2}</span>
    </div>""", unsafe_allow_html=True)

    # ── Sous-scores — repliés par défaut (signal Samuel) ─────────
    with st.expander("▶  DÉTAIL DU SCORE · GAP / CARDIAQUE / RÉGULARITÉ", expanded=False):
        st.markdown(f"""
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
            <div style="flex:1;min-width:120px;background:#0D1520;border:1px solid #152030;padding:16px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                            color:#2A4050;letter-spacing:0.18em;">GAP Q4/Q1 · {int(_w['gap']*100)}%</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;
                            font-weight:700;color:{_gap_color};">{_s_gap}</div>
                <div style="background:#152030;height:4px;border-radius:2px;margin-top:8px;">
                    <div style="background:{_gap_color};height:4px;width:{_s_gap}%;border-radius:2px;"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                            color:#3A5060;margin-top:8px;line-height:1.5;">{_gap_interp}</div>
            </div>
            <div style="flex:1;min-width:120px;background:#0D1520;border:1px solid #152030;padding:16px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                            color:#2A4050;letter-spacing:0.18em;">EFFICACITÉ CARDIAQUE · {int(_w['ef']*100)}%</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;
                            font-weight:700;color:{_ef_color};">{ef_val}</div>
                <div style="background:#152030;height:4px;border-radius:2px;margin-top:8px;">
                    <div style="background:{_ef_color};height:4px;width:{ef_bar}%;border-radius:2px;"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                            color:#3A5060;margin-top:8px;line-height:1.5;">{_ef_interp}</div>
            </div>
            <div style="flex:1;min-width:120px;background:#0D1520;border:1px solid #152030;padding:16px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                            color:#2A4050;letter-spacing:0.18em;">RÉGULARITÉ · {int(_w['var']*100)}%</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;
                            font-weight:700;color:{_var_color};">{_s_var}</div>
                <div style="background:#152030;height:4px;border-radius:2px;margin-top:8px;">
                    <div style="background:{_var_color};height:4px;width:{_s_var}%;border-radius:2px;"></div>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                            color:#3A5060;margin-top:8px;line-height:1.5;">{_var_interp}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#1A2A35;
                letter-spacing:0.08em;line-height:1.6;margin-bottom:12px;margin-top:-6px;">
        Score expérimental — basé sur un modèle physiologique (Minetti 2002 + découplage cardiaque).
        Non validé cliniquement. Usage personnel et pédagogique uniquement.
    </div>""", unsafe_allow_html=True)

    if not _zone_validated and _zone == 'Z2':
        st.markdown("""
        <div style="font-family:'DM Mono',monospace;font-size:0.52rem;color:#3A5060;
                    letter-spacing:0.06em;line-height:1.5;margin-bottom:8px;margin-top:-4px;
                    padding:6px 8px;border-left:2px solid #1A3040;">
            Analyse en cours de validation sur ce type de terrain.
        </div>""", unsafe_allow_html=True)

    # ── KPIs ligne 1 ────────────────────────────────────────────
    total_s = info['total_time_s']
    h_t = int(total_s//3600)
    m_t = int((total_s % 3600)//60)

    k1, k2, k3 = st.columns(3)
    k1.metric("DISTANCE", f"{info['distance_km']:.1f} km")
    k2.metric("TEMPS", f"{h_t}h{m_t:02d}'")
    k3.metric("D+", f"{int(info['elevation_gain'])} m")
    if info['has_hr']:
        k4, k5, k6 = st.columns(3)
        k4.metric("ALLURE MOY.", v_to_pace(info['avg_velocity_ms']) + "/km")
        k5.metric("FC MOYENNE", f"{int(info['hr_mean'])} bpm")
        k6.metric("FC MAX RÉELLE", f"{fcmax} bpm", delta=f"pic : {int(info['hr_max'])} bpm")
    else:
        k4 = st.columns(1)[0]  # Fix B7
        k4.metric("ALLURE MOY.", v_to_pace(info['avg_velocity_ms']) + "/km")

    # ── KPIs ligne 2 — pattern-aware v3.4 ───────────────────────
    if info['has_cad']:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        c1.metric("CADENCE MOY.", f"{int(info['cad_mean'])} spm")
        dr = fi['decay_ratio']
        dp = fi['decay_pct']
        c2.metric("TENUE D'ALLURE", f"{dr:.3f}" if not math.isnan(dr) else "N/A")
        c3.metric("PERTE D'ALLURE Q4/Q1", f"{dp:.1f}%" if not math.isnan(dp) else "N/A")

        # K2 — interprétation ratio et écart
        if not math.isnan(dr):
            if dr >= 0.90:
                _dr_interp = "Tenue d'allure solide · perte <10% entre début et fin."
            elif dr >= 0.80:
                _dr_interp = "Ralentissement notable · fatigue progressive détectée."
            else:
                _dr_interp = "Chute sévère · effort au-dessus du seuil sur la durée."
            c2.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;'
                f'color:#3A5060;line-height:1.5;margin-top:4px;">{_dr_interp}</div>',
                unsafe_allow_html=True
            )

        if not math.isnan(dp):
            if abs(dp) < 8:
                _dp_interp = "Écart normal · course bien dosée."
            elif abs(dp) < 15:
                _dp_interp = "Fatigue significative · revoir stratégie de départ."
            else:
                _dp_interp = "Alerte · perte d'allure GAP >15% entre Q1 et Q4."
            c3.markdown(
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;'
                f'color:#3A5060;line-height:1.5;margin-top:4px;">{_dp_interp}</div>',
                unsafe_allow_html=True
            )

        if info['has_hr'] and drift:
            _pattern      = drift.get('pattern')
            _insufficient = drift.get('insufficient_data', False)
            _collapse_pct = drift.get('collapse_pct')
            _drift_pct    = drift.get('drift_pct')
            if not _insufficient:
                if _pattern == 'COLLAPSE':
                    c4.metric("EFFONDREMENT FC", f"{_collapse_pct:.1f}%")
                elif _pattern == 'DRIFT-CARDIO' and _drift_pct is not None:
                    c4.metric("SURCHARGE CARDIAQUE", f"{_drift_pct:.1f}%")
                elif _pattern == 'DRIFT-NEURO' and _drift_pct is not None:
                    c4.metric("FATIGUE MUSCULAIRE", f"{_drift_pct:.1f}%")
                elif _drift_pct is not None:
                    c4.metric("DERIVE CARDIAQUE", f"{_drift_pct:.1f}%")

    # ── UX-3 : Bloc dérive cardiaque dédié ──────────────────────
    if info['has_hr'] and drift:
        _dp        = drift.get('pattern')
        _di        = drift.get('insufficient_data', False)
        _dfc_q1    = drift.get('fc_q1_mean')
        _dfc_q4    = drift.get('fc_q4_mean')
        _dslope    = drift.get('fc_slope_bph')
        _dcoll_pct = drift.get('collapse_pct')
        _ddrift_pct= drift.get('drift_pct')

        if _di:
            _dlabel = 'NON CALCULABLE'
            _dcolor = '#2A4050'
            _dsub   = 'Moins de 10 min de terrain plat détecté — réponse cardiaque non isolable sur ce parcours.'
            _dmetrics_html = ''
        elif _dp == 'COLLAPSE':
            _dlabel = 'SIGNAL CARDIAQUE ANORMAL'
            _dcolor = '#C84850'
            _dsub   = 'La fréquence cardiaque a chuté anormalement sur les parties plates — le signal cardiaque n\'est plus fiable après ce point. Utilise les données de fin de course comme signal d\'alerte, pas comme référence.'
            _dmetrics_html = f"""
            <div style="display:flex;flex-wrap:wrap;gap:24px;margin-top:10px;">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q1</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">{_dfc_q1:.0f} bpm</div>
                </div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;color:#2A4050;align-self:flex-end;padding-bottom:4px;">→</div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q4</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">{_dfc_q4:.0f} bpm</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">CHUTE FC</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">{abs(_dcoll_pct):.1f}%</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">PENTE FC</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">{_dslope:.1f} bpm/h</div>
                </div>
            </div>"""
        elif _dp == 'DRIFT':
            _dlabel = 'FATIGUE LEGERE'
            _dcolor = '#C8A84B'
            _dsub   = 'Legere fatigue cardiaque detectable sur la 2e moitie — signal precoce, non critique. Travaille le volume en Z2 pour repousser ce seuil.'
            _dmetrics_html = f"""
            <div style="display:flex;flex-wrap:wrap;gap:24px;margin-top:10px;">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q1</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_dfc_q1:.0f} bpm</div>
                </div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;color:#2A4050;align-self:flex-end;padding-bottom:4px;">→</div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q4</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_dfc_q4:.0f} bpm</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">DERIVE</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_ddrift_pct:.1f}%</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">PENTE FC</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_dslope:.1f} bpm/h</div>
                </div>
            </div>"""
        elif _dp == 'DRIFT-CARDIO':
            _dlabel = 'SURCHARGE CARDIAQUE'
            _dcolor = '#C84850'
            _dsub   = 'La frequence cardiaque a monte pendant que la vitesse baissait — ton coeur compensait la fatigue musculaire. Signal de surcharge cardio-metabolique.'
            _dmetrics_html = f"""
            <div style="display:flex;flex-wrap:wrap;gap:24px;margin-top:10px;">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q1</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">{_dfc_q1:.0f} bpm</div>
                </div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;color:#2A4050;align-self:flex-end;padding-bottom:4px;">→</div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q4</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">{_dfc_q4:.0f} bpm</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">DERIVE VITESSE</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">{_ddrift_pct:.1f}%</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">PENTE FC</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C84850;">+{_dslope:.1f} bpm/h</div>
                </div>
            </div>"""
        elif _dp == 'DRIFT-NEURO':
            _dlabel = 'FATIGUE MUSCULAIRE'
            _dcolor = '#C8A84B'
            _dsub   = 'La vitesse a baisse alors que la frequence cardiaque restait stable — tes muscles se sont fatigues avant ton coeur. Signal neuromusculaire.'
            _dmetrics_html = f"""
            <div style="display:flex;flex-wrap:wrap;gap:24px;margin-top:10px;">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q1</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_dfc_q1:.0f} bpm</div>
                </div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;color:#2A4050;align-self:flex-end;padding-bottom:4px;">→</div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q4</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_dfc_q4:.0f} bpm</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">DERIVE VITESSE</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_ddrift_pct:.1f}%</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">PENTE FC</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#C8A84B;">{_dslope:.1f} bpm/h</div>
                </div>
            </div>"""
        else:
            _dlabel = 'CARDIAQUE STABLE'
            _dcolor = '#41C8E8'
            _dsub   = 'La fréquence cardiaque est restée proportionnelle à l\'effort du début à la fin. Effort bien calibré : continue sur cette base.'
            _fc_q1_disp = f"{_dfc_q1:.0f}" if _dfc_q1 else '--'
            _fc_q4_disp = f"{_dfc_q4:.0f}" if _dfc_q4 else '--'
            _slope_disp = f"{_dslope:.1f}" if _dslope is not None else '--'
            _dmetrics_html = f"""
            <div style="display:flex;flex-wrap:wrap;gap:24px;margin-top:10px;">
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q1</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#41C8E8;">{_fc_q1_disp} bpm</div>
                </div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;color:#2A4050;align-self:flex-end;padding-bottom:4px;">→</div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">FC Q4</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#41C8E8;">{_fc_q4_disp} bpm</div>
                </div>
                <div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#3A5060;letter-spacing:0.18em;">PENTE FC</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;font-weight:700;color:#41C8E8;">{_slope_disp} bpm/h</div>
                </div>
            </div>"""

        st.markdown(f"""
        <div style="padding:16px 24px;background:#0D1620;border:1px solid #152030;
                    border-left:3px solid {_dcolor};margin-bottom:0;padding-bottom:10px;">
            <div style="font-family:'DM Mono',monospace;font-size:0.55rem;
                        color:#2A4050;letter-spacing:0.22em;margin-bottom:6px;">
                DÉRIVE CARDIAQUE
            </div>
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;
                        font-weight:700;letter-spacing:0.08em;color:{_dcolor};line-height:1.1;">
                {_dlabel}
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                        color:#4A6070;margin-top:6px;line-height:1.5;">
                {_dsub}
            </div>
        </div>""", unsafe_allow_html=True)
        if _dmetrics_html:
            st.markdown(
                f'<div style="padding:10px 24px 16px 24px;background:#0D1620;'
                f'border:1px solid #152030;border-top:none;'
                f'border-left:3px solid {_dcolor};margin-bottom:16px;">'
                + _dmetrics_html +
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Walk stats — Sprint 2 ④ ──────────────────────────────────
    if wstats and wstats.get('has_steep') and wstats.get('walk_ratio') is not None:
        walk_color = '#C8A84B' if wstats['walk_ratio'] > 30 else '#41C8E8'
        st.markdown(f"""
        <div style="padding:10px 18px;background:#0D1520;
                    border-left:2px solid {walk_color};margin-bottom:12px;
                    display:flex;gap:32px;align-items:center;">
            <div>
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                            color:#2A4050;letter-spacing:0.2em;">MARCHE ACTIVE (pente >15%)</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.6rem;
                            font-weight:700;color:{walk_color};">
                    {wstats['walk_ratio']:.0f}%
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3A5060;">
                    du temps sur sections raides
                </div>
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#4A6070;line-height:1.8;">
                Marche : {wstats['walk_time_min']:.0f} min
                &nbsp;|&nbsp; Course : {wstats['run_time_min']:.0f} min
                &nbsp;|&nbsp; {wstats['n_walk_segments']} segment(s) détecté(s)
            </div>
        </div>""", unsafe_allow_html=True)

    # ══ SECTION 1 : PROFIL DE COURSE ════════════════════════════
    with st.expander("▶  PROFIL DE COURSE", expanded=False):
        p1, p2 = st.columns(2)
        with p1:
            st.markdown('<div class="hud-label" style="margin-bottom:4px;">Profil altimétrique</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_elevation(df), use_container_width=True)
        with p2:
            st.markdown('<div class="hud-label" style="margin-bottom:4px;">Allure au fil des km</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_pace(df), use_container_width=True)

    # ══ SECTION 2 : FC ══════════════════════════════════════════
    if info['has_hr']:
        with st.expander("▶  FRÉQUENCE CARDIAQUE", expanded=False):
            h1, h2 = st.columns(2)
            with h1:
                st.markdown('<div class="hud-label" style="margin-bottom:4px;">FC sur la course</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_hr(df, fcmax), use_container_width=True)
            with h2:
                st.markdown('<div class="hud-label" style="margin-bottom:4px;">FC × Allure superposées</div>', unsafe_allow_html=True)
                st.plotly_chart(chart_hr_pace_overlay(df), use_container_width=True)

            if zones:
                zone_mode_label = '· ZONES MANUELLES' if zone_mode == 'manual' else f'· % FCMAX AUTO ({fcmax} bpm)'
                st.markdown(
                    f'<div class="hud-label" style="margin-top:1rem;margin-bottom:12px;">'
                    f'Distribution des zones FC <span style="color:#41C8E8">{zone_mode_label}</span></div>',
                    unsafe_allow_html=True,
                )
                zone_labels     = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
                zone_colors_hex = ['#1A3A4A', '#1A5060', '#1A8AAA', '#C8A84B', '#C84850']
                zone_cards_html = '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">'
                for i, z in enumerate(zone_labels):
                    pct = zones['pct'][z]
                    t   = zones['time'][z]
                    mm  = int(t//60)
                    bpm = zones['bpm'][z]
                    zone_cards_html += f"""
                    <div style="flex:1;min-width:80px;background:#0D1520;border:1px solid #152030;
                                border-top:2px solid {zone_colors_hex[i]};padding:10px;text-align:center;">
                        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;letter-spacing:0.2em">{z}</div>
                        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.8rem;font-weight:700;color:{zone_colors_hex[i]}">{pct:.0f}%</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3A5060">{mm} min</div>
                        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:#2A4050;margin-top:4px">{bpm[0]}–{bpm[1]} bpm</div>
                        <div style="font-family:'DM Sans',sans-serif;font-size:0.65rem;color:#2A4050;margin-top:4px">{ZONE_NAMES[z]}</div>
                    </div>"""
                zone_cards_html += '</div>'
                st.markdown(zone_cards_html, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            if hr_grade is not None and len(hr_grade) > 3:
                hz1, hz2 = st.columns(2)
                with hz1:
                    st.markdown('<div class="hud-label" style="margin-bottom:4px;">FC moyenne par pente</div>', unsafe_allow_html=True)
                    st.plotly_chart(chart_hr_by_grade(hr_grade), use_container_width=True)
                with hz2:
                    st.markdown('<div class="hud-label" style="margin-bottom:4px;">Réponse cardiaque par quartile</div>', unsafe_allow_html=True)
                    if drift['quartiles']:
                        st.plotly_chart(chart_ef_quartiles(drift['quartiles']), use_container_width=True)

    # ══ SECTION 3 : CADENCE ═════════════════════════════════════
    if info['has_cad']:
        with st.expander("▶  CADENCE", expanded=False):
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
                        bar_w  = int(v * 1.4)
                        is_opt = k in ['170-180', '180-190', '190-200']
                        color  = '#41C8E8' if is_opt else '#1A3A4A'
                        st.markdown(f"""
                        <div style="margin-bottom:5px;">
                            <div style="display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:0.6rem;color:#3A5060;margin-bottom:2px;">
                                <span>{k} spm</span><span>{v:.0f}%</span>
                            </div>
                            <div style="background:{color};height:4px;width:{min(bar_w,100)}%;border-radius:1px;"></div>
                        </div>""", unsafe_allow_html=True)

    # ══ SECTION 4 : GAP ═════════════════════════════════════════
    with st.expander("▶  ANALYSE DE FATIGUE GAP", expanded=False):
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

        st.markdown('<div class="section-title" style="margin-top:1rem;">ALLURE PAR TRANCHE DE PENTE</div>', unsafe_allow_html=True)
        st.dataframe(
            grade_df.style.set_properties(**{
                "background-color": "#0D1520",
                "color": "#C8D4DC",
                "border": "1px solid #152030",
            }),
            use_container_width=True, hide_index=True,
        )

    # ══ SECTION 5 : SPLITS PAR KM ═══════════════════════════════
    with st.expander("▶  SPLITS PAR KM", expanded=False):
        if splits:
            has_hr_sp  = any(s['hr'] for s in splits)
            has_cad_sp = any(s['cadence'] for s in splits)

            headers = ['KM', 'ALLURE', 'GAP', 'D+', 'D-']
            if has_hr_sp:  headers.append('FC')
            if has_cad_sp: headers.append('CAD')
            has_walk_sp = any(s.get('has_walk') for s in splits)
            if has_walk_sp: headers.append('🚶')

            header_row  = ''.join(f'<th>{h}</th>' for h in headers)
            rows_html   = ''
            valid_paces = [s['pace_s'] for s in splits if s.get('pace_s')]
            med_pace    = sum(valid_paces) / len(valid_paces) if valid_paces else None

            # ── K3 : détection zones chaudes ────────────────────
            worst_km = None
            best_km  = None
            fcmax_km = None

            if valid_paces:
                # Fix M2 : index-based, pas float equality
                worst_idx = valid_paces.index(max(valid_paces))
                best_idx  = valid_paces.index(min(valid_paces))
                paced_splits = [s for s in splits if s.get('pace_s')]
                worst_km = paced_splits[worst_idx]['km']
                best_km  = paced_splits[best_idx]['km']
                # Fix B1 : si même km, best non déclaré
                if best_km == worst_km:
                    best_km = None

            if has_hr_sp:
                valid_hr = [(s['km'], s['hr']) for s in splits if s.get('hr')]
                if valid_hr:
                    fcmax_km = max(valid_hr, key=lambda x: x[1])[0]

            # ── Légende zones chaudes ────────────────────────────
            legend_items = []
            if worst_km is not None:
                legend_items.append('<span style="color:#C84850">▼ KM LE PLUS LENT</span>')
            if best_km is not None:
                legend_items.append('<span style="color:#41C8E8">▲ KM LE PLUS RAPIDE</span>')
            if fcmax_km is not None:
                legend_items.append('<span style="color:#C8A84B">● FC MAX</span>')

            if legend_items:
                st.markdown(
                    '<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;'
                    'letter-spacing:0.18em;color:#2A4050;margin-bottom:10px;">'
                    + '&nbsp;&nbsp;·&nbsp;&nbsp;'.join(legend_items)
                    + '</div>',
                    unsafe_allow_html=True
                )

            for sp in splits:
                pace_s = sp.get('pace_s')
                km     = sp['km']

                # Fond de ligne : zones chaudes prioritaires
                if km == worst_km:
                    row_bg     = 'background:rgba(200,72,80,0.10);'
                    row_border = 'border-left:3px solid #C84850;'
                elif km == fcmax_km:
                    row_bg     = 'background:rgba(200,168,75,0.10);'
                    row_border = 'border-left:3px solid #C8A84B;'
                elif km == best_km:
                    row_bg     = 'background:rgba(65,200,232,0.08);'
                    row_border = 'border-left:3px solid #41C8E8;'
                else:
                    row_bg     = ''
                    row_border = ''

                # Couleur allure (coloration relative, inchangée)
                color = '#C8D4DC'
                if pace_s and med_pace:
                    if pace_s < med_pace * 0.92:
                        color = '#41C8E8'
                    elif pace_s > med_pace * 1.08:
                        color = '#C84850'

                row_style = f'style="{row_bg}{row_border}"'
                row  = f'<td>{km}</td>'
                row += f'<td style="color:{color}">{sp["pace"]}</td>'
                row += f'<td>{sp["gap"]}</td>'
                row += f'<td style="color:#41C8E8">+{sp["d_pos"]}m</td>'
                row += f'<td style="color:#C84850">-{sp["d_neg"]}m</td>'
                if has_hr_sp:
                    hr_color = '#C84850' if sp['hr'] and sp['hr'] > fcmax * 0.92 else '#C8D4DC'
                    row += f'<td style="color:{hr_color}">{sp["hr"] or "--"}</td>'
                if has_cad_sp:
                    cad_color = '#41C8E8' if sp['cadence'] and 170 <= sp['cadence'] <= 200 else '#C8D4DC'
                    row += f'<td style="color:{cad_color}">{sp["cadence"] or "--"}</td>'
                if has_walk_sp:
                    walk_flag = '🚶' if sp.get('has_walk') else ''
                    row += f'<td style="color:#C8A84B;text-align:center">{walk_flag}</td>'
                rows_html += f'<tr {row_style}>{row}</tr>'

            st.markdown(f"""
            <div class="km-table-wrapper">
            <table class="km-table">
                <thead><tr>{header_row}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="hud-label">Aucun split disponible.</div>', unsafe_allow_html=True)

    # ══ SECTION 6 : RECOMMANDATIONS COACH ═══════════════════════
    # Contrat Kai : top 3 (les plus critiques, triées par engine.py)
    # affichées par défaut — reste dans expander imbriqué.
    with st.expander("▶  RECOMMANDATIONS COACH", expanded=False):
        level_styles = {
            'info': ('rgba(65,200,232,0.3)', '#41C8E8', '◆ INFO'),
            'warn': ('rgba(200,168,75,0.6)', '#C8A84B', '▲ ATTENTION'),
            'crit': ('rgba(200,72,80,0.6)',  '#C84850', '● PRIORITAIRE'),
        }

        def _render_rec(rec):
            border_color, text_color, label = level_styles.get(rec['level'], level_styles['info'])
            st.markdown(f"""
            <div style="padding:14px 18px;background:#0D1520;border-left:3px solid {border_color};margin-bottom:10px;">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;color:{text_color};letter-spacing:0.2em;margin-bottom:6px">{label}</div>
                <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;color:#ffffff;margin-bottom:6px">{rec['title']}</div>
                <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:#4A6070;line-height:1.6">{rec['body']}</div>
            </div>""", unsafe_allow_html=True)

        # Top 3 — toujours visibles (CRIT en premier grâce au tri engine.py)
        for rec in recs[:3]:
            _render_rec(rec)

        # Recos supplémentaires — repliées
        if len(recs) > 3:
            with st.expander(f"▶  {len(recs) - 3} recommandation(s) supplémentaire(s)", expanded=False):
                for rec in recs[3:]:
                    _render_rec(rec)

    # ══ SECTION 7 : PDF ═════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">FEUILLE DE ROUTE TACTIQUE</div>', unsafe_allow_html=True)
    email = st.text_input(
        "Recevoir mes futurs plans par email (optionnel)",
        placeholder="ton@email.com",
        key="email_input",
    )
    if st.button("▲  GENERER LE PDF"):
        try:
            with st.spinner("Génération du rapport..."):
                pdf_bytes = generate_pdf(
                    info, fi, flat_v, profile, grade_df,
                    zones, drift, cad_an, splits, recs, fcmax, perf,
                    verdict,
                    st.session_state.get("email_input", "")
                )
            fname = f"VERTEX_{info['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"
            st.download_button(
                "⬇  TELECHARGER LE RAPPORT", data=pdf_bytes,
                file_name=fname, mime="application/pdf",
            )
        except Exception:
            st.error(
                "La génération du rapport a échoué. "
                "Réessaie ou contacte le support si le problème persiste."
            )

    # ── Feedback beta — toujours visible sous la section PDF ────
    st.markdown("""
    <div style="margin-top:20px;padding:14px 18px;background:#0D1520;
                border:1px solid #152030;border-left:3px solid #C8A84B;">
        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                    color:#C8A84B;letter-spacing:0.2em;margin-bottom:6px;">
            BETA FEEDBACK
        </div>
        <div style="font-family:'Barlow Condensed',sans-serif;font-size:1rem;
                    font-weight:700;color:#ffffff;margin-bottom:8px;">
            Cette analyse t'a été utile ?
        </div>
        <a href="https://tally.so/r/zxeJPM" target="_blank"
           style="font-family:'Barlow Condensed',sans-serif;font-size:0.9rem;
                  color:#C8A84B;letter-spacing:0.15em;text-decoration:none;">
            ▶ 2 MINUTES DE FEEDBACK →
        </a>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# 3 — MAIN
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
