"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — charts.py                                       ║
║         Plotly charts · PDF generator                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine import gap_correction_vec, v_to_pace


def _isnan(v) -> bool:
    """Copie locale — évite dépendance sur symbole privé engine.py."""
    if v is None:
        return True
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


# ══════════════════════════════════════════════════════════════════
# PLOTLY — LAYOUT DE BASE
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


# ══════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════

def chart_elevation(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    fig = go.Figure()

    # Courbe de base — élévation complète
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['elevation'],
        mode='lines', line=dict(color='#41C8E8', width=1.5),
        fill='tozeroy', fillcolor='rgba(65,200,232,0.06)',
        name='Élévation',
    ))

    # Sprint 2 ④ : segments marche active en surimpression orange
    if 'is_walk' in df.columns and df['is_walk'].any():
        walk_mask = df['is_walk']
        # On trace uniquement les points marche (NaN ailleurs → segments discontinus)
        walk_ele = df['elevation'].where(walk_mask)
        fig.add_trace(go.Scatter(
            x=dist_km, y=walk_ele,
            mode='lines', line=dict(color='#C8A84B', width=3),
            name='Marche active',
            connectgaps=False,
        ))

    fig.update_layout(**_layout(
        height=180,
        showlegend=bool(df['is_walk'].any()) if 'is_walk' in df.columns else False,
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#4A6070', size=9)),
        yaxis_title="m",
        xaxis_title="km",
    ))
    return fig


def chart_pace(df: pd.DataFrame) -> go.Figure:
    dist_km = df['distance'] / 1000
    v = df['velocity'].to_numpy()
    pace = np.where(v > 0.3, 1000 / v / 60, np.nan)
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
    zone_bounds = [
        (0, 0.60, '#1A3A4A'), (0.60, 0.70, '#1A5060'),
        (0.70, 0.80, '#1A8AAA'), (0.80, 0.90, '#C8A84B'), (0.90, 1.0, '#C84850'),
    ]
    for lo, hi, color in zone_bounds:
        fig.add_hrect(y0=lo*fcmax, y1=hi*fcmax, fillcolor=color, opacity=0.08, line_width=0)
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['hr'],
        mode='lines', line=dict(color='#C84850', width=1.5), name='FC',
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
    v = df['velocity'].to_numpy()
    pace = np.where(v > 0.3, 1000 / v / 60, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['hr'],
        mode='lines', line=dict(color='#C84850', width=1.5),
        name='FC', yaxis='y1',
    ))
    fig.add_trace(go.Scatter(
        x=dist_km, y=pace,
        mode='lines', line=dict(color='#C8A84B', width=1.5),
        name='Allure', yaxis='y2',
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


def chart_quartiles(quartiles: dict, decay_mode: str = 'Q4/Q1') -> go.Figure:
    labels = list(quartiles.keys())
    values = [round(v, 4) if not _isnan(v) else 0 for v in quartiles.values()]
    # C-1 Sprint 8 : Q1 amber si decay_mode Q4/Qmax (SCI-8) — Q1 exclu du ratio, signal visuel cohérent
    _q1_color = '#C8A84B' if decay_mode == 'Q4/Qmax' else '#41C8E8'
    colors_bar = [
        _q1_color if i == 0 else ('#C84850' if i == 3 else '#1A3A4A')
        for i in range(4)
    ]
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
        marker_color=['#41C8E8', '#1A8AAA', '#1A5060', '#1A3A4A'],
        text=[f"{v:.1f} km" for v in dist_by_bin.values],
        textposition="outside", textfont=dict(color="#4A6070", size=10),
    ))
    fig.update_layout(**_layout(height=240))
    return fig


def chart_gap_profile(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    # v3.3 : vectorisé numpy
    df2['gap'] = gap_correction_vec(df2['velocity'].to_numpy(), df2['grade'].to_numpy())
    dist_km = df2['distance'] / 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df2['gap'],
        mode='lines', line=dict(color='#41C8E8', width=1), name='GAP',
    ))
    df2['gap_smooth'] = df2['gap'].rolling(20, center=True, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=dist_km, y=df2['gap_smooth'],
        mode='lines', line=dict(color='#C8A84B', width=2), name='Tendance',
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
    # v3.1 : zones optimales recalculées en SPM total
    fig.add_hrect(y0=170, y1=200, fillcolor='rgba(65,200,232,0.05)', line_width=0)
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['cadence'],
        mode='lines', line=dict(color='#41C8E8', width=1.2), name='Cadence',
    ))
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
        marker_color=[
            '#41C8E8' if i == 0 else '#C84850' if i == len(values)-1 else '#1A5060'
            for i in range(len(values))
        ],
        text=[f"{v:.3f}" for v in values],
        textposition="outside", textfont=dict(color="#4A6070", size=10),
    ))
    fig.update_layout(**_layout(height=220, yaxis_title="EF"))
    return fig


# ══════════════════════════════════════════════════════════════════
# PDF GENERATOR — v4.0 ReportLab  (VERTEX Claude Design)
# ══════════════════════════════════════════════════════════════════

def clean(text: str) -> str:
    return _safe_str(text)


def _safe_str(s) -> str:
    """Encode vers Latin-1 (polices Helvetica standard ReportLab)."""
    if s is None:
        return ""
    s = str(s)
    for old, new in (
        ('\u2014', '--'), ('\u2013', '-'), ('\u2019', "'"), ('\u2192', '>'),
        ('\u25b2', '^'), ('\u25c6', '*'), ('\u25a0', '='), ('\u25cf', 'o'),
        ('\u00b7', '.'), ('\u00d7', 'x'),
    ):
        s = s.replace(old, new)
    return s.encode('latin-1', errors='replace').decode('latin-1')


def _wrap_text(text: str, max_chars: int) -> list:
    """Word-wrap simple."""
    words = _safe_str(text).split()
    lines, cur = [], ""
    for w in words:
        candidate = (cur + " " + w).strip()
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _get_verdict_code(label: str) -> str:
    mapping = {
        'GESTION MAITRISEE': 'V1', 'SORTIE SOLIDE': 'V2',
        'BONNE SORTIE': 'V3',      'PERFORMANCE CORRECTE': 'V4',
        'MARGE DE PROGRESSION': 'V5', 'EFFONDREMENT TOTAL': 'V5-C',
        'ANALYSE INCOMPLETE': 'INSUF', 'DONNEES INSUFFISANTES': 'INSUF',
        'INSUFFICIENT': 'INSUF',
    }
    # Normalise accents basiques avant lookup
    n = label.upper()
    for a, b in (('É','E'),('È','E'),('Ê','E'),('À','A'),('Â','A'),('Î','I'),('Ô','O')):
        n = n.replace(a, b)
    return mapping.get(n, 'V?')


def generate_pdf_reportlab(report_data: dict) -> bytes:
    """PDF v6 — ReportLab canvas — VERTEX Design System."""
    from io import BytesIO
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os

    PW, PH = A4
    MX = 24
    CW = PW - 2 * MX

    # ── palette ────────────────────────────────────────────────────
    BG_VOID       = HexColor('#080E14')
    BG_PANEL      = HexColor('#0D1520')
    BG_PANEL_WARM = HexColor('#0D1620')
    BORDER_LINE   = HexColor('#152030')
    BORDER_MUTED  = HexColor('#1A3040')
    FG_PRIMARY    = HexColor('#FFFFFF')
    FG_BODY       = HexColor('#C8D4DC')
    FG_DIM        = HexColor('#8899AA')
    FG_MUTE       = HexColor('#7A9AAA')
    FG_GHOST      = HexColor('#4A6070')
    FG_HUD        = HexColor('#3A5060')
    FG_WHISPER    = HexColor('#2A4050')
    FG_TRACE      = HexColor('#1A2A35')
    CYAN          = HexColor('#41C8E8')
    WARN          = HexColor('#C8A84B')
    CRIT          = HexColor('#C84850')

    # ── font loading — graceful fallback to Helvetica ──────────────
    FONT_BLACK      = 'Helvetica-Bold'
    FONT_BOLD       = 'Helvetica-Bold'
    FONT_SEMI       = 'Helvetica-Bold'
    FONT_MONO       = 'Helvetica'
    FONT_MONO_LIGHT = 'Helvetica'
    FONT_MONO_MED   = 'Helvetica'

    _assets = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts')
    for _fname, _rname in [
        ('BarlowCondensed-Black.ttf',    'BarlowBlack'),
        ('BarlowCondensed-Bold.ttf',     'BarlowBold'),
        ('BarlowCondensed-SemiBold.ttf', 'BarlowSemi'),
        ('DMMono-Regular.ttf',           'DMMono'),
        ('DMMono-Light.ttf',             'DMMono-Light'),
        ('DMMono-Medium.ttf',            'DMMono-Med'),
    ]:
        _fpath = os.path.join(_assets, _fname)
        if os.path.exists(_fpath):
            try:
                pdfmetrics.registerFont(TTFont(_rname, _fpath))
                if   _rname == 'BarlowBlack':  FONT_BLACK      = _rname
                elif _rname == 'BarlowBold':   FONT_BOLD       = _rname
                elif _rname == 'BarlowSemi':   FONT_SEMI       = _rname
                elif _rname == 'DMMono':       FONT_MONO       = _rname
                elif _rname == 'DMMono-Light': FONT_MONO_LIGHT = _rname
                elif _rname == 'DMMono-Med':   FONT_MONO_MED   = _rname
            except Exception:
                pass

    HERO_SIZE = 90 if FONT_BLACK == 'BarlowBlack' else 70
    H2_SIZE   = 18 if FONT_BOLD  == 'BarlowBold'  else 14
    H3_SIZE   = 16 if FONT_BOLD  == 'BarlowBold'  else 12
    MV_SIZE   = 22 if FONT_BOLD  == 'BarlowBold'  else 18

    # ── verdict rail color (unchanged) ─────────────────────────────
    def _verdict_rail_color(label: str):
        l = label.upper()
        if any(x in l for x in ['V5', 'V6', 'V7', 'COLLAPSE', 'INSUFFISANT']):
            return CRIT
        if any(x in l for x in ['V3', 'V4', 'DRIFT', 'FRAGILE']):
            return WARN
        return CYAN

    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    c.setPageCompression(0)

    # ── helpers (unchanged signatures) ────────────────────────────
    def Y(y_top): return PH - y_top

    def rfill(x, y_top, w, h, color):
        c.setFillColor(color)
        c.rect(x, Y(y_top + h), w, h, fill=1, stroke=0)

    def hline(y_top, x1, x2, color, lw=0.5):
        c.setStrokeColor(color); c.setLineWidth(lw)
        c.line(x1, Y(y_top), x2, Y(y_top))

    def vline(x, y_top, h, color, lw=0.5):
        c.setStrokeColor(color); c.setLineWidth(lw)
        c.line(x, Y(y_top), x, Y(y_top + h))

    def txt(x, y_top, s, font=None, size=8, color=FG_PRIMARY, align='left'):
        c.setFont(font or FONT_MONO, size)
        c.setFillColor(color)
        s = _safe_str(s)
        if align == 'center': c.drawCentredString(x, Y(y_top), s)
        elif align == 'right': c.drawRightString(x, Y(y_top), s)
        else: c.drawString(x, Y(y_top), s)

    def tw(s, font, size): return c.stringWidth(_safe_str(s), font or FONT_MONO, size)

    def mini_bar(x, y_top, w, h, pct, fg=None, bg=None):
        if fg is None: fg = CYAN
        if bg is None: bg = BORDER_LINE
        rfill(x, y_top, w, h, bg)
        rfill(x, y_top, max(1, w * max(0.0, min(pct, 1.0))), h, fg)

    def tri_up(cx, y_top, s, color):
        c.setFillColor(color)
        p = c.beginPath()
        p.moveTo(cx - s, Y(y_top + s)); p.lineTo(cx + s, Y(y_top + s)); p.lineTo(cx, Y(y_top))
        p.close(); c.drawPath(p, fill=1, stroke=0)

    def diamond(cx, cy_top, s, color):
        c.setFillColor(color)
        p = c.beginPath()
        p.moveTo(cx, Y(cy_top - s)); p.lineTo(cx + s, Y(cy_top))
        p.lineTo(cx, Y(cy_top + s)); p.lineTo(cx - s, Y(cy_top))
        p.close(); c.drawPath(p, fill=1, stroke=0)

    def v_logo(x, y_top, sc=1.0):
        c.setFillColor(CYAN)
        p = c.beginPath()
        p.moveTo(x,          Y(y_top));         p.lineTo(x + 7*sc,  Y(y_top + 12*sc))
        p.lineTo(x + 10*sc,  Y(y_top +  8*sc)); p.lineTo(x + 7*sc,  Y(y_top + 16*sc))
        p.lineTo(x +  4*sc,  Y(y_top +  8*sc))
        p.close(); c.drawPath(p, fill=1, stroke=0)

    # ── data extraction ────────────────────────────────────────────
    athlete    = _safe_str(report_data.get('athlete_name', 'Athlete'))
    race_name  = _safe_str(report_data.get('race_name', 'Course')).upper()
    race_date  = _safe_str(report_data.get('race_date', ''))
    dist_km    = float(report_data.get('distance_km') or 0)
    elev_gain  = int(report_data.get('elevation_gain') or 0)
    total_time = _safe_str(report_data.get('total_time', '--:--'))
    adj_pace   = _safe_str(report_data.get('adjusted_pace', '--:--'))
    avg_hr     = int(report_data.get('avg_hr') or 0)
    fat_idx    = float(report_data.get('fatigue_index') or 0)
    score      = int(report_data.get('score') or 0)
    pattern    = _safe_str(report_data.get('pattern', 'STABLE')).upper()
    prof_type  = _safe_str(report_data.get('profile_type', 'MIXED')).upper()
    verd_label = _safe_str(report_data.get('verdict_label', '')).upper()
    verd_text  = _safe_str(report_data.get('verdict_text', ''))
    pat_title  = _safe_str(report_data.get('pattern_title', ''))
    pat_text   = _safe_str(report_data.get('pattern_text', ''))
    ins_title  = _safe_str(report_data.get('insight_title', ''))
    ins_text   = _safe_str(report_data.get('insight_text', ''))
    prg_title  = _safe_str(report_data.get('progression_title', ''))
    prg_text   = _safe_str(report_data.get('progression_text', ''))
    quartiles  = report_data.get('quartiles') or []
    elev_prof  = report_data.get('elevation_profile') or []
    gpx_pts    = int(report_data.get('gpx_point_count') or 0)
    report_id  = _safe_str(report_data.get('report_id', 'RPT-0000-0000-XX'))
    version    = _safe_str(report_data.get('version', 'v3.5'))

    MONTHS = ['','JANVIER','FEVRIER','MARS','AVRIL','MAI','JUIN',
              'JUILLET','AOUT','SEPTEMBRE','OCTOBRE','NOVEMBRE','DECEMBRE']
    try:
        from datetime import datetime as _dt
        _d = _dt.strptime(race_date, '%Y-%m-%d')
        race_date_str = f"{_d.day} {MONTHS[_d.month]} {_d.year}"
    except Exception:
        race_date_str = race_date.upper() if race_date else ''

    elev_str   = f"{elev_gain:,}".replace(',', ' ')
    fat_str    = ("+" if fat_idx >= 0 else "") + f"{fat_idx:.1f}%"
    _score_col = CYAN if score >= 80 else WARN

    if not pat_text:
        _pat_defaults = {
            'STABLE':         "Effort regulier. Aucun signal de decrochage physiologique detecte.",
            'DRIFT-CARDIO':   "Derive cardiaque progressive detectee. Effort soutenu au-dela du seuil.",
            'DRIFT-NEURO':    "Derive neuromusculaire. Fatigue musculaire dominante sur la seconde moitie.",
            'COLLAPSE':       "Effondrement physiologique. Effort non soutenable sur la duree totale.",
            'NEGATIVE_SPLIT': "Progression de l effort. Seconde moitie plus rapide, gestion optimale.",
        }
        pat_text = _pat_defaults.get(
            pattern,
            "Analyse du pattern d effort cardiaque et metabolique sur la course."
        )

    # ════════════════════════════════════════════════════════════════
    # BACKGROUND
    # ════════════════════════════════════════════════════════════════
    rfill(0, 0, PW, PH, BG_VOID)

    # ════════════════════════════════════════════════════════════════
    # [1] HEADER  y=0, h=60
    # ════════════════════════════════════════════════════════════════
    rfill(0, 0, PW, 60, BG_PANEL)
    hline(60, 0, PW, BORDER_LINE, 1.0)

    C1w, C3w = 132, 132
    C2w = CW - C1w - C3w
    C1x = MX
    C2x = MX + C1w
    C3x = MX + C1w + C2w
    vline(C2x, 8, 52, BORDER_LINE, 0.5)
    vline(C3x, 8, 52, BORDER_LINE, 0.5)

    # Col 1 — logo + brand
    v_logo(C1x + 6, 12, sc=1.8)
    txt(C1x + 36, 32, "VERTEX",             font=FONT_BOLD, size=18, color=FG_PRIMARY)
    txt(C1x + 36, 46, "PERFORMANCE INTEL.", font=FONT_MONO, size=6,  color=FG_GHOST)

    # Col 2 — centred race info
    c2_cx = C2x + C2w / 2
    txt(c2_cx, 16, "// RAPPORT D'ANALYSE //", font=FONT_MONO, size=6, color=FG_GHOST,   align='center')
    rn = race_name[:34] if len(race_name) > 34 else race_name
    txt(c2_cx, 34, rn, font=FONT_BOLD, size=14,                       color=FG_PRIMARY, align='center')
    info_line = f"{athlete}  -  {race_date_str}" if race_date_str else athlete
    txt(c2_cx, 48, info_line, font=FONT_MONO, size=7.5,               color=FG_BODY,    align='center')

    # Col 3 — report meta + LOCAL badge
    txt(C3x + 6, 15, f"// {report_id} //", font=FONT_MONO, size=6, color=FG_GHOST)
    txt(C3x + 6, 25, version,              font=FONT_MONO, size=6, color=FG_GHOST)
    _badge_w = tw("LOCAL", FONT_MONO, 6) + 10
    c.setStrokeColor(CYAN); c.setLineWidth(0.7)
    c.rect(C3x + 6, Y(40), _badge_w, 11, fill=0, stroke=1)
    txt(C3x + 11, 37, "LOCAL", font=FONT_MONO, size=6, color=CYAN)
    if gpx_pts:
        txt(C3x + 6, 52, f"GPX  {gpx_pts:,} pts".replace(',', ' '), font=FONT_MONO, size=6, color=FG_GHOST)

    # ════════════════════════════════════════════════════════════════
    # [2] STAT STRIP  y=62, h=38
    # ════════════════════════════════════════════════════════════════
    rfill(0, 62, PW, 38, BG_PANEL)
    hline(100, 0, PW, BORDER_LINE, 1.0)

    _Sw = CW / 3
    for i, (lbl, val, unit) in enumerate([
        ("DISTANCE",         f"{dist_km:.1f}", "km"),
        ("DENIVELE POSITIF", elev_str,          "m D+"),
        ("TEMPS",            total_time,         ""),
    ]):
        sx = MX + i * _Sw
        if i > 0:
            vline(sx, 68, 26, BORDER_LINE, 0.5)
        txt(sx + 8, 74, lbl, font=FONT_MONO, size=6, color=FG_WHISPER)
        txt(sx + 8, 94, val, font=FONT_BOLD, size=20, color=CYAN)
        if unit:
            txt(sx + 8 + tw(val, FONT_BOLD, 20) + 3, 92, unit, font=FONT_MONO, size=8, color=FG_DIM)

    # ════════════════════════════════════════════════════════════════
    # [3] HERO SCORE  y=102, h=105
    # ════════════════════════════════════════════════════════════════
    rfill(0, 102, PW, 105, BG_PANEL_WARM)
    hline(102, 0, PW, CYAN, 2.0)

    HERO_COL_W = 190
    txt(MX + 10, 116, "SCORE VERTEX", font=FONT_MONO, size=6, color=FG_GHOST)

    score_str = str(score) if score else "--"
    _score_baseline = 195 if HERO_SIZE == 90 else 183
    c.setFont(FONT_BLACK, HERO_SIZE)
    c.setFillColor(_score_col)
    c.drawString(MX + 10, Y(_score_baseline), score_str)

    # 10-segment progress bar below score (width 160pt, height 4pt)
    _n_filled = round(score / 10) if score else 0
    _sw, _sg  = 14, 2
    for si in range(10):
        rfill(MX + 10 + si * (_sw + _sg), _score_baseline + 5, _sw, 4,
              _score_col if si < _n_filled else BG_VOID)

    # right column — badges + verdict
    RCx = MX + HERO_COL_W
    vline(RCx, 108, 93, BORDER_LINE, 0.5)

    bx = RCx + 8
    for btext, bfcol in [
        (_get_verdict_code(verd_label), _verdict_rail_color(verd_label)),
        (pattern,                        FG_GHOST),
        (prof_type,                      FG_GHOST),
    ]:
        bw_i = tw(btext, FONT_BOLD, 7) + 10
        c.setStrokeColor(bfcol); c.setLineWidth(0.7)
        c.rect(bx, Y(130), bw_i, 12, fill=0, stroke=1)
        txt(bx + 5, 127, btext, font=FONT_BOLD, size=7, color=bfcol)
        bx += bw_i + 5

    txt(RCx + 8, 149, verd_label[:34], font=FONT_BOLD, size=H3_SIZE, color=FG_PRIMARY)
    if verd_text:
        for i, line in enumerate(_wrap_text(verd_text, 52)[:3]):
            txt(RCx + 8, 167 + i * 12, line, font=FONT_MONO, size=8, color=FG_BODY)

    # ════════════════════════════════════════════════════════════════
    # [4] 4 METRIC CARDS  y=209, h=58
    # ════════════════════════════════════════════════════════════════
    _cgap = 8
    _cw   = (CW - 3 * _cgap) / 4
    _cards = [
        ("ALLURE AJUSTEE RELIEF", adj_pace,                       "/km", "Allure normalisee relief",  CYAN),
        ("FC MOYENNE",            str(avg_hr) if avg_hr else "--", "bpm", "Effort cardio moyen",       CYAN),
        ("PROFIL",                prof_type,                       "",    "",                          CYAN),
        ("INDICE DE FATIGUE",     fat_str,                         "",    "Derive cardio/metabolique", WARN),
    ]
    for i, (lbl, val, unit, sub, col) in enumerate(_cards):
        cx = MX + i * (_cw + _cgap)
        rfill(cx, 209, _cw, 58, BG_PANEL)
        rfill(cx, 209, _cw, 2,  col)
        txt(cx + 6, 221, lbl, font=FONT_MONO, size=6,      color=FG_WHISPER)
        txt(cx + 6, 248, val, font=FONT_BOLD, size=MV_SIZE, color=col)
        if unit:
            txt(cx + 6 + tw(val, FONT_BOLD, MV_SIZE) + 3, 246, unit, font=FONT_MONO, size=7, color=FG_DIM)
        if sub:
            txt(cx + 6, 260, sub, font=FONT_MONO, size=5.5, color=FG_GHOST)

    # ════════════════════════════════════════════════════════════════
    # [5] QUARTILES TABLE  y=269, h=110
    # ════════════════════════════════════════════════════════════════
    QC = [50, 80, 60, 140]
    QC.append(CW - sum(QC))   # ETAT col fills remainder
    QHH = 16
    QRH = (110 - QHH) / 4    # ≈ 23.5

    rfill(MX, 269, CW, QHH, BG_VOID)
    qcx = MX
    for ch, cw_q in zip(["QUARTILE", "ALLURE", "FC", "PROGRESSION", "ETAT"], QC):
        txt(qcx + 4, 280, ch, font=FONT_MONO, size=6, color=FG_WHISPER)
        qcx += cw_q

    for ri in range(4):
        ry      = 269 + QHH + ri * QRH
        row_mid = ry + QRH * 0.65
        rfill(MX, ry, CW, QRH, BG_PANEL if ri % 2 == 0 else BG_VOID)
        hline(ry + QRH, MX, MX + CW, BORDER_LINE, 0.3)
        q   = quartiles[ri] if ri < len(quartiles) else {}
        qcx = MX

        txt(qcx + 4, row_mid, q.get('label', f'Q{ri+1}'), font=FONT_BOLD, size=9, color=FG_PRIMARY)
        qcx += QC[0]

        txt(qcx + 4, row_mid, _safe_str(q.get('pace', '--')), font=FONT_MONO, size=7, color=CYAN)
        qcx += QC[1]

        hr_v = q.get('hr')
        txt(qcx + 4, row_mid, str(int(hr_v)) if hr_v else "--", font=FONT_MONO, size=7, color=FG_MUTE)
        qcx += QC[2]

        delta   = float(q.get('delta_pct') or 0)
        bar_pct = (delta + 15) / 20.0 if ri > 0 else 0.75
        mini_bar(qcx + 4, ry + QRH * 0.35, 60, 4, max(0.0, min(bar_pct, 1.0)))
        dt_s = "baseline" if ri == 0 else (f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%")
        txt(qcx + 68, row_mid, dt_s, font=FONT_MONO, size=6.5, color=FG_DIM)
        qcx += QC[3]

        st_raw = _safe_str(q.get('state', 'TENU')).upper()
        if any(x in st_raw for x in ('SOLIDE', 'PROPRE', 'LANCEMENT')):
            st_col = CYAN
        elif 'DECROCHAGE' in st_raw:
            st_col = CRIT
        else:
            st_col = WARN
        txt(qcx + 5, row_mid, st_raw, font=FONT_MONO, size=6, color=st_col)

    c.setStrokeColor(BORDER_LINE); c.setLineWidth(1.0)
    c.rect(MX, Y(379), CW, 110, fill=0, stroke=1)
    hline(379, MX, MX + CW, BORDER_LINE, 1.0)

    # ════════════════════════════════════════════════════════════════
    # [6] PATTERN BLOCK  y=381, h=52
    # ════════════════════════════════════════════════════════════════
    rfill(MX, 381, CW, 52, BG_PANEL_WARM)
    rfill(MX, 381, 3,  52, CYAN)

    PAT_LW = 130
    txt(MX + 12, 399, pattern, font=FONT_BOLD, size=H2_SIZE, color=CYAN)
    vline(MX + PAT_LW, 385, 44, BORDER_LINE, 0.5)
    Ptx = MX + PAT_LW + 10

    if pat_title:
        txt(Ptx, 396, pat_title.upper()[:52], font=FONT_BOLD, size=10, color=FG_PRIMARY)
    for i, line in enumerate(_wrap_text(pat_text, 58)[:3]):
        txt(Ptx, (408 if pat_title else 396) + i * 10, line, font=FONT_MONO, size=7, color=FG_BODY)

    # ════════════════════════════════════════════════════════════════
    # [7] RECOMMANDATIONS  y=435, h=72
    # ════════════════════════════════════════════════════════════════
    _rw = (CW - 8) / 2

    # left — ANALYSE
    rfill(MX, 435, _rw, 72, BG_PANEL)
    rfill(MX, 435, 3,   72, CYAN)
    txt(MX + 8, 446, "ANALYSE", font=FONT_MONO, size=5, color=FG_GHOST)
    if ins_title:
        txt(MX + 8, 458, ins_title.upper()[:40], font=FONT_BOLD, size=10, color=FG_PRIMARY)
        for i, line in enumerate(_wrap_text(ins_text, 38)[:4]):
            txt(MX + 8, 472 + i * 9, line, font=FONT_MONO, size=7, color=FG_BODY)
    else:
        txt(MX + 8, 460, "Donnees insuffisantes pour cette section.", font=FONT_MONO, size=7, color=FG_MUTE)

    # right — AXE DE PROGRESSION
    R7x = MX + _rw + 8
    rfill(R7x, 435, _rw, 72, BG_PANEL)
    rfill(R7x, 435, 3,   72, WARN)
    txt(R7x + 8, 446, "AXE DE PROGRESSION", font=FONT_MONO, size=5, color=FG_GHOST)
    if prg_title:
        txt(R7x + 8, 458, prg_title.upper()[:40], font=FONT_BOLD, size=10, color=FG_PRIMARY)
        for i, line in enumerate(_wrap_text(prg_text, 38)[:4]):
            txt(R7x + 8, 472 + i * 9, line, font=FONT_MONO, size=7, color=FG_BODY)
    else:
        txt(R7x + 8, 460, "Donnees insuffisantes pour cette section.", font=FONT_MONO, size=7, color=FG_MUTE)

    # ════════════════════════════════════════════════════════════════
    # [8] ELEVATION STRIP  y=509, h=52
    # ════════════════════════════════════════════════════════════════
    rfill(MX, 509, CW, 52, BG_PANEL)
    txt(MX + 4, 517, "// PROFIL ELEVATION //", font=FONT_MONO, size=6, color=FG_GHOST)

    G8y, G8h = 520, 38

    if len(elev_prof) > 3:
        n = len(elev_prof)
        # filled area
        c.saveState()
        fp = c.beginPath()
        fp.moveTo(MX, Y(G8y + G8h))
        for i, v in enumerate(elev_prof):
            fp.lineTo(MX + CW * i / (n - 1), Y(G8y + G8h * (1 - max(0.0, min(v, 1.0)))))
        fp.lineTo(MX + CW, Y(G8y + G8h))
        fp.close()
        c.setFillColor(HexColor('#0D3A42'))
        c.drawPath(fp, fill=1, stroke=0)
        c.restoreState()
        # stroke line
        c.saveState()
        c.setStrokeColor(CYAN); c.setLineWidth(1.5)
        lp = c.beginPath()
        for i, v in enumerate(elev_prof):
            px = MX + CW * i / (n - 1)
            py = G8y + G8h * (1 - max(0.0, min(v, 1.0)))
            if i == 0: lp.moveTo(px, Y(py))
            else:      lp.lineTo(px, Y(py))
        c.drawPath(lp, fill=0, stroke=1)
        c.restoreState()
        # Q2/Q3/Q4 dashed markers
        for qi in range(1, 4):
            qx = MX + CW * qi / 4
            c.saveState()
            c.setStrokeColor(WARN); c.setLineWidth(0.5); c.setDash([2, 3])
            c.line(qx, Y(G8y + 2), qx, Y(G8y + G8h))
            c.restoreState()
            txt(qx, G8y + 8, f"Q{qi+1}", font=FONT_MONO, size=6, color=FG_DIM, align='center')
    else:
        txt(MX + CW / 2, G8y + G8h / 2 + 4, "Profil non disponible",
            font=FONT_MONO, size=7.5, color=FG_DIM, align='center')

    # ════════════════════════════════════════════════════════════════
    # [9] FOOTER  absolute bottom, h=42
    # ════════════════════════════════════════════════════════════════
    F9y = PH - 42
    hline(F9y, 0, PW, BORDER_LINE, 1.0)
    txt(MX, F9y + 10, "// DISCLAIMER MEDICAL //", font=FONT_MONO, size=6, color=FG_DIM)
    disc = ("Score experimental - modele physiologique Minetti 2002 + decouplage cardiaque. "
            "Non valide cliniquement. Usage personnel et pedagogique uniquement. "
            "Ne remplace pas un suivi medical ou un test d effort encadre.")
    for i, dl in enumerate(_wrap_text(disc, 82)[:3]):
        txt(MX, F9y + 19 + i * 7, dl, font=FONT_MONO_LIGHT, size=5.5, color=FG_TRACE)
    v_logo(PW - MX - 56, F9y + 6, sc=1.1)
    txt(PW - MX - 44, F9y + 16, "VERTEX",            font=FONT_BOLD, size=8,   color=FG_PRIMARY)
    txt(PW - MX - 52, F9y + 28, "(c) 2026  BSL 1.1", font=FONT_MONO, size=5.5, color=FG_GHOST)

    c.showPage()
    c.save()
    return buf.getvalue()


def generate_pdf(info, fi, flat_v, profile, grade_df,
                 zones, drift, cad_analysis, splits, recs,
                 fcmax, perf=None, verdict=None, email="") -> bytes:
    """
    Bridge → construit report_data depuis les anciens paramètres et appelle generate_pdf_reportlab.
    Signature inchangée pour compatibilité app.py et test_engine.py.
    """

    # ── Bridge : construit report_data depuis les anciens params ────
    from datetime import datetime as _dt

    dist_km = float(info.get('distance_km') or 0)
    total_s = float(info.get('total_time_s') or 0)
    h_t = int(total_s // 3600)
    m_t = int((total_s % 3600) // 60)
    s_t = int(total_s % 60)
    total_str = f"{h_t}:{m_t:02d}:{s_t:02d}"

    score = int((perf.get('score') if perf else None) or 0)

    # C-2 Sprint 8 : utiliser decay_pct_corrected si correction appliquée (SCI-3)
    _corr_applied = (fi.get('correction_applied', False) if fi else False)
    dp = (fi.get('decay_pct_corrected') if _corr_applied else fi.get('decay_pct')) if fi else None
    fat_idx = float(dp) if (dp is not None and not _isnan(dp)) else 0.0

    verd_label = ''
    verd_text  = ''
    action_line = ''
    if verdict:
        verd_label   = str(verdict.get('label', '')).upper()
        verd_text    = str(verdict.get('sub', ''))
        action_line  = str(verdict.get('action_line', ''))
    if not verd_label:
        if score >= 90:   verd_label = "GESTION MAITRISEE"
        elif score >= 80: verd_label = "SORTIE SOLIDE"
        elif score >= 70: verd_label = "BONNE SORTIE"
        elif score >= 60: verd_label = "PERFORMANCE CORRECTE"
        else:             verd_label = "MARGE DE PROGRESSION"

    pattern = str((drift.get('pattern') if drift else None) or 'STABLE')

    # C-3 Sprint 8 : seuils adaptatifs SCI-4 — cohérence avec get_drift_ef_threshold()
    # Court <2h : -4% / Long 2-4h : -6% / Ultra >4h : -9%
    _d_pct = float(drift.get('drift_pct') or 0) if drift else 0.0
    if not _isnan(_d_pct):
        _drift_thr = float(drift.get('drift_ef_thr', -4.0) or -4.0) if drift else -4.0
        _qualif = (
            "(normal)"          if _d_pct > _drift_thr / 2 else
            "(fatigue moderee)" if _d_pct > _drift_thr      else
            "(signal fort)"
        )
    else:
        _qualif = ""

    # Quartiles
    qraw = (fi.get('quartiles') if fi else {}) or {}
    qkeys = list(qraw.keys())
    q_list = []
    for i, (qk, qv) in enumerate(qraw.items()):
        if i == 0:
            delta, state = 0.0, 'LANCEMENT PROPRE'
        else:
            base = qraw.get(qkeys[0], 1) or 1
            delta = ((qv - base) / base * 100) if (qv and not _isnan(qv) and base) else 0.0
            if delta < -8:  state = 'DECROCHAGE'
            elif i == 3:    state = 'FINISH SOLIDE'
            else:           state = 'TENU'
        pace_s = v_to_pace(qv) if (qv and not _isnan(qv)) else '--:--'
        q_list.append({'label': qk, 'pace': pace_s, 'hr': None,
                       'delta_pct': delta, 'state': state})

    # Profil élévation (grade_df n'a pas de colonne elevation → vide)
    elev_prof = []
    if grade_df is not None and not grade_df.empty and 'elevation' in grade_df.columns:
        elev_raw = grade_df['elevation'].dropna().tolist()
        if elev_raw:
            emin, emax = min(elev_raw), max(elev_raw)
            rng = (emax - emin) or 1
            step = max(1, len(elev_raw) // 200)
            elev_prof = [(v - emin) / rng for v in elev_raw[::step]]

    # Recs → insight + progression
    ins_title = ins_text = prg_title = prg_text = ''
    if recs:
        if len(recs) > 0:
            ins_title = str(recs[0].get('title', ''))
            ins_text  = str(recs[0].get('body',  ''))
        if len(recs) > 1:
            prg_title = str(recs[1].get('title', ''))
            prg_text  = str(recs[1].get('body',  ''))
    # action_line intégré dans progression_text (visible dans PDF → test M6)
    if action_line and len(action_line) > 5:
        prg_text = (action_line + "  " + prg_text).strip()

    now = _dt.now()
    initials = ''.join(w[0].upper() for w in str(info.get('name') or 'XX').split()[:2])
    report_id = f"RPT-{now.year}-{now.month:02d}{now.day:02d}-{initials}"

    report_data = {
        'athlete_name':      str(info.get('athlete_name', info.get('name', 'Athlete'))),
        'race_name':         str(info.get('name', 'Course')),
        'race_date':         str(info.get('date', now.strftime('%Y-%m-%d'))),
        'distance_km':       dist_km,
        'elevation_gain':    int(info.get('elevation_gain') or 0),
        'total_time':        total_str,
        'adjusted_pace':     v_to_pace(flat_v) if flat_v else '--:--',
        'avg_hr':            int(info.get('hr_mean') or 0),
        'fatigue_index':     fat_idx,
        'score':             score,
        'pattern':           pattern,
        'profile_type':      str(profile or 'MIXED'),
        'verdict_label':     verd_label,
        'verdict_text':      verd_text,
        'pattern_title':     '',
        'pattern_text':      _qualif,
        'insight_title':     ins_title,
        'insight_text':      ins_text,
        'progression_title': prg_title,
        'progression_text':  prg_text,
        'quartiles':         q_list,
        'elevation_profile': elev_prof,
        'gpx_point_count':   int(info.get('gpx_point_count') or 0),
        'report_id':         report_id,
        'version':           'v3.5  BSL 1.1',
    }
    return generate_pdf_reportlab(report_data)
