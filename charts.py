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
        marker_color=['#41C8E8', '#1A8AAA', '#1A5060', '#0D2A34'],
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
    """
    PDF v4 — ReportLab canvas — VERTEX Claude Design.
    9 sections sur page A4 portrait fond DEEP_NAVY.
    """
    from io import BytesIO
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os

    PW, PH = A4          # 595.27 × 841.89
    MX = 24              # marges latérales
    CW = PW - 2 * MX    # largeur contenu ≈ 547

    DEEP_NAVY     = HexColor('#080E14')
    PANEL_DARK    = HexColor('#0D1520')
    CYAN          = HexColor('#41C8E8')
    GOLD          = HexColor('#C8A84B')
    WHITE         = HexColor('#FFFFFF')
    GREY_LIGHT    = HexColor('#8899AA')
    BORDER_SUBTLE = HexColor('#1A2535')
    CYAN_BLEND    = HexColor('#0E2129')
    GOLD_BLEND    = HexColor('#131210')
    ROW_ALT       = HexColor('#0F1B28')

    BF = 'Helvetica-Bold'
    RF = 'Helvetica'
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts')
    for fname, rname, use_bold in [
        ('BarlowCondensed-Bold.ttf', 'BarlowBold', True),
        ('DMSans-Regular.ttf',       'DMSans',     False),
    ]:
        fpath = os.path.join(assets_dir, fname)
        if os.path.exists(fpath):
            try:
                pdfmetrics.registerFont(TTFont(rname, fpath))
                if use_bold:
                    BF = rname
                else:
                    RF = rname
            except Exception:
                pass

    buf = BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=A4)
    c.setPageCompression(0)   # texte lisible dans les bytes bruts (tests M3/M6)

    # ── helpers coords ─────────────────────────────────────────────
    def Y(y_top):
        return PH - y_top

    def rfill(x, y_top, w, h, color):
        c.setFillColor(color)
        c.rect(x, Y(y_top + h), w, h, fill=1, stroke=0)

    def hline(y_top, x1, x2, color, lw=0.5):
        c.setStrokeColor(color)
        c.setLineWidth(lw)
        c.line(x1, Y(y_top), x2, Y(y_top))

    def vline(x, y_top, h, color, lw=0.5):
        c.setStrokeColor(color)
        c.setLineWidth(lw)
        c.line(x, Y(y_top), x, Y(y_top + h))

    def txt(x, y_top, s, font=None, size=8, color=WHITE, align='left'):
        c.setFont(font or RF, size)
        c.setFillColor(color)
        s = _safe_str(s)
        if align == 'center':
            c.drawCentredString(x, Y(y_top), s)
        elif align == 'right':
            c.drawRightString(x, Y(y_top), s)
        else:
            c.drawString(x, Y(y_top), s)

    def tw(s, font, size):
        return c.stringWidth(_safe_str(s), font or RF, size)

    def section_hdr(y_top, label):
        rfill(MX, y_top, 3, 12, CYAN)
        txt(MX + 7, y_top + 9.5, label, font=BF, size=7, color=WHITE)

    def mini_bar(x, y_top, w, h, pct, fg=None, bg=BORDER_SUBTLE):
        if fg is None:
            fg = CYAN
        rfill(x, y_top, w, h, bg)
        rfill(x, y_top, max(1, w * max(0, min(pct, 1.0))), h, fg)

    def tri_up(cx, y_top, s, color):
        c.setFillColor(color)
        p = c.beginPath()
        p.moveTo(cx - s, Y(y_top + s)); p.lineTo(cx + s, Y(y_top + s)); p.lineTo(cx, Y(y_top))
        p.close(); c.drawPath(p, fill=1, stroke=0)

    def diamond(cx, cy_top, s, color):
        c.setFillColor(color)
        p = c.beginPath()
        p.moveTo(cx,     Y(cy_top - s)); p.lineTo(cx + s, Y(cy_top))
        p.lineTo(cx,     Y(cy_top + s)); p.lineTo(cx - s, Y(cy_top))
        p.close(); c.drawPath(p, fill=1, stroke=0)

    def v_logo(x, y_top, sc=1.0):
        c.setFillColor(CYAN)
        p = c.beginPath()
        p.moveTo(x,          Y(y_top));         p.lineTo(x + 7*sc,   Y(y_top + 12*sc))
        p.lineTo(x + 10*sc,  Y(y_top +  8*sc)); p.lineTo(x + 7*sc,   Y(y_top + 16*sc))
        p.lineTo(x + 4*sc,   Y(y_top +  8*sc))
        p.close(); c.drawPath(p, fill=1, stroke=0)

    # ── données ────────────────────────────────────────────────────
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
    version    = _safe_str(report_data.get('version', 'v3.5  BSL 1.1'))

    MONTHS = ['','JANVIER','FEVRIER','MARS','AVRIL','MAI','JUIN',
              'JUILLET','AOUT','SEPTEMBRE','OCTOBRE','NOVEMBRE','DECEMBRE']
    try:
        from datetime import datetime as _dt
        _d = _dt.strptime(race_date, '%Y-%m-%d')
        race_date_str = f"{_d.day} {MONTHS[_d.month]} {_d.year}"
    except Exception:
        race_date_str = race_date.upper() if race_date else ''

    elev_str = f"{elev_gain:,}".replace(',', ' ')

    # ════════════════════════════════════════════════════════════════
    # FOND PLEINE PAGE
    # ════════════════════════════════════════════════════════════════
    rfill(0, 0, PW, PH, DEEP_NAVY)

    # ════════════════════════════════════════════════════════════════
    # [1] HEADER  y=20, h=60
    # ════════════════════════════════════════════════════════════════
    H1y, H1h = 20, 60
    rfill(0, H1y, PW, H1h, PANEL_DARK)
    hline(H1y + H1h, 0, PW, BORDER_SUBTLE)

    C1w, C3w = 132, 132
    C2w = CW - C1w - C3w
    C1x, C2x, C3x = MX, MX + C1w, MX + C1w + C2w
    vline(C2x, H1y + 8, H1h - 16, BORDER_SUBTLE)
    vline(C3x, H1y + 8, H1h - 16, BORDER_SUBTLE)

    # Col 1 — logo + wordmark
    v_logo(C1x + 8, H1y + 10, sc=1.8)
    txt(C1x + 36, H1y + 30, "VERTEX",                  font=BF, size=18, color=WHITE)
    txt(C1x + 36, H1y + 44, "PERFORMANCE INTELLIGENCE", font=RF, size=6,  color=GREY_LIGHT)

    # Col 2 — titre course
    txt(C2x + 8, H1y + 13, "// RAPPORT D'ANALYSE //", font=RF, size=6, color=GREY_LIGHT)
    rn = race_name[:36] if len(race_name) > 36 else race_name
    txt(C2x + 8, H1y + 30, rn, font=BF, size=12, color=WHITE)
    tri_up(C2x + 12, H1y + 38, 3, GOLD)
    info_line = f"{athlete}  -  {race_date_str}" if race_date_str else athlete
    txt(C2x + 20, H1y + 45, info_line, font=RF, size=7.5, color=WHITE)

    # Col 3 — métadonnées
    txt(C3x + 6, H1y + 13, f"// {report_id} //", font=RF, size=6,   color=GREY_LIGHT)
    txt(C3x + 6, H1y + 23, version,               font=RF, size=6.5, color=GREY_LIGHT)
    rfill(C3x + 6, H1y + 27, 82, 12, PANEL_DARK)
    c.setFillColor(CYAN); c.setFont(RF, 7)
    c.drawString(C3x + 10, Y(H1y + 36), "o")
    txt(C3x + 19, H1y + 36, "ANALYSE LOCALE", font=RF, size=6, color=CYAN)
    if gpx_pts:
        txt(C3x + 6, H1y + 49, f"GPX  {gpx_pts:,} pts".replace(',', ' '), font=RF, size=6.5, color=GREY_LIGHT)

    # ════════════════════════════════════════════════════════════════
    # [2] BANDEAU METRIQUES  y=82, h=52
    # ════════════════════════════════════════════════════════════════
    B2y, B2h = H1y + H1h + 2, 52
    rfill(0, B2y, PW, B2h, PANEL_DARK)
    hline(B2y + B2h, 0, PW, BORDER_SUBTLE)

    Mw = CW / 3
    for i, (lbl, val, unit) in enumerate([
        ("DISTANCE",         f"{dist_km:.1f}", "km"),
        ("DENIVELE POSITIF", elev_str,          "m D+"),
        ("TEMPS",            total_time,         ""),
    ]):
        mx = MX + i * Mw
        if i > 0:
            vline(mx, B2y + 6, B2h - 12, BORDER_SUBTLE)
        txt(mx + 8, B2y + 13, lbl, font=RF, size=6, color=GREY_LIGHT)
        txt(mx + 8, B2y + 40, val, font=BF, size=28, color=CYAN)
        if unit:
            txt(mx + 8 + tw(val, BF, 28) + 3, B2y + 38, unit, font=RF, size=10, color=GREY_LIGHT)

    # ════════════════════════════════════════════════════════════════
    # [3] RESULTAT GLOBAL  y=136, h=100
    # ════════════════════════════════════════════════════════════════
    B3y, B3h = B2y + B2h + 2, 100
    section_hdr(B3y, "RESULTAT GLOBAL")

    Bly, Blh = B3y + 14, B3h - 14
    rfill(MX, Bly, CW, Blh, PANEL_DARK)

    SCW = int(CW * 0.28)
    vline(MX + SCW, Bly + 6, Blh - 12, BORDER_SUBTLE)
    txt(MX + 10, Bly + 13, "SCORE VERTEX", font=RF, size=6, color=GREY_LIGHT)
    score_str = str(score) if score else "--"
    c.setFont(BF, 46); c.setFillColor(CYAN)
    c.drawString(MX + 10, Y(Bly + 66), score_str)
    txt(MX + 10 + tw(score_str, BF, 46) + 2, Bly + 64, "/100", font=RF, size=10, color=GREY_LIGHT)
    mini_bar(MX + 10, Bly + 72, SCW - 20, 3, score / 100 if score else 0)

    VCx = MX + SCW + 10
    # Badges
    bx = VCx
    for btext, bfont, bcol in [
        (f"VERDICT  {_get_verdict_code(verd_label)}", BF, GOLD),
        (f"PATTERN  {pattern}",                       RF, GREY_LIGHT),
        (f"PROFIL  {prof_type}",                      RF, GREY_LIGHT),
    ]:
        bw = tw(btext, bfont, 7) + 10
        c.setStrokeColor(bcol); c.setLineWidth(0.7)
        c.rect(bx, Y(Bly + 17), bw, 12, fill=0, stroke=1)
        txt(bx + 4, Bly + 13, btext, font=bfont, size=7, color=bcol)
        bx += bw + 5

    VTOP = Bly + 28
    tri_up(VCx + 5, VTOP + 2, 4, GOLD)
    vl_disp = verd_label[:34]
    txt(VCx + 14, VTOP + 8, vl_disp, font=BF, size=15, color=WHITE)
    if verd_text:
        for i, line in enumerate(_wrap_text(verd_text, 56)[:3]):
            txt(VCx, VTOP + 24 + i * 11, line, font=RF, size=8, color=WHITE)

    # ════════════════════════════════════════════════════════════════
    # [4] METRIQUES CLES  y≈238, h=78
    # ════════════════════════════════════════════════════════════════
    B4y, B4h = B3y + B3h + 2, 78
    section_hdr(B4y, "METRIQUES CLES")

    C4y, C4h = B4y + 14, B4h - 14
    Kw = CW / 4
    fat_col = CYAN if abs(fat_idx) < 10 else GOLD
    fat_str = ("+" if fat_idx >= 0 else "") + f"{fat_idx:.0f}"
    cards4 = [
        ("ALLURE AJUSTEE RELIEF", adj_pace,                     "/km",  "Allure normalisee pour le denivele.",           CYAN),
        ("FC MOYENNE",           str(avg_hr) if avg_hr else "--","bpm",  "Effort en zone tempo/seuil.",                  CYAN),
        ("DENIVELE POSITIF",     elev_str,                      "m D+", f"{elev_gain/max(dist_km,0.01):.0f} m/km",      CYAN),
        ("INDICE DE FATIGUE",    fat_str,                        "%",    "Derive contenue" if abs(fat_idx)<10 else "Derive detectee", fat_col),
    ]
    for i, (lbl, val, unit, sub, col) in enumerate(cards4):
        cx = MX + i * Kw
        rfill(cx, C4y, Kw, C4h, PANEL_DARK)
        rfill(cx, C4y, Kw, 2, col)
        if i > 0:
            vline(cx, C4y, C4h, BORDER_SUBTLE, 0.3)
        txt(cx + 5, C4y + 11, lbl, font=RF, size=6, color=GREY_LIGHT)
        txt(cx + 5, C4y + 34, val, font=BF, size=20, color=col)
        txt(cx + 5 + tw(val, BF, 20) + 2, C4y + 32, unit, font=RF, size=8, color=GREY_LIGHT)
        for j, sl in enumerate(_wrap_text(sub, 24)[:2]):
            txt(cx + 5, C4y + 47 + j * 9, sl, font=RF, size=6, color=GREY_LIGHT)

    # ════════════════════════════════════════════════════════════════
    # [5] PROFIL EFFORT Q1->Q4  y≈318, h=104
    # ════════════════════════════════════════════════════════════════
    B5y, B5h = B4y + B4h + 2, 104
    section_hdr(B5y, "PROFIL D'EFFORT  Q1 -> Q4")

    T5y, T5h = B5y + 14, B5h - 14
    QCW = [50, 78, 78, 148, int(CW) - 354]   # total ≈ CW

    HDR_H = 16
    ROW_H = (T5h - HDR_H) / 4

    rfill(MX, T5y, CW, HDR_H, DEEP_NAVY)
    qcx = MX
    for ch, cw_i in zip(["QUARTILE","ALLURE","FC MOY","TENDANCE","ETAT"], QCW):
        txt(qcx + 4, T5y + 11, ch, font=BF, size=6, color=GREY_LIGHT)
        qcx += cw_i

    STATE_CFG = {
        'LANCEMENT PROPRE': (CYAN,       CYAN_BLEND,  '^'),
        'TENU':             (GREY_LIGHT, BORDER_SUBTLE,'='),
        'SOLIDE':           (CYAN,       CYAN_BLEND,  '^'),
        'FINISH SOLIDE':    (GOLD,       GOLD_BLEND,  '^'),
        'DECROCHAGE':       (HexColor('#C84850'), HexColor('#200810'), 'v'),
    }
    for ri in range(4):
        ry = T5y + HDR_H + ri * ROW_H
        rfill(MX, ry, CW, ROW_H, PANEL_DARK if ri % 2 == 0 else ROW_ALT)
        hline(ry + ROW_H, MX, MX + CW, BORDER_SUBTLE, 0.3)
        q = quartiles[ri] if ri < len(quartiles) else {}
        qcx = MX
        my = ry + ROW_H * 0.65

        txt(qcx + 4, my, q.get('label', f'Q{ri+1}'), font=BF, size=10, color=CYAN)
        qcx += QCW[0]
        pace = q.get('pace', '--')
        txt(qcx + 4, my, pace, font=RF, size=9, color=WHITE)
        txt(qcx + 4 + tw(pace, RF, 9) + 2, my - 1, "/km", font=RF, size=6.5, color=GREY_LIGHT)
        qcx += QCW[1]
        hr_s = str(q.get('hr')) if q.get('hr') else "--"
        txt(qcx + 4, my, hr_s, font=RF, size=9, color=WHITE)
        txt(qcx + 4 + tw(hr_s, RF, 9) + 2, my - 1, "bpm", font=RF, size=6.5, color=GREY_LIGHT)
        qcx += QCW[2]
        delta = float(q.get('delta_pct') or 0)
        bar_pct = 0.55 + (delta / 100) if ri > 0 else 0.55
        mini_bar(qcx + 4, ry + ROW_H * 0.3, QCW[3] - 54, 4, max(0.05, min(bar_pct, 0.95)))
        dt_s = "baseline" if ri == 0 else (f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%")
        txt(qcx + QCW[3] - 46, my, dt_s, font=RF, size=6.5, color=GREY_LIGHT)
        qcx += QCW[3]
        st_raw = q.get('state', 'TENU').upper()
        sc_col, sc_bg, sc_icon = STATE_CFG.get(st_raw, (GREY_LIGHT, BORDER_SUBTLE, '='))
        rfill(qcx + 3, ry + ROW_H * 0.12, QCW[4] - 6, ROW_H * 0.76, sc_bg)
        txt(qcx + 7, my, f"{sc_icon} {st_raw}", font=BF, size=6.5, color=sc_col)

    # ════════════════════════════════════════════════════════════════
    # [6] LECTURE PATTERN  y≈424, h=66
    # ════════════════════════════════════════════════════════════════
    B6y, B6h = B5y + B5h + 2, 66
    section_hdr(B6y, "LECTURE DU PATTERN")

    P6y, P6h = B6y + 14, B6h - 14
    rfill(MX, P6y, CW, P6h, PANEL_DARK)
    rfill(MX, P6y, 3, P6h, CYAN)

    P6Cw = int(CW * 0.26)
    txt(MX + 10, P6y + 12, "PATTERN DETECTE", font=RF, size=6, color=GREY_LIGHT)
    txt(MX + 10, P6y + 28, pattern,            font=BF, size=14, color=CYAN)
    for bi in range(4):
        rfill(MX + 10 + bi * 9, P6y + 36, 6, 12, CYAN)

    vline(MX + P6Cw, P6y + 4, P6h - 8, BORDER_SUBTLE, 0.3)
    Ptx = MX + P6Cw + 10
    if pat_title:
        txt(Ptx, P6y + 16, pat_title.upper()[:50], font=BF, size=11, color=WHITE)
    if pat_text:
        for i, line in enumerate(_wrap_text(pat_text, 58)[:3]):
            txt(Ptx, P6y + 31 + i * 10, line, font=RF, size=7.5, color=WHITE)

    # ════════════════════════════════════════════════════════════════
    # [7] POUR LA SUITE  y≈492, h=112
    # ════════════════════════════════════════════════════════════════
    B7y, B7h = B6y + B6h + 2, 112
    section_hdr(B7y, "POUR LA SUITE")

    S7y, S7h = B7y + 14, B7h - 14
    HW = (CW - 3) / 2

    # Card gauche INSIGHT
    rfill(MX, S7y, HW, S7h, PANEL_DARK)
    rfill(MX, S7y, 3, S7h, CYAN)
    diamond(MX + 12, S7y + 10, 4, CYAN)
    txt(MX + 20, S7y + 12, "INSIGHT FORT", font=BF, size=6, color=CYAN)
    if ins_title:
        for i, il in enumerate(_wrap_text(ins_title.upper(), 32)[:2]):
            txt(MX + 8, S7y + 26 + i * 12, il, font=BF, size=9, color=WHITE)
    if ins_text:
        for i, line in enumerate(_wrap_text(ins_text, 36)[:5]):
            txt(MX + 8, S7y + 52 + i * 9, line, font=RF, size=7, color=WHITE)

    # Séparateur GOLD
    rfill(MX + HW, S7y, 3, S7h, GOLD)

    # Card droite PROGRESSION
    R7x = MX + HW + 3
    rfill(R7x, S7y, HW, S7h, PANEL_DARK)
    rfill(R7x, S7y, 3, S7h, GOLD)
    diamond(R7x + 12, S7y + 10, 4, GOLD)
    txt(R7x + 20, S7y + 12, "AXE DE PROGRESSION", font=BF, size=6, color=GOLD)
    if prg_title:
        for i, pl in enumerate(_wrap_text(prg_title.upper(), 32)[:2]):
            txt(R7x + 8, S7y + 26 + i * 12, pl, font=BF, size=9, color=WHITE)
    if prg_text:
        for i, line in enumerate(_wrap_text(prg_text, 36)[:5]):
            txt(R7x + 8, S7y + 52 + i * 9, line, font=RF, size=7, color=WHITE)

    # ════════════════════════════════════════════════════════════════
    # [8] PROFIL ELEVATION  y≈606, h=86
    # ════════════════════════════════════════════════════════════════
    B8y, B8h = B7y + B7h + 2, 86
    elev_lbl = f"// PROFIL ELEVATION  {dist_km:.1f} KM  {elev_gain} M D+  {prof_type} //"
    txt(PW / 2, B8y + 9, elev_lbl, font=RF, size=6, color=GREY_LIGHT, align='center')

    G8y, G8h = B8y + 12, B8h - 12
    rfill(MX, G8y, CW, G8h, DEEP_NAVY)

    if len(elev_prof) > 3:
        n = len(elev_prof)
        GH = G8h - 12
        # fill sous courbe
        c.saveState()
        fp = c.beginPath()
        fp.moveTo(MX, Y(G8y + 6 + GH))
        for i, v in enumerate(elev_prof):
            fp.lineTo(MX + CW * i / (n - 1), Y(G8y + 6 + GH * (1 - max(0, min(v, 1)))))
        fp.lineTo(MX + CW, Y(G8y + 6 + GH))
        fp.close()
        c.setFillColor(HexColor('#0D3A42'))
        c.drawPath(fp, fill=1, stroke=0)
        c.restoreState()
        # ligne courbe
        c.saveState()
        c.setStrokeColor(CYAN); c.setLineWidth(1.5)
        lp = c.beginPath()
        for i, v in enumerate(elev_prof):
            px = MX + CW * i / (n - 1)
            py = G8y + 6 + GH * (1 - max(0, min(v, 1)))
            if i == 0: lp.moveTo(px, Y(py))
            else:       lp.lineTo(px, Y(py))
        c.drawPath(lp, fill=0, stroke=1)
        c.restoreState()
        # marqueurs Q
        for qi in range(1, 4):
            qx = MX + CW * qi / 4
            c.saveState()
            c.setStrokeColor(GREY_LIGHT); c.setLineWidth(0.5); c.setDash([2, 3])
            c.line(qx, Y(G8y + 6), qx, Y(G8y + 6 + GH))
            c.restoreState()
            txt(qx, G8y + 9, f"Q{qi+1}", font=RF, size=6, color=GREY_LIGHT, align='center')
    else:
        txt(MX + CW / 2, G8y + G8h / 2 + 4, "Profil non disponible",
            font=RF, size=7.5, color=GREY_LIGHT, align='center')

    # ════════════════════════════════════════════════════════════════
    # [9] FOOTER
    # ════════════════════════════════════════════════════════════════
    F9y = PH - 20 - 42
    hline(F9y, 0, PW, BORDER_SUBTLE)
    txt(MX, F9y + 10, "// DISCLAIMER MEDICAL //", font=BF, size=6, color=GREY_LIGHT)
    disc = ("Score experimental - modele physiologique Minetti 2002 + decouplage cardiaque. "
            "Non valide cliniquement. Usage personnel et pedagogique uniquement. "
            "Ne remplace pas un suivi medical ou un test d'effort encadre.")
    for i, dl in enumerate(_wrap_text(disc, 80)[:3]):
        txt(MX, F9y + 19 + i * 7, dl, font=RF, size=5.5, color=GREY_LIGHT)
    v_logo(PW - MX - 56, F9y + 6, sc=1.1)
    txt(PW - MX - 44, F9y + 16, "VERTEX", font=BF, size=8, color=WHITE)
    txt(PW - MX - 52, F9y + 28, "(c) 2026  BSL 1.1", font=RF, size=5.5, color=GREY_LIGHT)

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
