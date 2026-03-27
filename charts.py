"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — charts.py                                       ║
║         Plotly charts · PDF generator                           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF

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


def chart_quartiles(quartiles: dict) -> go.Figure:
    labels = list(quartiles.keys())
    values = [round(v, 4) if not _isnan(v) else 0 for v in quartiles.values()]
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
# PDF GENERATOR — v3.4
# ══════════════════════════════════════════════════════════════════

def clean(text: str) -> str:
    # NFC préserve les caractères accentués composés (é, è, à...)
    # compatibles latin-1 — NFKD les décompose et crée des combining chars
    # qui font planter multi_cell sur certaines versions FPDF2
    text = unicodedata.normalize("NFC", str(text))
    text = text.replace("\u2014", "--").replace("\u2013", "-").replace("\u2019", "'")
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf(info, fi, flat_v, profile, grade_df,
                 zones, drift, cad_analysis, splits, recs,
                 fcmax, perf=None, verdict=None, email="") -> bytes:
    """
    PDF v3 — 2 pages structurées.
    Page 1 : hero zone (bande couleur + verdict + score + capsules) + splits complets
    Page 2 : graphes FPDF2 natifs (GAP quartiles / zones FC / dérive) + recos (2 max)
    Zéro dépendance externe — FPDF2 uniquement.
    """

    # ── Palette couleurs ────────────────────────────────────────────
    C_BG       = (8,  14,  20)
    C_BG2      = (13, 21,  32)
    C_SEP      = (21, 32,  48)
    C_DIM      = (42, 64,  80)
    C_MID      = (100, 130, 150)
    C_CYAN     = (65,  200, 232)
    C_AMBER    = (200, 168, 75)
    C_RED      = (200, 72,  80)
    C_WHITE    = (255, 255, 255)

    _HEX_RGB = {
        '#41C8E8': C_CYAN,
        '#C8A84B': C_AMBER,
        '#C84850': C_RED,
        '#2A4050': C_DIM,
    }

    # ── Résolution verdict/couleur ──────────────────────────────────
    if verdict:
        _vcolor = _HEX_RGB.get(verdict.get('color', ''), C_CYAN)
        _vlabel = verdict.get('label', 'ANALYSE INCOMPLETE')
        _vsub   = verdict.get('sub', '').replace('\u2014', '--').replace('\u2013', '-')
        _vcode  = verdict.get('code', '')
    else:
        _vcolor = C_DIM
        _vlabel = 'DONNEES INSUFFISANTES'
        _vsub   = ''
        _vcode  = '--'

    _score       = perf.get('score')       if perf else None
    _score_gap   = perf.get('score_gap')   if perf else None
    _score_ef    = perf.get('score_ef')    if perf else None
    _score_var   = perf.get('score_var')   if perf else None
    _partial     = perf.get('partial')     if perf else False
    _p_reason    = perf.get('partial_reason', '') if perf else ''

    _score_color = (
        C_CYAN  if (_score or 0) >= 80 else
        C_AMBER if (_score or 0) >= 60 else
        C_RED
    ) if _score is not None else C_DIM

    # ── Classe PDF ──────────────────────────────────────────────────
    class VertexPDF(FPDF):
        def header(self):
            # Fond sombre sur toutes les pages (y compris auto page break)
            self.set_fill_color(*C_BG)
            self.rect(0, 0, 210, 297, 'F')
            # Bande couleur verdict — persistante sur chaque page
            self.set_fill_color(*_vcolor)
            self.rect(0, 0, 4, 297, 'F')

    pdf = VertexPDF()
    pdf.set_compression(False)
    pdf.set_auto_page_break(auto=True, margin=12)

    # ── Helpers ─────────────────────────────────────────────────────
    def sep():
        pdf.set_draw_color(*C_SEP)
        pdf.set_line_width(0.3)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(3)

    def section(title):
        pdf.ln(2)
        sep()
        pdf.set_font("Courier", "", 6)
        pdf.set_text_color(*C_DIM)
        pdf.cell(0, 4, clean(f"-- {title} --"), ln=True)
        pdf.ln(2)

    def kpi(label, value, color=None):
        if color is None:
            color = C_CYAN
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_DIM)
        pdf.cell(70, 5, clean(label), border=0)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*color)
        pdf.cell(0, 5, clean(value), ln=True)

    def hbar(x, y, w_total, pct, color, height=4):
        """Barre horizontale proportionnelle — fond sombre + remplissage coloré."""
        pdf.set_fill_color(*C_SEP)
        pdf.rect(x, y, w_total, height, 'F')
        filled = max(1, int(w_total * min(pct, 100) / 100))
        pdf.set_fill_color(*color)
        pdf.rect(x, y, filled, height, 'F')

    def capsule(x, y, w, label, value, color):
        """Mini-capsule KPI : fond sombre, label micro, valeur colorée."""
        pdf.set_fill_color(*C_BG2)
        pdf.rect(x, y, w, 14, 'F')
        pdf.set_draw_color(*C_SEP)
        pdf.set_line_width(0.2)
        pdf.rect(x, y, w, 14, 'D')
        pdf.set_font("Courier", "", 5)
        pdf.set_text_color(*C_DIM)
        pdf.set_xy(x + 2, y + 1.5)
        pdf.cell(w - 4, 4, clean(label), border=0)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*color)
        pdf.set_xy(x + 2, y + 6)
        pdf.cell(w - 4, 6, clean(value), border=0)

    # ══════════════════════════════════════════════════════════════
    # PAGE 1 — HERO ZONE + SPLITS
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()

    # ── Header VERTEX ─────────────────────────────────────────────
    pdf.set_xy(12, 10)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*C_CYAN)
    pdf.cell(80, 10, clean("VERTEX"), border=0)
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(*C_DIM)
    pdf.set_xy(12, 20)
    pdf.cell(0, 4, clean("PERFORMANCE INTELLIGENCE  |  RACE ANALYSIS v3.5"), border=0, ln=True)

    # Date — coin droit
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(*C_DIM)
    pdf.set_xy(130, 10)
    pdf.cell(65, 4, clean(datetime.now().strftime('%d/%m/%Y')), border=0, align="R", ln=True)

    pdf.set_xy(15, 26)
    pdf.set_draw_color(*C_SEP)
    pdf.set_line_width(0.3)
    pdf.line(15, 26, 195, 26)

    # ── HERO ZONE — score + verdict + nom ─────────────────────────
    # Bloc gauche : SCORE grand + label VERDICT
    if _score is not None:
        pdf.set_xy(15, 29)
        pdf.set_font("Helvetica", "B", 42)
        pdf.set_text_color(*_score_color)
        pdf.cell(32, 16, clean(str(_score)), border=0, align="L")
        pdf.set_xy(15, 45)
        pdf.set_font("Courier", "", 5)
        pdf.set_text_color(*C_DIM)
        pdf.cell(32, 4, clean("/ 100"), border=0, align="L", ln=False)
        # Ligne contextuelle sous le score
        if _score >= 80:
            _ctx_label, _ctx_color = "course parfaitement maitrisee", C_CYAN
        elif _score >= 60:
            _ctx_label, _ctx_color = "marge de progression", C_AMBER
        else:
            _ctx_label, _ctx_color = "effort au-dessus du seuil", C_RED
        pdf.set_xy(15, 49)
        pdf.set_font("Courier", "", 5)
        pdf.set_text_color(*_ctx_color)
        pdf.cell(32, 4, clean(_ctx_label), border=0, align="L")

    # Bloc droite du score : verdict code + label + sub
    _hero_x = 50 if _score is not None else 15
    pdf.set_xy(_hero_x, 29)
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(*C_DIM)
    pdf.cell(145, 4, clean(f"VERDICT · {_vcode}"), border=0, ln=True)

    pdf.set_xy(_hero_x, 38)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(*_vcolor)
    pdf.cell(145, 7, clean(_vlabel), border=0, ln=True)

    if _vsub:
        pdf.set_xy(_hero_x, 45)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_MID)
        pdf.multi_cell(145 - (_hero_x - 15), 4, clean(_vsub))

    # Nom de la course — y garanti > 52 pour éviter collision avec _vsub long
    _name_y = max(pdf.get_y(), 52)
    pdf.set_xy(15, _name_y)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(*C_WHITE)
    pdf.cell(180, 6, clean(info.get('name', 'Course').upper()), border=0, ln=True)

    # ── 3 capsules métriques ──────────────────────────────────────
    dist_km = info.get('distance_km', 0)
    total_s = info.get('total_time_s', 0)
    h_t = int(total_s // 3600)
    m_t = int((total_s % 3600) // 60)
    d_plus = int(info.get('elevation_gain', 0))

    cap_y = int(pdf.get_y()) + 2  # suit le nom de course, jamais de collision
    if info.get('has_hr') and info.get('hr_mean') is not None:
        capsule(15,  cap_y, 52, "FC MOY",     f"{int(info['hr_mean'])} bpm",                            C_CYAN)
        capsule(70,  cap_y, 52, "ALLURE GAP", (v_to_pace(flat_v) if flat_v else '--:--') + "/km",       C_AMBER)
        capsule(125, cap_y, 52, "D+",          f"{d_plus} m",                                           C_MID)
    else:
        capsule(15,  cap_y, 52, "DISTANCE",   f"{dist_km:.1f} km",                                     C_CYAN)
        capsule(70,  cap_y, 52, "ALLURE MOY", v_to_pace(info.get('avg_velocity_ms', 0)) + "/km",       C_AMBER)
        capsule(125, cap_y, 52, "D+",          f"{d_plus} m",                                           C_MID)

    # Profil + allure plat
    pdf.set_xy(15, cap_y + 17)
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(*C_DIM)
    avg_pace = v_to_pace(info.get('avg_velocity_ms', 0))
    flat_pace = v_to_pace(flat_v) if flat_v else '--:--'
    pdf.cell(0, 4, clean(f"{profile}  |  Allure moy : {avg_pace}/km  |  Allure plat : {flat_pace}/km  |  Temps : {h_t}h{m_t:02d}"), ln=True)

    if info.get('hr_mean'):
        pdf.set_font("Courier", "", 6)
        pdf.set_text_color(*C_DIM)
        pdf.cell(0, 4, clean(
            f"FC moy : {int(info['hr_mean'])} bpm  |  FC max obs. : {int(info.get('hr_max', 0))} bpm"
            f"  |  FCmax saisie : {fcmax} bpm"
        ), ln=True)

    pdf.ln(2)
    _sep_y = pdf.get_y()
    pdf.set_draw_color(*C_SEP)
    pdf.set_line_width(0.3)
    pdf.line(15, _sep_y, 195, _sep_y)
    pdf.ln(3)

    # ── BLOC CE QUE TU DOIS RETENIR ──────────────────────────────
    _action_line = verdict.get('action_line', '') if verdict else ''
    _action_valid = bool(_action_line and len(_action_line) > 20)
    _rec0_valid   = len(recs) > 0

    if _action_valid or _rec0_valid:
        base_h = 8
        if _action_valid:
            base_h += 8
        if _rec0_valid:
            base_h += 14

        _block_y = pdf.get_y()
        pdf.set_fill_color(*C_BG2)
        pdf.rect(15, _block_y, 180, base_h, 'F')
        pdf.set_draw_color(*_vcolor)
        pdf.set_line_width(1.5)
        pdf.line(15, _block_y, 15, _block_y + base_h)

        pdf.set_xy(19, _block_y + 2)
        pdf.set_font("Courier", "", 5)
        pdf.set_text_color(*C_DIM)
        pdf.cell(0, 4, clean("CE QUE TU DOIS RETENIR"), ln=True)

        if _action_valid:
            pdf.set_x(19)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*_vcolor)
            pdf.cell(0, 4, clean(f"-> {_action_line}"), ln=True)

        if _rec0_valid:
            rec0 = recs[0]
            _badge_map = {'crit': ('*', C_RED), 'warn': ('!', C_AMBER), 'info': ('>', C_CYAN)}
            _badge, _badge_c = _badge_map.get(rec0.get('level', 'info'), ('>', C_CYAN))
            pdf.set_x(19)
            pdf.set_font("Courier", "", 5)
            pdf.set_text_color(*_badge_c)
            pdf.cell(6, 4, clean(_badge), border=0)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*C_WHITE)
            pdf.cell(0, 4, clean(rec0.get('title', '')), ln=True)
            pdf.set_x(19)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*C_MID)
            pdf.multi_cell(174, 4, clean(rec0.get('body', '')))

        pdf.ln(2)

    # ── SPLITS — tableau complet ──────────────────────────────────
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(*C_DIM)
    pdf.cell(0, 4, clean("-- SPLITS PAR KM --"), ln=True)
    pdf.ln(1)

    if splits:
        has_hr  = any(s.get('hr')      for s in splits)
        has_cad = any(s.get('cadence') for s in splits)

        # En-têtes
        # Largeur imprimable 180mm — colonnes adaptées selon présence FC/cadence
        # UX-4 : colonnes ajustées à la zone imprimable (180mm)
        if has_hr and has_cad:
            col_w = [8, 30, 30, 22, 22, 34, 34]   # total 180mm
        elif has_hr or has_cad:
            col_w = [8, 38, 38, 28, 28, 40]        # total 180mm
        else:
            col_w = [8, 52, 52, 34, 34]             # total 180mm
        headers = ['KM', 'ALLURE', 'GAP', 'D+', 'D-']
        if has_hr:  headers.append('FC')
        if has_cad: headers.append('CAD')

        pdf.set_font("Helvetica", "B", 6)
        pdf.set_text_color(*C_DIM)
        for h_txt, w in zip(headers, col_w):
            pdf.cell(w, 4, clean(h_txt), border=0)
        pdf.ln()

        # Détection zones chaudes
        valid_paces = [s['pace_s'] for s in splits if s.get('pace_s')]
        worst_km = best_km = fcmax_km = None
        if valid_paces:
            paced = [s for s in splits if s.get('pace_s')]
            worst_km = paced[valid_paces.index(max(valid_paces))]['km']
            best_km  = paced[valid_paces.index(min(valid_paces))]['km']
            if best_km == worst_km:
                best_km = None
        if has_hr:
            valid_hr = [(s['km'], s['hr']) for s in splits if s.get('hr')]
            if valid_hr:
                fcmax_km = max(valid_hr, key=lambda x: x[1])[0]

        med_pace = sum(valid_paces) / len(valid_paces) if valid_paces else None

        pdf.set_font("Helvetica", "", 6)
        for sp in splits:
            km     = sp['km']
            pace_s = sp.get('pace_s')

            # Couleur ligne
            if km == worst_km:
                row_c = C_RED
            elif km == fcmax_km:
                row_c = C_AMBER
            elif km == best_km:
                row_c = C_CYAN
            else:
                row_c = C_MID

            # Couleur allure relative
            if pace_s and med_pace:
                pace_c = C_CYAN if pace_s < med_pace * 0.92 else C_RED if pace_s > med_pace * 1.08 else row_c
            else:
                pace_c = row_c

            pdf.set_text_color(*row_c)
            pdf.cell(col_w[0], 3.5, clean(str(km)), border=0)
            pdf.set_text_color(*pace_c)
            pdf.cell(col_w[1], 3.5, clean(sp.get('pace', '--')), border=0)
            pdf.set_text_color(*row_c)
            pdf.cell(col_w[2], 3.5, clean(sp.get('gap',  '--')), border=0)
            pdf.set_text_color(*C_CYAN)
            pdf.cell(col_w[3], 3.5, clean(f"+{sp.get('d_pos', 0)}m"), border=0)
            pdf.set_text_color(*C_RED)
            pdf.cell(col_w[4], 3.5, clean(f"-{sp.get('d_neg', 0)}m"), border=0)
            if has_hr:
                hr_val = sp.get('hr')
                hr_c   = C_RED if hr_val and hr_val > fcmax * 0.92 else C_MID
                pdf.set_text_color(*hr_c)
                pdf.cell(col_w[5], 3.5, clean(str(hr_val) if hr_val else '--'), border=0)
            if has_cad:
                cad_val = sp.get('cadence')
                cad_c   = C_CYAN if cad_val and 170 <= cad_val <= 200 else C_MID
                pdf.set_text_color(*cad_c)
                pdf.cell(col_w[6] if has_hr else col_w[5], 3.5, clean(str(cad_val) if cad_val else '--'), border=0)
            pdf.ln()

        # Légende
        pdf.ln(2)
        pdf.set_font("Courier", "", 5)
        pdf.set_text_color(*C_RED)
        pdf.cell(30, 3, clean("v KM LENT"), border=0)
        pdf.set_text_color(*C_CYAN)
        pdf.cell(30, 3, clean("^ KM RAPIDE"), border=0)
        pdf.set_text_color(*C_AMBER)
        pdf.cell(30, 3, clean("o FC MAX"), border=0)
        pdf.ln()

    # ── Phrase de partage — S1 ────────────────────────────────────
    pdf.ln(4)
    sep()
    _share_score = _score
    _race_name   = info.get('name', 'Course')
    if _share_score is not None:
        _share_line = f"{_race_name}  --  {_vlabel}  --  VERTEX Score {_share_score}/100"
    else:
        _share_line = f"{_race_name}  --  {_vlabel}"
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(*C_CYAN)
    pdf.multi_cell(0, 5, clean(_share_line), align="C")
    if _vsub:
        pdf.set_font("Helvetica", "", 6)
        pdf.set_text_color(*C_MID)
        _vsub_clean = clean(_vsub)
        try:
            pdf.multi_cell(0, 4, _vsub_clean, align="C")
        except Exception:
            pdf.cell(0, 4, _vsub_clean[:80], ln=True, align="C")
    pdf.ln(2)


    # ── Footer page 1 ─────────────────────────────────────────────
    pdf.set_xy(15, 285)
    pdf.set_font("Courier", "", 5)
    pdf.set_text_color(*C_DIM)
    pdf.cell(85, 4, clean("Analyse algorithmique -- non validee cliniquement."), border=0)
    pdf.cell(0, 4, clean("1/2"), border=0, align="R", ln=True)

    # ══════════════════════════════════════════════════════════════
    # PAGE 2 — ANALYSE GRAPHIQUE + RECOS
    # ══════════════════════════════════════════════════════════════
    pdf.add_page()

    # ── Rappel en-tête page 2 ─────────────────────────────────────
    pdf.set_xy(15, 15)
    pdf.set_font("Courier", "", 6)
    pdf.set_text_color(*C_DIM)
    _score_display = str(_score) if _score is not None else "--"
    _recall = f"VERTEX  ·  {info.get('name','').upper()}  ·  Score {_score_display}/100  ·  {_vlabel}"
    pdf.cell(0, 4, clean(_recall), ln=True)
    pdf.ln(2)

    # ── GAP QUARTILES — barres horizontales ───────────────────────
    section("PROFIL FATIGUE GAP -- QUARTILES")

    quartiles = fi.get('quartiles', {})
    dr = fi.get('decay_ratio', float('nan'))
    dp = fi.get('decay_pct',   float('nan'))

    q_vals = {k: v for k, v in quartiles.items() if not _isnan(v)}
    if q_vals:
        max_v = max(q_vals.values())
        bar_x = 55
        bar_w = 120
        q_colors = [C_CYAN, C_CYAN, C_AMBER, C_RED]

        for i, (q_key, q_val) in enumerate(q_vals.items()):
            y_row = pdf.get_y()
            pct   = (q_val / max_v * 100) if max_v > 0 else 0
            col   = q_colors[min(i, 3)]

            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*C_DIM)
            pdf.set_xy(15, y_row)
            pdf.cell(38, 5, clean(q_key), border=0)

            hbar(bar_x, y_row + 0.5, bar_w, pct, col, height=4)

            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*col)
            pdf.set_xy(bar_x + bar_w + 3, y_row)
            pdf.cell(25, 5, clean(f"{v_to_pace(q_val)}/km"), border=0)
            pdf.ln(6)

        pdf.ln(1)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_DIM)
        dr_str = f"{dr:.3f}" if not _isnan(dr) else "N/A"
        dp_str = f"{dp:.1f}%" if not _isnan(dp) else "N/A"
        pdf.cell(0, 4, clean(f"Ratio Q4/Q1 : {dr_str}   |   Ecart GAP : {dp_str}"), ln=True)
    else:
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_DIM)
        pdf.cell(0, 5, clean("Donnees GAP insuffisantes"), ln=True)

    # ── ZONES FC — barres horizontales ───────────────────────────
    if zones:
        section("ZONES DE FREQUENCE CARDIAQUE")

        z_colors = {
            'Z1': C_CYAN,
            'Z2': (65, 180, 130),
            'Z3': C_AMBER,
            'Z4': (200, 120, 60),
            'Z5': C_RED,
        }
        bar_x = 55
        bar_w = 100

        for z in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']:
            bpm_range = zones['bpm'].get(z, (0, 0))
            pct       = zones['pct'].get(z, 0)
            t_s       = zones['time'].get(z, 0)
            mm        = int(t_s // 60)
            z_col     = z_colors.get(z, C_MID)
            y_row     = pdf.get_y()

            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*z_col)
            pdf.set_xy(15, y_row)
            pdf.cell(10, 5, clean(z), border=0)

            pdf.set_font("Helvetica", "", 6)
            pdf.set_text_color(*C_DIM)
            pdf.set_xy(26, y_row)
            pdf.cell(28, 5, clean(f"{bpm_range[0]}-{bpm_range[1]} bpm"), border=0)

            hbar(bar_x, y_row + 0.5, bar_w, pct, z_col, height=4)

            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*z_col)
            pdf.set_xy(bar_x + bar_w + 3, y_row)
            pdf.cell(20, 5, clean(f"{pct:.0f}%"), border=0)

            pdf.set_font("Helvetica", "", 6)
            pdf.set_text_color(*C_DIM)
            pdf.set_xy(bar_x + bar_w + 20, y_row)
            pdf.cell(20, 5, clean(f"{mm} min"), border=0)
            pdf.ln(6)

    # ── DÉRIVE CARDIAQUE ─────────────────────────────────────────
    _pattern = drift.get('pattern')
    _insuf   = drift.get('insufficient_data', False)

    if not _insuf and _pattern:
        section("DECOUPLAGE CARDIAQUE")

        if _pattern in ('COLLAPSE A', 'COLLAPSE B', 'COLLAPSE'):
            fc_q1 = drift.get('fc_q1_mean') or 0
            fc_q4 = drift.get('fc_q4_mean') or 0
            cp    = abs(drift.get('collapse_pct') or 0)

            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*C_RED)
            pdf.cell(0, 6, clean("SIGNAL CARDIAQUE ANORMAL"), ln=True)

            # Barre visuelle FC Q1 vs Q4
            y_bar = pdf.get_y()
            max_fc = max(fc_q1, fc_q4, 1)
            hbar(15,  y_bar, 80, fc_q1 / max_fc * 100, C_MID,  height=5)
            hbar(100, y_bar, 80, fc_q4 / max_fc * 100, C_RED,  height=5)
            pdf.ln(7)

            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*C_DIM)
            pdf.cell(85, 4, clean(f"Q1 (plat) : {fc_q1:.0f} bpm"), border=0)
            pdf.set_text_color(*C_RED)
            pdf.cell(0, 4, clean(f"Q4 (plat) : {fc_q4:.0f} bpm  |  Chute : {cp:.1f}%"), ln=True)
        else:
            ef1   = drift.get('ef1')
            ef2   = drift.get('ef2')
            d_pct = drift.get('drift_pct')

            if d_pct is not None:
                d_col = C_RED if d_pct < -5 else C_AMBER if d_pct < -2 else C_CYAN
                _pattern_labels = {
                    'DRIFT':  'DÉRIVE CARDIAQUE',
                    'STABLE': 'CARDIAQUE STABLE',
                }
                _plabel = _pattern_labels.get(_pattern, _pattern)
                pdf.set_font("Helvetica", "B", 8)
                pdf.set_text_color(*d_col)
                pdf.cell(0, 5, clean(_plabel), ln=True)

                # Barre EF1 vs EF2
                # KNOWN LIMITATION : ef1/ef2 calculés sur segments plats uniquement.
                # Sur terrain quasi-plat, la dérive peut diverger de l'EF globale UI.
                if ef1 and ef2:
                    y_bar = pdf.get_y()
                    max_ef = max(ef1, ef2, 0.001)
                    hbar(15,  y_bar, 80, ef1 / max_ef * 100, C_CYAN,  height=5)
                    hbar(100, y_bar, 80, ef2 / max_ef * 100, d_col,   height=5)
                    pdf.ln(7)
                    pdf.set_font("Helvetica", "", 7)
                    pdf.set_text_color(*C_DIM)
                    pdf.cell(85, 4, clean(f"Efficacite cardiaque 1ere moitie : {ef1:.3f}"), border=0)
                    pdf.set_text_color(*d_col)
                    if not _isnan(d_pct):
                        _qualif = "(normal)" if d_pct > -4 else "(fatigue moderee)" if d_pct > -8 else "(signal fort)"
                    else:
                        _qualif = ""
                    _qualif_str = f"  {_qualif}" if _qualif else ""
                    pdf.cell(0, 4, clean(f"Efficacite cardiaque 2eme moitie : {ef2:.3f}  |  Baisse : {d_pct:.1f}%{_qualif_str}"), ln=True)

    elif _insuf:
        section("DECOUPLAGE CARDIAQUE")
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_DIM)
        pdf.cell(0, 5, clean("Non calculable -- moins de 10 min de terrain plat sur ce parcours."), ln=True)

    # ── CADENCE (compact) ─────────────────────────────────────────
    if cad_analysis.get('mean'):
        section("CADENCE")
        cad_mean    = cad_analysis['mean']
        cad_opt_pct = cad_analysis.get('optimal_pct', 0)
        cad_col     = C_CYAN if 170 <= cad_mean <= 200 else C_AMBER

        y_bar = pdf.get_y()
        hbar(55, y_bar, 120, cad_opt_pct, cad_col, height=5)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(*cad_col)
        pdf.set_xy(55 + 120 + 3, y_bar)
        pdf.cell(0, 5, clean(f"{cad_opt_pct:.0f}% en zone"), ln=True)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(*C_DIM)
        pdf.cell(0, 4, clean(f"Cadence moy : {cad_mean:.0f} spm  |  Zone optimale : 170-200 spm"), ln=True)

    # ── SCORE DÉTAIL ──────────────────────────────────────────────
    if perf and _score is not None:
        section("SCORE VERTEX -- DETAIL")

        weights = perf.get('weights', {})
        sub_data = [
            ("Regularite d'allure",   _score_gap, int(weights.get('gap', 0) * 100)),
            ("Efficacite cardiaque", _score_ef,  int(weights.get('ef',  0) * 100)),
            ("Regularite",           _score_var, int(weights.get('var', 0) * 100)),
        ]
        bar_x = 70
        bar_w = 90

        for s_label, s_val, s_w in sub_data:
            if s_val is None:
                continue
            s_col = C_CYAN if s_val >= 80 else C_AMBER if s_val >= 60 else C_RED
            y_row = pdf.get_y()

            pdf.set_font("Helvetica", "", 6)
            pdf.set_text_color(*C_DIM)
            pdf.set_xy(15, y_row)
            pdf.cell(53, 5, clean(f"{s_label} ({s_w}%)"), border=0)

            hbar(bar_x, y_row + 0.5, bar_w, s_val, s_col, height=4)

            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*s_col)
            pdf.set_xy(bar_x + bar_w + 3, y_row)
            pdf.cell(15, 5, clean(str(s_val)), border=0)
            pdf.ln(6)

        if _partial and _p_reason:
            pdf.set_font("Courier", "", 6)
            pdf.set_text_color(*C_AMBER)
            pdf.cell(0, 4, clean(f"Score partiel : {_p_reason}"), ln=True)

        # Disclaimer Minetti
        pdf.set_font("Courier", "", 5)
        pdf.set_text_color(*C_DIM)
        pdf.cell(0, 4,
            clean("Score experimental -- modele GAP Minetti (2002) + decouplage cardiaque. Non valide cliniquement."),
            ln=True)

    # ── RECOMMANDATIONS COACH — recs[1:3] uniquement (rec[0] déjà en p1) ──
    recs_p2 = recs[1:3]
    if recs_p2:
        section("RECOMMANDATIONS COACH")

        level_colors = {'crit': C_RED, 'warn': C_AMBER, 'info': C_CYAN}
        level_labels = {'crit': 'PRIORITAIRE', 'warn': 'ATTENTION', 'info': 'INFO'}

        for rec in recs_p2:
            col   = level_colors.get(rec.get('level', 'info'), C_CYAN)
            label = level_labels.get(rec.get('level', 'info'), 'INFO')

            y_start = pdf.get_y()

            pdf.set_font("Courier", "", 7)
            pdf.set_text_color(*col)
            pdf.set_xy(19, y_start + 1)
            pdf.cell(0, 4, clean(label), ln=True)

            pdf.set_font("Helvetica", "B", 7)
            pdf.set_text_color(*C_WHITE)
            pdf.set_x(19)
            pdf.cell(0, 5, clean(rec.get('title', '')), ln=True)

            pdf.set_font("Helvetica", "", 6)
            pdf.set_text_color(*C_MID)
            pdf.set_x(19)
            pdf.multi_cell(174, 4, clean(rec.get('body', '')))
            pdf.ln(2)

            # Barre latérale dynamique — hauteur calculée après le contenu
            y_end = pdf.get_y()
            bar_h = max(y_end - y_start, 4)
            pdf.set_fill_color(*col)
            pdf.rect(15, y_start, 2, bar_h, 'F')
            pdf.ln(2)

    # ── Footer ────────────────────────────────────────────────────
    pdf.ln(2)
    sep()
    pdf.set_font("Courier", "", 5)
    pdf.set_text_color(*C_SEP)
    _footer = f"VERTEX v3.5  |  GAP : Minetti et al. (2002)  |  FCmax : {fcmax} bpm  |  {datetime.now().strftime('%d/%m/%Y')}"
    if email:
        _footer += f"  |  {email}"
    pdf.cell(0, 3, clean(_footer), ln=True, align="C")

    # ── Footer page 2 ─────────────────────────────────────────────
    pdf.set_xy(15, 285)
    pdf.set_font("Courier", "", 5)
    pdf.set_text_color(*C_DIM)
    pdf.cell(0, 4, clean("2/2"), border=0, align="R", ln=True)

    return bytes(pdf.output())
