"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — charts.py                                       ║
║         Plotly charts · PDF generator · v3.4                    ║
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
    fig.add_trace(go.Scatter(
        x=dist_km, y=df['elevation'],
        mode='lines', line=dict(color='#41C8E8', width=1.5),
        fill='tozeroy', fillcolor='rgba(65,200,232,0.06)',
    ))
    fig.update_layout(**_layout(height=180, yaxis_title="m", xaxis_title="km"))
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
    pdf.cell(0, 5, clean("PERFORMANCE INTELLIGENCE  |  RACE ANALYSIS v3.4"), ln=True, align="C")
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
        for z in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']:
            bpm = zones['bpm'].get(z, (0, 0))
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
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(42, 64, 80)
        pdf.cell(30, 5, clean(q + " :"), border=0)
        pdf.set_text_color(100, 130, 150)
        pdf.cell(0, 5, clean(val), ln=True)

    # Section découplage cardiaque — v3.4 : pattern-aware
    _pattern = drift.get('pattern')
    _insuf   = drift.get('insufficient_data', False)
    if not _insuf:
        section("DECOUPLAGE CARDIAQUE")
        if _pattern == 'COLLAPSE':
            fc_q1 = drift.get('fc_q1_mean') or 0
            fc_q4 = drift.get('fc_q4_mean') or 0
            cp    = drift.get('collapse_pct') or 0
            kpi("Pattern detecte :", "CARDIAC COLLAPSE", color=(200, 72, 80))
            kpi("FC Q1 moyen (plat) :", f"{fc_q1:.0f} bpm")
            kpi("FC Q4 moyen (plat) :", f"{fc_q4:.0f} bpm", color=(200, 72, 80))
            kpi("Effondrement FC :", f"{abs(cp):.1f}%", color=(200, 72, 80))
            kpi("EF :", "Non interpretable (collapse)", color=(100, 130, 150))
        else:
            if drift.get('drift_pct') is not None:
                kpi("EF 1ere moitie (plat) :", f"{drift['ef1']:.3f}" if drift['ef1'] else "N/A")
                kpi("EF 2eme moitie (plat) :", f"{drift['ef2']:.3f}" if drift['ef2'] else "N/A")
                drift_val = drift['drift_pct']
                color = (200, 72, 80) if drift_val < -5 else (200, 168, 75) if drift_val < -2 else (65, 200, 232)
                kpi("Derive EF :", f"{drift_val:.1f}%", color=color)
                kpi("Pattern :", _pattern or "N/A")

    section("PROFIL PENTE")
    for _, row in grade_df.iterrows():
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(42, 64, 80)
        pdf.cell(40, 5, clean(str(row['Tranche pente']) + " :"), border=0)
        pdf.set_text_color(100, 130, 150)
        pdf.cell(0, 5, clean(f"{row['Allure (min/km)']} /km"), ln=True)

    if cad_analysis.get('mean'):
        section("ANALYSE CADENCE")
        kpi("Cadence moyenne :", f"{cad_analysis['mean']:.0f} spm")
        kpi("Zone optimale (170-200spm) :", f"{cad_analysis['optimal_pct']:.0f}% du temps")
        for k, v in cad_analysis['dist'].items():
            if v > 1:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(42, 64, 80)
                pdf.cell(40, 5, clean(k + " spm :"), border=0)
                pdf.set_text_color(100, 130, 150)
                pdf.cell(0, 5, clean(f"{v:.0f}%"), ln=True)

    if splits:
        section("SPLITS PAR KM (resume)")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_text_color(42, 64, 80)
        cols   = ["Km", "Allure", "GAP", "D+", "FC", "Cad"]
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
    level_colors = {'info': (65, 200, 232), 'warn': (200, 168, 75), 'crit': (200, 72, 80)}
    for i, rec in enumerate(recs, 1):
        color = level_colors.get(rec['level'], (100, 130, 150))
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
        clean(f"VERTEX v3.4 — GAP Minetti (2002) — FCmax: {fcmax} bpm — {datetime.now().strftime('%d/%m/%Y')}"),
        ln=True, align="C")

    return bytes(pdf.output())
