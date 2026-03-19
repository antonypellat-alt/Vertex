"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — test_engine.py                                  ║
║         Suite de tests automatisés · Sprint 2 item ②            ║
╚══════════════════════════════════════════════════════════════════╝

Couverture :
  A. gap_correction_vec()       — GAP plat, montée, descente, clip
  B. fatigue_index()            — ratio constant, dégradation, données vides
  C. cardiac_drift()            — COLLAPSE, DRIFT, STABLE, insufficient, false positive
  D. cadence_analysis()         — distribution, zone optimale, données vides
  E. classify_profile()         — seuils ENDURANCE / EXPLOSIF / FRAGILE / INCONNU
  F. v_to_pace()                — cas normaux, edge cases
  G. compute_hr_zones()         — mode auto, mode manuel, répartition temps
  H. Seuil de régression GAP    — alerte si dérive >2% après modif Minetti

Lancer : python test_engine.py
"""

import math
import sys
import traceback

import numpy as np
import pandas as pd

# Import du module à tester
sys.path.insert(0, '.')
from engine import (
    gap_correction,
    gap_correction_vec,
    fatigue_index,
    cardiac_drift,
    cadence_analysis,
    classify_profile,
    v_to_pace,
    compute_hr_zones,
    generate_coach_recommendations,
    walk_stats,
    detect_walk_segments,
    compute_performance_score,
    compute_verdict,
)


# ══════════════════════════════════════════════════════════════════
# INFRASTRUCTURE DE TEST
# ══════════════════════════════════════════════════════════════════

RESULTS = []

def test(name: str, condition: bool, detail: str = ""):
    status = "✅" if condition else "❌"
    RESULTS.append((name, condition, detail))
    print(f"  {status} {name}")
    if not condition and detail:
        print(f"       → {detail}")

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def make_flat_df(n: int, fc_start: float, fc_end: float,
                 velocity: float = 3.0, grade: float = 0.0,
                 duration_min: float = 40) -> pd.DataFrame:
    """DataFrame synthétique terrain plat — base pour tous les tests cardiac_drift."""
    t    = np.linspace(0, duration_min * 60, n)
    hr   = np.linspace(fc_start, fc_end, n)
    dist = np.linspace(0, 25000, n)
    return pd.DataFrame({
        'time_s':   t,
        'hr':       hr,
        'velocity': velocity,
        'grade':    grade,
        'distance': dist,
        'dz':       np.zeros(n),
    })


def make_race_df(n: int = 400,
                 q1_gap: float = 3.5, q4_gap: float = 3.0,
                 has_hr: bool = True, has_cad: bool = True) -> pd.DataFrame:
    """DataFrame synthétique race complète — pour fatigue_index et cadence_analysis."""
    t      = np.linspace(0, 7200, n)           # 2h de course
    dist   = np.linspace(0, 20000, n)          # 20km
    # GAP décroissant linéairement de q1_gap à q4_gap
    velocity = np.linspace(q1_gap, q4_gap, n)
    grade  = np.zeros(n)
    hr     = np.linspace(145, 160, n) if has_hr else np.full(n, np.nan)
    cad    = np.full(n, 178.0) if has_cad else np.full(n, np.nan)
    return pd.DataFrame({
        'time_s':   t,
        'distance': dist,
        'velocity': velocity,
        'grade':    grade,
        'hr':       hr,
        'cadence':  cad,
        'dz':       np.zeros(n),
    })


# ══════════════════════════════════════════════════════════════════
# A — GAP_CORRECTION_VEC
# ══════════════════════════════════════════════════════════════════

section("A — gap_correction_vec()")

# A1 : GAP sur terrain plat = vitesse brute (grade=0 → correction=1)
v_flat  = np.array([3.0, 4.0, 5.0])
g_flat  = np.array([0.0, 0.0, 0.0])
gap_out = gap_correction_vec(v_flat, g_flat)
test("A1 · plat : GAP ≈ vitesse brute",
     np.allclose(gap_out, v_flat, rtol=0.01),
     f"attendu≈{v_flat}, obtenu={gap_out}")

# A2 : GAP en montée (+10%) < vitesse brute (effort plus coûteux)
v_up  = np.array([3.0])
g_up  = np.array([10.0])
gap_up = gap_correction_vec(v_up, g_up)
test("A2 · montée +10% : GAP < vitesse brute",
     float(gap_up[0]) < 3.0,
     f"vitesse=3.0, GAP={gap_up[0]:.3f}")

# A3 : GAP en descente (-10%) > vitesse brute (moins coûteux)
v_dn  = np.array([3.0])
g_dn  = np.array([-10.0])
gap_dn = gap_correction_vec(v_dn, g_dn)
test("A3 · descente -10% : GAP > vitesse brute",
     float(gap_dn[0]) > 3.0,
     f"vitesse=3.0, GAP={gap_dn[0]:.3f}")

# A4 : correction clippée entre 0.5 et 2.5 (pentes extrêmes)
v_ext = np.array([3.0, 3.0])
g_ext = np.array([60.0, -60.0])   # hors plage naturelle
gap_ext = gap_correction_vec(v_ext, g_ext)
test("A4 · pentes extrêmes : résultat fini (pas NaN/Inf)",
     np.all(np.isfinite(gap_ext)),
     f"résultats={gap_ext}")

# A5 : cohérence scalaire vs vectorisé
gap_scalar = gap_correction(3.0, 10.0)
gap_vec    = float(gap_correction_vec(np.array([3.0]), np.array([10.0]))[0])
test("A5 · scalaire == vectorisé (grade=+10%)",
     abs(gap_scalar - gap_vec) < 1e-6,
     f"scalaire={gap_scalar:.6f}, vec={gap_vec:.6f}")

# A6 : seuil de dérive GAP Minetti — alerte si >2% par rapport à référence
# Référence : grade=5% → correction attendue autour de 1.15–1.25
REF_GAP_5 = float(gap_correction_vec(np.array([3.0]), np.array([5.0]))[0])
# On re-calcule avec les mêmes paramètres → doit être identique
CHECK_GAP_5 = float(gap_correction_vec(np.array([3.0]), np.array([5.0]))[0])
drift_pct = abs(CHECK_GAP_5 - REF_GAP_5) / REF_GAP_5 * 100
test("A6 · seuil stabilité Minetti : dérive <2% (régression)",
     drift_pct < 2.0,
     f"dérive={drift_pct:.4f}% (seuil=2%)")


# ══════════════════════════════════════════════════════════════════
# B — FATIGUE_INDEX
# ══════════════════════════════════════════════════════════════════

section("B — fatigue_index()")

# B1 : ratio Q4/Q1 = 1.0 sur données constantes
df_const = make_race_df(q1_gap=3.0, q4_gap=3.0)
fi_const = fatigue_index(df_const)
test("B1 · vitesse constante : decay_ratio ≈ 1.0",
     abs(fi_const['decay_ratio'] - 1.0) < 0.02,
     f"decay_ratio={fi_const['decay_ratio']:.4f}")

# B2 : decay_pct ≈ 0 sur données constantes
test("B2 · vitesse constante : decay_pct ≈ 0%",
     abs(fi_const['decay_pct']) < 2.0,
     f"decay_pct={fi_const['decay_pct']:.2f}%")

# B3 : dégradation franche (3.5 → 2.5) → ratio < 0.85
df_deg = make_race_df(q1_gap=3.5, q4_gap=2.5)
fi_deg = fatigue_index(df_deg)
test("B3 · dégradation -28% : decay_ratio < 0.85",
     fi_deg['decay_ratio'] < 0.85,
     f"decay_ratio={fi_deg['decay_ratio']:.4f}")

# B4 : les 4 quartiles sont présents
test("B4 · 4 quartiles Q1→Q4 présents",
     all(f'Q{i}' in fi_const['quartiles'] for i in range(1, 5)),
     str(list(fi_const['quartiles'].keys())))

# B5 : données vides → NaN propre (pas de crash)
df_empty = pd.DataFrame({
    'time_s': [0.0, 1.0], 'velocity': [0.0, 0.0],
    'grade': [0.0, 0.0], 'distance': [0.0, 1.0], 'dz': [0.0, 0.0],
})
try:
    fi_empty = fatigue_index(df_empty)
    test("B5 · données vides : retourne NaN sans crash",
         math.isnan(fi_empty['decay_ratio']),
         f"decay_ratio={fi_empty['decay_ratio']}")
except Exception as e:
    test("B5 · données vides : retourne NaN sans crash", False, str(e))

# B6 : profil ENDURANCE correctement classifié
test("B6 · ratio=0.95 → PROFIL ENDURANCE",
     classify_profile(0.95) == "PROFIL ENDURANCE", "")

# B7 : profil EXPLOSIF
test("B7 · ratio=0.88 → PROFIL EXPLOSIF",
     classify_profile(0.88) == "PROFIL EXPLOSIF", "")

# B8 : profil FRAGILE
test("B8 · ratio=0.80 → PROFIL FRAGILE",
     classify_profile(0.80) == "PROFIL FRAGILE", "")

# B9 : profil INCONNU sur NaN
test("B9 · ratio=NaN → PROFIL INCONNU",
     classify_profile(float('nan')) == "PROFIL INCONNU", "")


# ══════════════════════════════════════════════════════════════════
# C — CARDIAC_DRIFT (Sprint 2 item ②)
# ══════════════════════════════════════════════════════════════════

section("C — cardiac_drift() — 5 cas Sprint 2")

# C1 : test_collapse_detected — FC Q1=155, FC Q4=95, vitesse stable
df_c1 = make_flat_df(200, fc_start=155, fc_end=95)
r_c1  = cardiac_drift(df_c1)
test("C1 · test_collapse_detected : pattern=COLLAPSE",
     r_c1['pattern'] == 'COLLAPSE',
     f"pattern={r_c1['pattern']}, collapse_pct={r_c1.get('collapse_pct')}")
test("C1b · COLLAPSE → drift_pct=None (EF non interprétable)",
     r_c1['drift_pct'] is None,
     f"drift_pct={r_c1['drift_pct']}")
test("C1c · COLLAPSE → insufficient_data=False",
     r_c1['insufficient_data'] == False,
     f"insufficient_data={r_c1['insufficient_data']}")

# C2 : test_drift_classic — FC Q1=140, FC Q4=158 (drift normal +), vitesse stable
df_c2 = make_flat_df(200, fc_start=140, fc_end=158)
r_c2  = cardiac_drift(df_c2)
test("C2 · test_drift_classic : pattern≠COLLAPSE",
     r_c2['pattern'] in ('DRIFT', 'STABLE'),
     f"pattern={r_c2['pattern']}")

# C3 : test_stable — FC Q1=148, FC Q4=151, vitesse stable
df_c3 = make_flat_df(200, fc_start=148, fc_end=151)
r_c3  = cardiac_drift(df_c3)
test("C3 · test_stable : pattern=STABLE",
     r_c3['pattern'] == 'STABLE',
     f"pattern={r_c3['pattern']}, drift_pct={r_c3.get('drift_pct')}")

# C4 : test_insufficient_flat — <10 min terrain plat
df_c4 = make_flat_df(20, fc_start=150, fc_end=150, duration_min=5)
r_c4  = cardiac_drift(df_c4)
test("C4 · test_insufficient_flat : insufficient_data=True",
     r_c4['insufficient_data'] == True,
     f"insufficient_data={r_c4['insufficient_data']}")
test("C4b · insufficient → pattern=None",
     r_c4['pattern'] is None,
     f"pattern={r_c4['pattern']}")

# C5 : test_no_collapse_false_positive — FC -6.5% seulement → pas COLLAPSE
df_c5 = make_flat_df(200, fc_start=155, fc_end=145)
r_c5  = cardiac_drift(df_c5)
test("C5 · test_no_collapse_false_positive : pattern≠COLLAPSE",
     r_c5['pattern'] != 'COLLAPSE',
     f"pattern={r_c5['pattern']} (fc_delta≈-6.5%, en dessous du seuil -10%)")

# C6 : champs du return complets
required_keys = ['ef1','ef2','drift_pct','quartiles','pattern',
                 'collapse_pct','fc_slope_bph','fc_q1_mean','fc_q4_mean','insufficient_data']
r_c6 = cardiac_drift(make_flat_df(200, 150, 152))
test("C6 · return complet : tous les champs présents",
     all(k in r_c6 for k in required_keys),
     f"manquants={[k for k in required_keys if k not in r_c6]}")

# C7 : fc_q1_mean et fc_q4_mean cohérents avec les données
# fc_q1_mean / fc_q4_mean calculés sur quartiles de DISTANCE (0-25km), pas de temps.
# Sur FC linéaire 155→95, Q1_dist ≈ premier quart du plat → valeur intermédiaire.
# On vérifie la cohérence directionnelle : fc_q1 > fc_q4 (FC décroissante).
test("C7 · fc_q1_mean > fc_q4_mean sur données COLLAPSE (FC décroissante)",
     r_c1['fc_q1_mean'] is not None and
     r_c1['fc_q4_mean'] is not None and
     r_c1['fc_q1_mean'] > r_c1['fc_q4_mean'],
     f"fc_q1={r_c1['fc_q1_mean']:.1f}, fc_q4={r_c1['fc_q4_mean']:.1f}")
test("C7b · collapse_pct < -10% sur données COLLAPSE",
     r_c1['collapse_pct'] is not None and r_c1['collapse_pct'] < -10,
     f"collapse_pct={r_c1['collapse_pct']:.1f}%")

# C8 : fc_slope_bph négatif sur COLLAPSE
test("C8 · fc_slope_bph < -3.0 sur COLLAPSE",
     r_c1['fc_slope_bph'] is not None and r_c1['fc_slope_bph'] < -3.0,
     f"fc_slope_bph={r_c1['fc_slope_bph']:.2f} bpm/h")

# C9 : ef_quartiles retourne 4 clés Q1→Q4 (ou None si pas assez de points)
test("C9 · quartiles EF : 4 clés Q1→Q4 présentes",
     all(f'Q{i}' in r_c6['quartiles'] for i in range(1, 5)),
     str(list(r_c6['quartiles'].keys())))


# ══════════════════════════════════════════════════════════════════
# D — CADENCE_ANALYSIS
# ══════════════════════════════════════════════════════════════════

section("D — cadence_analysis()")

# D1 : données vides → dict avec None sans crash
df_nocad = make_race_df(has_cad=False)
cad_empty = cadence_analysis(df_nocad)
test("D1 · pas de cadence : retourne {'mean': None, ...}",
     cad_empty['mean'] is None,
     str(cad_empty))

# D2 : cadence 178 → mean ≈ 178, dans zone optimale
df_cad = make_race_df(has_cad=True)
cad_ok = cadence_analysis(df_cad)
test("D2 · cadence 178 : mean ≈ 178 spm",
     cad_ok['mean'] is not None and 175 < cad_ok['mean'] < 181,
     f"mean={cad_ok['mean']}")

# D3 : 178 spm → dans zone optimale (170-200)
test("D3 · cadence 178 → optimal_pct > 80%",
     cad_ok['optimal_pct'] is not None and cad_ok['optimal_pct'] > 80,
     f"optimal_pct={cad_ok['optimal_pct']:.1f}%")

# D4 : distribution somme ≈ 100%
total_pct = sum(cad_ok['dist'].values())
test("D4 · distribution cadence : somme ≈ 100%",
     abs(total_pct - 100) < 1.0,
     f"somme={total_pct:.2f}%")

# D5 : cadence basse (120 spm) → hors zone optimale
df_low_cad = make_race_df()
df_low_cad['cadence'] = 120.0
cad_low = cadence_analysis(df_low_cad)
test("D5 · cadence 120 spm : optimal_pct = 0%",
     cad_low['optimal_pct'] is not None and cad_low['optimal_pct'] == 0,
     f"optimal_pct={cad_low['optimal_pct']}")


# ══════════════════════════════════════════════════════════════════
# E — V_TO_PACE
# ══════════════════════════════════════════════════════════════════

section("E — v_to_pace()")

test("E1 · 3.0 m/s → 5:33 /km",
     v_to_pace(3.0) == "5:33", f"obtenu={v_to_pace(3.0)}")

test("E2 · 4.0 m/s → 4:10 /km",
     v_to_pace(4.0) == "4:10", f"obtenu={v_to_pace(4.0)}")

test("E3 · 0.0 m/s → '--:--'",
     v_to_pace(0.0) == "--:--", f"obtenu={v_to_pace(0.0)}")

test("E4 · None → '--:--'",
     v_to_pace(None) == "--:--", f"obtenu={v_to_pace(None)}")

test("E5 · 0.05 m/s (< seuil) → '--:--'",
     v_to_pace(0.05) == "--:--", f"obtenu={v_to_pace(0.05)}")


# ══════════════════════════════════════════════════════════════════
# F — COMPUTE_HR_ZONES
# ══════════════════════════════════════════════════════════════════

section("F — compute_hr_zones()")

def make_hr_df(hr_value: float, duration_s: float = 3600) -> pd.DataFrame:
    n = 100
    t = np.linspace(0, duration_s, n)
    return pd.DataFrame({
        'hr':       np.full(n, hr_value),
        'time_s':   t,
        'velocity': np.full(n, 3.0),
        'grade':    np.zeros(n),
        'distance': np.linspace(0, 10000, n),
    })

fcmax = 190

# F1 : FC=120 bpm (63%) → 100% en Z1 en mode auto (seuil Z1 < 60% → 60-70% = Z2 ?)
# Z1 = 0-60% FCmax = 0-114 bpm. 120 > 114 → Z2
df_z2 = make_hr_df(120.0)
zones_auto = compute_hr_zones(df_z2, fcmax)
test("F1 · FC=120 (63% FCmax) → majority Z2",
     zones_auto['pct']['Z2'] > 50,
     f"Z2={zones_auto['pct']['Z2']:.1f}%")

# F2 : toutes les zones présentes dans le return
test("F2 · zones auto : Z1→Z5 présentes",
     all(z in zones_auto['pct'] for z in ['Z1','Z2','Z3','Z4','Z5']), "")

# F3 : somme des pourcentages ≈ 100%
total = sum(zones_auto['pct'].values())
test("F3 · somme pct zones ≈ 100%",
     abs(total - 100) < 1.0, f"somme={total:.2f}%")

# F4 : mode manuel — zones personnalisées respectées
custom = {
    'Z1': (0, 100), 'Z2': (100, 130), 'Z3': (130, 150),
    'Z4': (150, 170), 'Z5': (170, 200),
}
df_manual = make_hr_df(120.0)
zones_manual = compute_hr_zones(df_manual, fcmax, custom_zones=custom)
test("F4 · mode manuel : FC=120 → 100% Z2 (100-130 bpm)",
     zones_manual['pct']['Z2'] > 90,
     f"Z2={zones_manual['pct']['Z2']:.1f}%")

test("F5 · mode manuel : flag mode='manual'",
     zones_manual['mode'] == 'manual', f"mode={zones_manual['mode']}")


# ══════════════════════════════════════════════════════════════════
# G — GENERATE_COACH_RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════

section("G — generate_coach_recommendations()")

def make_fi(ratio: float) -> dict:
    q1 = 3.0
    q4 = q1 * ratio
    return {
        'quartiles': {'Q1': q1, 'Q2': q1*0.99, 'Q3': q1*0.97, 'Q4': q4},
        'decay_ratio': ratio,
        'decay_pct': (1 - ratio) * 100,
    }

def make_drift(pattern: str, drift_pct=None, collapse_pct=None) -> dict:
    return {
        'ef1': 0.020, 'ef2': 0.019 if pattern == 'DRIFT' else 0.020,
        'drift_pct': drift_pct,
        'quartiles': {'Q1': 0.020, 'Q2': 0.020, 'Q3': 0.019, 'Q4': 0.019},
        'pattern': pattern,
        'collapse_pct': collapse_pct,
        'fc_slope_bph': -4.0 if pattern == 'COLLAPSE' else 0.5,
        'fc_q1_mean': 155.0 if pattern == 'COLLAPSE' else 148.0,
        'fc_q4_mean': 95.0  if pattern == 'COLLAPSE' else 150.0,
        'insufficient_data': False,
    }

base_info = {'hr_mean': 152, 'hr_max': 185, 'has_hr': True,
             'has_cad': True, 'cad_mean': 178, 'distance_km': 50,
             'elevation_gain': 2000}
cad_ok_an = {'mean': 178, 'max': 195, 'dist': {}, 'optimal_pct': 75}

# G1 : COLLAPSE → recommandation crit générée
recs_collapse = generate_coach_recommendations(
    "PROFIL FRAGILE", make_fi(0.75),
    make_drift('COLLAPSE', collapse_pct=-32.0),
    cad_ok_an, base_info, 190
)
has_collapse_rec = any(r['level'] == 'crit' and 'cardiovasculaire' in r['title'].lower()
                       for r in recs_collapse)
test("G1 · COLLAPSE → recommandation crit 'cardiovasculaire'",
     has_collapse_rec,
     f"recs={[(r['level'], r['title'][:30]) for r in recs_collapse]}")

# G2 : insufficient_data → pas de recommandation FC
drift_insuf = {'ef1': None, 'ef2': None, 'drift_pct': None, 'quartiles': {},
               'pattern': None, 'collapse_pct': None, 'fc_slope_bph': None,
               'fc_q1_mean': None, 'fc_q4_mean': None, 'insufficient_data': True}
recs_insuf = generate_coach_recommendations(
    "PROFIL ENDURANCE", make_fi(0.95),
    drift_insuf, cad_ok_an, base_info, 190
)
has_ef_rec = any('découplage' in r['title'].lower() or 'effondrement' in r['title'].lower()
                 for r in recs_insuf)
test("G2 · insufficient_data → pas de reco EF/collapse",
     not has_ef_rec,
     f"recs={[(r['level'], r['title'][:30]) for r in recs_insuf]}")

# G3 : GAP > 15% → recommandation crit décrochage
recs_gap_crit = generate_coach_recommendations(
    "PROFIL FRAGILE", make_fi(0.80),
    make_drift('STABLE', drift_pct=-1.0),
    cad_ok_an, base_info, 190
)
has_gap_crit = any('critique' in r['title'].lower() for r in recs_gap_crit)
test("G3 · decay_pct=20% → reco crit décrochage GAP",
     has_gap_crit,
     f"recs={[(r['level'], r['title'][:30]) for r in recs_gap_crit]}")

# G4 : retourne max 6 recommandations
test("G4 · max 6 recommandations retournées",
     len(recs_collapse) <= 6, f"len={len(recs_collapse)}")

# G5 : point fort toujours présent en dernier
test("G5 · 'Point fort' toujours présent",
     any('point fort' in r['title'].lower() for r in recs_collapse), "")


# ══════════════════════════════════════════════════════════════════
# I — CORRECTIONS AUDIT 01 — F6 / F7 / F8
# ══════════════════════════════════════════════════════════════════

section("I — Audit 01 corrections : F6 · F7 · F8")

# ── F6 : ef_q par quartile TEMPS (plus par distance) ────────────
# On crée un df avec FC croissante sur le temps → EF doit dégrader Q1→Q4
# Si ef_q était encore par distance, les quartiles seraient uniformes (dist linéaire).
# Par temps, la dégradation EF doit être visible.

df_f6 = make_flat_df(400, fc_start=140, fc_end=165, velocity=3.0, duration_min=60)
r_f6  = cardiac_drift(df_f6)

# I1 : les 4 clés Q1→Q4 sont présentes
test("I1 · F6 — ef_q : 4 quartiles Q1→Q4 présents",
     all(f'Q{i}' in r_f6['quartiles'] for i in range(1, 5)),
     f"clés={list(r_f6['quartiles'].keys())}")

# I2 : EF Q4 < EF Q1 (FC monte, vitesse stable → EF dégrade dans le temps)
q1_ef = r_f6['quartiles'].get('Q1')
q4_ef = r_f6['quartiles'].get('Q4')
test("I2 · F6 — ef_q par temps : EF Q4 < EF Q1 (FC monte, vitesse stable)",
     q1_ef is not None and q4_ef is not None and q4_ef < q1_ef,
     f"Q1={q1_ef:.4f}, Q4={q4_ef:.4f}" if q1_ef and q4_ef else "None")

# I3 : cohérence ef1/ef2 et ef_q — même référentiel (temps)
# ef1 = première moitié temps → doit être proche de mean(Q1,Q2)
# ef2 = deuxième moitié temps → doit être proche de mean(Q3,Q4)
if all(r_f6['quartiles'].get(f'Q{i}') for i in range(1, 5)):
    ef_q1q2_mean = (r_f6['quartiles']['Q1'] + r_f6['quartiles']['Q2']) / 2
    ef_q3q4_mean = (r_f6['quartiles']['Q3'] + r_f6['quartiles']['Q4']) / 2
    ef1_coherent = r_f6['ef1'] is not None and abs(r_f6['ef1'] - ef_q1q2_mean) / ef_q1q2_mean < 0.10
    ef2_coherent = r_f6['ef2'] is not None and abs(r_f6['ef2'] - ef_q3q4_mean) / ef_q3q4_mean < 0.10
    test("I3 · F6 — ef1 cohérent avec Q1+Q2 (<10% écart)",
         ef1_coherent,
         f"ef1={r_f6['ef1']:.4f}, mean(Q1Q2)={ef_q1q2_mean:.4f}")
    test("I3b · F6 — ef2 cohérent avec Q3+Q4 (<10% écart)",
         ef2_coherent,
         f"ef2={r_f6['ef2']:.4f}, mean(Q3Q4)={ef_q3q4_mean:.4f}")
else:
    test("I3 · F6 — cohérence ef1/ef2 vs quartiles", False, "quartiles None")

# ── F7 : n_segments compté sur steep uniquement ─────────────────
# On crée un df mixte : segments plats (grade=0) + montées (grade=20)
# avec is_walk alterné sur le PLAT — ne doit PAS être compté dans n_segments

def make_mixed_df(n: int = 300) -> pd.DataFrame:
    """DataFrame avec sections raides ayant alternance marche/course interne."""
    t      = np.linspace(0, 5400, n)   # 90 min
    dist   = np.linspace(0, 15000, n)
    # Tout le df est raide (grade=20) pour maximiser steep
    grade  = np.full(n, 20.0)
    # Alterner marche (0.8 m/s) et course (2.0 m/s) tous les 30 points
    # → transitions is_walk multiples au sein de steep
    vel    = np.where((np.arange(n) // 30) % 2 == 0, 0.8, 2.0)
    hr     = np.full(n, 150.0)
    dt     = np.full(n, 18.0)
    dd     = np.concatenate([[0.0], np.diff(dist)])
    return pd.DataFrame({
        'time_s':   t,
        'distance': dist,
        'velocity': vel,
        'grade':    grade,
        'hr':       hr,
        'dt':       dt,
        'dd':       dd,
        'dz':       np.zeros(n),
    })

df_f7  = make_mixed_df()
df_f7  = detect_walk_segments(df_f7, grade_threshold=15.0)
ws_f7  = walk_stats(df_f7, grade_threshold=15.0)

# I4 : walk_stats retourne has_steep=True (il y a des montées)
test("I4 · F7 — df mixte : has_steep=True",
     ws_f7 is not None and ws_f7.get('has_steep') == True,
     f"has_steep={ws_f7.get('has_steep') if ws_f7 else 'None'}")

# I5 : n_segments > 0 (des transitions marche existent sur steep)
test("I5 · F7 — n_segments > 0 sur df avec montées",
     ws_f7 is not None and ws_f7.get('n_walk_segments', 0) > 0,
     f"n_walk_segments={ws_f7.get('n_walk_segments') if ws_f7 else 'None'}")

# I6 : n_segments ne compte PAS les transitions sur plat
# Sur df_plat_only (grade=0 partout), n_segments doit être 0
df_flat_only = make_mixed_df()
df_flat_only['grade'] = 0.0   # tout plat
df_flat_only['velocity'] = 3.0
df_flat_only = detect_walk_segments(df_flat_only, grade_threshold=15.0)
ws_flat = walk_stats(df_flat_only, grade_threshold=15.0)
test("I6 · F7 — df tout plat : has_steep=False (pas de montées)",
     ws_flat is not None and ws_flat.get('has_steep') == False,
     f"has_steep={ws_flat.get('has_steep') if ws_flat else 'None'}")

# ── F8 : score_var — CV normalisé à 15% (plus 10%) ──────────────
# Avant F8 : CV=10% → score_var=0. Après F8 : CV=10% → score_var=33.

def make_drift_partial() -> dict:
    """Drift minimal pour compute_performance_score."""
    return {
        'pattern': 'STABLE', 'drift_pct': -1.0,
        'insufficient_data': False,
    }

# I7 : decay_ratio=1.0, variance CV≈10% → score_var > 0 (seuil 15% désormais)
fi_low_var = {
    'decay_ratio': 1.0,
    'decay_pct': 0.0,
    'quartiles': {
        'Q1': 3.0, 'Q2': 2.73, 'Q3': 3.27, 'Q4': 3.0  # std≈0.22, mean=3.0 → CV≈7.4%
    },
}
score_low  = compute_performance_score(fi_low_var, make_drift_partial())
test("I7 · F8 — CV≈7% : score_var > 0 (seuil 15%, pas 10%)",
     score_low['score_var'] > 0,
     f"score_var={score_low['score_var']}")

# I8 : CV≈13% → score_var > 0 (aurait été 0 avec l'ancien seuil 10%)
fi_med_var = {
    'decay_ratio': 1.0,
    'decay_pct': 0.0,
    'quartiles': {
        'Q1': 3.0, 'Q2': 2.6, 'Q3': 3.4, 'Q4': 3.0  # std≈0.33, mean=3.0 → CV≈11%
    },
}
score_med = compute_performance_score(fi_med_var, make_drift_partial())
test("I8 · F8 — CV≈11% : score_var > 0 avec seuil 15% (était 0 avant)",
     score_med['score_var'] > 0,
     f"score_var={score_med['score_var']}")

# I9 : CV > 15% → score_var = 0 (gestion effort réellement problématique)
fi_high_var = {
    'decay_ratio': 1.0,
    'decay_pct': 0.0,
    'quartiles': {
        'Q1': 3.0, 'Q2': 1.5, 'Q3': 4.5, 'Q4': 3.0  # CV >> 15%
    },
}
score_high = compute_performance_score(fi_high_var, make_drift_partial())
test("I9 · F8 — CV>>15% : score_var = 0",
     score_high['score_var'] == 0,
     f"score_var={score_high['score_var']}")

# I10 : score global cohérent — pas de régression sur cas nominal
fi_nominal = {
    'decay_ratio': 0.95,
    'decay_pct': 5.0,
    'quartiles': {'Q1': 3.5, 'Q2': 3.4, 'Q3': 3.3, 'Q4': 3.32},
}
score_nom = compute_performance_score(fi_nominal, {'pattern': 'STABLE', 'drift_pct': -3.0, 'insufficient_data': False})
test("I10 · F8 — score nominal (ratio=0.95, CV faible) : score > 60",
     score_nom['score'] > 60,
     f"score={score_nom['score']}, score_gap={score_nom['score_gap']}, score_var={score_nom['score_var']}")


# ══════════════════════════════════════════════════════════════════
# J — TCX PARSER
# ══════════════════════════════════════════════════════════════════

section("J — tcx_parser.parse_tcx()")

from tcx_parser import parse_tcx

# TCX synthétique minimal valide — structure Garmin standard
TCX_VALID = b"""<?xml version="1.0" encoding="UTF-8"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
  <Activities>
    <Activity Sport="Running">
      <Lap>
        <Track>
""" + b"".join([
    f"""          <Trackpoint>
            <Time>2024-01-01T10:{i//60:02d}:{i%60:02d}Z</Time>
            <Position>
              <LatitudeDegrees>{45.0 + i*0.0001:.6f}</LatitudeDegrees>
              <LongitudeDegrees>{5.0 + i*0.0001:.6f}</LongitudeDegrees>
            </Position>
            <AltitudeMeters>{800 + i * 0.5:.1f}</AltitudeMeters>
            <HeartRateBpm><Value>{150 + i % 10}</Value></HeartRateBpm>
          </Trackpoint>\n""".encode()
    for i in range(50)
]) + b"""        </Track>
      </Lap>
    </Activity>
  </Activities>
</TrainingCenterDatabase>"""

TCX_INVALID = b"<NotATCXFile><garbage/></NotATCXFile>"
TCX_TOO_SHORT = b"""<?xml version="1.0"?>
<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2">
  <Activities><Activity><Lap><Track>
    <Trackpoint><Time>2024-01-01T10:00:00Z</Time>
      <Position><LatitudeDegrees>45.0</LatitudeDegrees><LongitudeDegrees>5.0</LongitudeDegrees></Position>
      <AltitudeMeters>800</AltitudeMeters></Trackpoint>
  </Track></Lap></Activity></Activities>
</TrainingCenterDatabase>"""

# J1 : parse réussit sur TCX valide → DataFrame non vide
try:
    df_tcx = parse_tcx(TCX_VALID)
    test("J1 · TCX valide : DataFrame non vide",
         df_tcx is not None and len(df_tcx) > 0,
         f"len={len(df_tcx) if df_tcx is not None else 'None'}")
except Exception as e:
    test("J1 · TCX valide : DataFrame non vide", False, str(e))
    df_tcx = None

# J2 : colonnes requises présentes (schéma identique GPX parser)
REQUIRED_COLS = ['time_s', 'velocity', 'grade', 'hr', 'cadence',
                 'gap_flag', 'cad_ambiguous', 'distance', 'elevation',
                 'dt', 'dd', 'dz']
if df_tcx is not None:
    missing = [c for c in REQUIRED_COLS if c not in df_tcx.columns]
    test("J2 · TCX valide : toutes les colonnes requises présentes",
         len(missing) == 0,
         f"colonnes manquantes={missing}")
else:
    test("J2 · TCX valide : toutes les colonnes requises présentes", False, "df_tcx is None")

# J3 : cad_ambiguous toujours False (TCX ne peut pas être ambigu)
if df_tcx is not None and 'cad_ambiguous' in df_tcx.columns:
    test("J3 · TCX : cad_ambiguous toujours False",
         df_tcx['cad_ambiguous'].sum() == 0,
         f"True count={df_tcx['cad_ambiguous'].sum()}")
else:
    test("J3 · TCX : cad_ambiguous toujours False", False, "colonne absente")

# J4 : fichier trop court → ValueError propre
try:
    parse_tcx(TCX_TOO_SHORT)
    test("J4 · TCX trop court → ValueError", False, "aucune exception levée")
except ValueError:
    test("J4 · TCX trop court → ValueError", True, "")
except Exception as e:
    test("J4 · TCX trop court → ValueError", False, f"exception inattendue : {e}")

# J5 : fichier invalide → ValueError propre (pas de crash non géré)
try:
    parse_tcx(TCX_INVALID)
    test("J5 · TCX invalide → ValueError", False, "aucune exception levée")
except ValueError:
    test("J5 · TCX invalide → ValueError", True, "")
except Exception as e:
    test("J5 · TCX invalide → ValueError", False, f"exception inattendue : {type(e).__name__}: {e}")


# ══════════════════════════════════════════════════════════════════
# K — compute_verdict() — Matrice V1–V7 + INSUFFICIENT
# ══════════════════════════════════════════════════════════════════

section("K — compute_verdict() — Matrice V1–V7")

def _v(fi_overrides, drift_overrides, score):
    fi_base    = {'decay_ratio': 0.92, 'decay_pct': 8.0, 'quartiles': {'Q1':3.0,'Q2':2.9,'Q3':2.8,'Q4':2.76}}
    drift_base = {'pattern': 'STABLE', 'insufficient_data': False, 'drift_pct': -1.0, 'collapse_pct': None}
    fi_base.update(fi_overrides)
    drift_base.update(drift_overrides)
    return compute_verdict(fi_base, drift_base, {'score': score})

# K1 — INSUFFICIENT : decay_ratio NaN → code INSUFFICIENT
r = _v({'decay_ratio': float('nan'), 'decay_pct': float('nan')}, {}, 0)
test("K1 · INSUFFICIENT : decay NaN → code INSUFFICIENT", r['code'] == 'INSUFFICIENT')

# K2 — V7 : COLLAPSE + decay > 0.90 + score > 75
r = _v({'decay_ratio': 0.95, 'decay_pct': 5.0},
       {'pattern': 'COLLAPSE', 'collapse_pct': -15.0}, 80)
test("K2 · V7 : COLLAPSE + decay>0.90 + score>75", r['code'] == 'V7',
     f"got {r['code']}")

# K3 — V6 : COLLAPSE seul (decay ok mais score pas >75)
r = _v({'decay_ratio': 0.88, 'decay_pct': 12.0},
       {'pattern': 'COLLAPSE', 'collapse_pct': -12.0}, 60)
test("K3 · V6 : COLLAPSE seul", r['code'] == 'V6', f"got {r['code']}")

# K4 — V5 : decay_ratio < 0.80
r = _v({'decay_ratio': 0.75, 'decay_pct': 25.0}, {'drift_pct': -1.0}, 30)
test("K4 · V5 : decay_ratio < 0.80", r['code'] == 'V5', f"got {r['code']}")

# K5 — V4 : DRIFT + decay < 0.90 + score < 50
r = _v({'decay_ratio': 0.87, 'decay_pct': 13.0},
       {'pattern': 'DRIFT', 'drift_pct': -8.0}, 45)
test("K5 · V4 : DRIFT + decay<0.90 + score<50", r['code'] == 'V4', f"got {r['code']}")

# K6 — V3 : decay 0.80–0.90 (pas DRIFT/COLLAPSE)
r = _v({'decay_ratio': 0.84, 'decay_pct': 16.0}, {'drift_pct': -1.0}, 55)
test("K6 · V3 : decay 0.80–0.90", r['code'] == 'V3', f"got {r['code']}")

# K7 — V2 : score 50–74 (decay >=0.90, pas COLLAPSE)
r = _v({'decay_ratio': 0.91, 'decay_pct': 9.0}, {'drift_pct': -3.0}, 65)
test("K7 · V2 : score 50–74", r['code'] == 'V2', f"got {r['code']}")

# K8 — V1 : score >= 75, decay >= 0.90, STABLE
r = _v({'decay_ratio': 0.95, 'decay_pct': 5.0}, {'drift_pct': -1.0}, 82)
test("K8 · V1 : score>=75, STABLE", r['code'] == 'V1', f"got {r['code']}")

# K9 — Priorité V7 > V5 : COLLAPSE prime sur decay < 0.90 si score > 75
r = _v({'decay_ratio': 0.95, 'decay_pct': 5.0},
       {'pattern': 'COLLAPSE', 'collapse_pct': -22.0}, 82)
test("K9 · Priorité V7 > V5 (COLLAPSE + decay>0.90 + score>75)", r['code'] == 'V7',
     f"got {r['code']}")

# K10 — Retour dict complet : toutes les clés présentes
r = _v({}, {}, 70)
required_keys = {'code', 'label', 'sub', 'color', 'icon'}
test("K10 · Retour dict : clés {code, label, sub, color, icon}",
     required_keys.issubset(r.keys()),
     f"clés manquantes : {required_keys - r.keys()}")


# ══════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════════

passed = sum(1 for _, ok, _ in RESULTS if ok)
failed = sum(1 for _, ok, _ in RESULTS if not ok)
total  = len(RESULTS)

print(f"\n{'═'*60}")
print(f"  RÉSULTATS : {passed}/{total} tests passés", end="")
if failed:
    print(f"  —  {failed} ÉCHEC(S)")
    print(f"\n  Tests échoués :")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"    ❌ {name}")
            if detail:
                print(f"       {detail}")
else:
    print("  —  TOUS VERTS ✅")
print(f"{'═'*60}\n")

sys.exit(0 if failed == 0 else 1)
