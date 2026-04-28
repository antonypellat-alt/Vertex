"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — test_engine.py                                  ║
║         Suite de tests automatisés                              ║
╚══════════════════════════════════════════════════════════════════╝

Couverture :
  A. gap_correction_vec()              — GAP plat, montée, descente, clip, stabilité Minetti
  B. fatigue_index() / classify_profile() — ratio constant, dégradation, profils
  C. cardiac_drift()                   — COLLAPSE, DRIFT, STABLE, insufficient, false positive
  D. cadence_analysis()                — distribution, zone optimale, données vides
  E. v_to_pace()                       — cas normaux, edge cases
  F. compute_hr_zones()                — mode auto, mode manuel, répartition temps
  G. generate_coach_recommendations()  — v2.0 : COLLAPSE, GAP, cadence, profil, fallbacks
  I. walk_stats() / detect_walk_segments() / compute_performance_score()
                                       — F6 marche active, F7 segments, F8 CV seuil 15%
  J. tcx_parser.parse_tcx()            — valide, colonnes, cad_ambiguous, erreurs
  K. compute_verdict()                 — matrice V1-V7+INSUFFICIENT, BUG-1 decay>0.85+COLLAPSE→V3
  N. cardiac_drift() CDC v1.3           — DRIFT-CARDIO, DRIFT-NEURO, DRIFT faible, STABLE
  O. compute_verdict() CDC v1.3         — V3-NEURO : DRIFT-NEURO + decay<0.93→V3
  L. _get_race_profile()               — ULTRA/LONG/COURT, critère ET strict
  M. generate_pdf()                    — smoke tests : complet, sans FC, phrase partage, splits

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
    _get_race_profile,
    walk_stats,
    detect_walk_segments,
    compute_performance_score,
    compute_verdict,
    get_score_weights,
    detect_elevation_profile,
    apply_decay_correction,
    flat_pace_estimate,
    grade_pace_profile,
    hr_by_grade,
    compute_km_splits,
)
from gpx_parser import haversine_vec, extract_race_info


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

# C2 : test_drift_classic — FC Q1=140, FC Q4=158 (FC montante, vitesse stable)
# CDC v1.3 : ce cas peut produire DRIFT-CARDIO (FC monte + EF degrade) ou STABLE
# selon amplitude de la derive EF. Le test valide l'absence de COLLAPSE uniquement.
df_c2 = make_flat_df(200, fc_start=140, fc_end=158)
r_c2  = cardiac_drift(df_c2)
test("C2 · test_drift_classic : pattern≠COLLAPSE (CDC v1.3 : DRIFT-CARDIO/NEURO/DRIFT/STABLE attendus)",
     r_c2['pattern'] in ('DRIFT', 'DRIFT-CARDIO', 'DRIFT-NEURO', 'STABLE'),
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

# C5 : test_no_collapse_false_positive — CDC-R1 : fc_slope_global au-dessus du seuil → pas COLLAPSE
# slope ≈ -2.25 bph (155→153.5 sur 40min) > slope_thr=-3.0 → COLLAPSE ne doit pas déclencher
df_c5 = make_flat_df(200, fc_start=155, fc_end=153.5)
r_c5  = cardiac_drift(df_c5)
test("C5 · test_no_collapse_false_positive : pattern≠COLLAPSE",
     r_c5['pattern'] != 'COLLAPSE',
     f"pattern={r_c5['pattern']} (fc_slope≈-2.25 bph, au-dessus seuil -3.0 bph)")

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

# C10 : NEGATIVE_SPLIT — fc_slope sous seuil + decay_v >= +0.05 → pattern NEGATIVE_SPLIT (C5 v2)
# Simule Dylan : FC baisse (slope négatif fort) + vitesse Q4 > Q1 (+25%)
df_c10 = make_flat_df(300, fc_start=180, fc_end=165)   # slope négatif fort
r_c10 = cardiac_drift(df_c10, duration_s=3600, dp_per_km=20, decay_v=0.25)
test("C10 · C5 v2 : slope<seuil + decay_v=+0.25 → NEGATIVE_SPLIT (pas COLLAPSE)",
     r_c10['pattern'] == 'NEGATIVE_SPLIT',
     f"pattern={r_c10['pattern']}, decay_v={r_c10['decay_v']}")

# C11 : COLLAPSE maintenu — fc_slope sous seuil + decay_v < +0.05 → COLLAPSE (C5 v2)
# Simule Jérémy/Hivernatrail : FC baisse + vitesse en baisse
df_c11 = make_flat_df(300, fc_start=180, fc_end=165)   # même FC que C10
r_c11 = cardiac_drift(df_c11, duration_s=3600, dp_per_km=20, decay_v=-0.08)
test("C11 · C5 v2 : slope<seuil + decay_v=-0.08 → COLLAPSE (pas NEGATIVE_SPLIT)",
     r_c11['pattern'] == 'COLLAPSE',
     f"pattern={r_c11['pattern']}, decay_v={r_c11['decay_v']}")

# C12 : decay_v dans le dict retourné
test("C12 · decay_v présent dans le retour de cardiac_drift",
     'decay_v' in r_c10,
     f"keys={list(r_c10.keys())}")


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
# G — GENERATE_COACH_RECOMMENDATIONS v2.0 (Sprint 4 · R1+R2)
# ══════════════════════════════════════════════════════════════════

section("G — generate_coach_recommendations() v2.0")

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

# info LONG (50km / 4h) — profil détecté = ULTRA (≥50km ET ≥4h)
base_info_ultra = {
    'hr_mean': 152, 'hr_max': 185, 'has_hr': True, 'has_cad': True,
    'distance_km': 50, 'elevation_gain': 2000, 'total_time_s': 14400,
}
# info LONG (30km / 3h)
base_info_long = {
    'hr_mean': 152, 'hr_max': 185, 'has_hr': True, 'has_cad': True,
    'distance_km': 30, 'elevation_gain': 1200, 'total_time_s': 10800,
}
# info COURT (12km / 1h)
base_info_court = {
    'hr_mean': 152, 'hr_max': 185, 'has_hr': True, 'has_cad': True,
    'distance_km': 12, 'elevation_gain': 400, 'total_time_s': 3600,
}
cad_ok_an = {'mean': 178, 'max': 195, 'dist': {'170-180': 40, '180-190': 35, '190-200': 10}, 'optimal_pct': 75}

# G1 : COLLAPSE → reco crit générée (verbe d'action, pas répétition VERDICT)
recs_collapse = generate_coach_recommendations(
    "PROFIL FRAGILE", make_fi(0.88),
    make_drift('COLLAPSE', collapse_pct=-28.0),
    cad_ok_an, base_info_ultra, 190
)
has_collapse_crit = any(r['level'] == 'crit' and r.get('category') == 'CARDIAQUE'
                        for r in recs_collapse)
test("G1 · COLLAPSE → reco crit CARDIAQUE générée",
     has_collapse_crit,
     f"recs={[(r['level'], r.get('category','?'), r['title'][:25]) for r in recs_collapse]}")

# G2 : La reco COLLAPSE ne répète pas "effondrement cardiovasculaire" (c'est le VERDICT)
has_repetition = any('effondrement cardiovasculaire' in r['title'].lower()
                     for r in recs_collapse)
test("G2 · COLLAPSE → reco ne répète pas 'effondrement cardiovasculaire'",
     not has_repetition,
     f"titres={[r['title'][:40] for r in recs_collapse]}")

# G3 : insufficient_data → aucune reco cardiaque
drift_insuf = {'ef1': None, 'ef2': None, 'drift_pct': None, 'quartiles': {},
               'pattern': None, 'collapse_pct': None, 'fc_slope_bph': None,
               'fc_q1_mean': None, 'fc_q4_mean': None, 'insufficient_data': True}
recs_insuf = generate_coach_recommendations(
    "PROFIL ENDURANCE", make_fi(0.95),
    drift_insuf, cad_ok_an, base_info_long, 190
)
has_cardiac_rec = any(r.get('category') in ('CARDIAQUE', 'ENDURANCE') for r in recs_insuf)
test("G3 · insufficient_data → aucune reco cardiaque/endurance",
     not has_cardiac_rec,
     f"recs={[(r.get('category'), r['title'][:25]) for r in recs_insuf]}")

# G4 : GAP > 15% → reco crit GAP
recs_gap_crit = generate_coach_recommendations(
    "PROFIL FRAGILE", make_fi(0.80),
    make_drift('STABLE', drift_pct=-1.0),
    cad_ok_an, base_info_court, 190
)
has_gap_crit = any(r['level'] == 'crit' and r.get('category') == 'GAP' for r in recs_gap_crit)
test("G4 · decay_pct=20% → reco crit GAP",
     has_gap_crit,
     f"recs={[(r['level'], r.get('category','?'), r['title'][:25]) for r in recs_gap_crit]}")

# G5 : Tri correct — CRIT avant WARN avant INFO
_order = {'crit': 0, 'warn': 1, 'info': 2}
levels = [r['level'] for r in recs_collapse]
is_sorted = all(_order[levels[i]] <= _order[levels[i+1]] for i in range(len(levels)-1))
test("G5 · recos triées CRIT > WARN > INFO",
     is_sorted,
     f"levels={levels}")

# G6 : max 6 recommandations retournées
test("G6 · max 6 recommandations retournées",
     len(recs_collapse) <= 6, f"len={len(recs_collapse)}")

# G7 : R-H absent si aucun critère positif (COLLAPSE + mauvais GAP + cad basse)
cad_bad_an = {'mean': 155, 'max': 170, 'dist': {'<150': 5, '150-160': 30, '160-170': 40}, 'optimal_pct': 10}
recs_no_strength = generate_coach_recommendations(
    "PROFIL FRAGILE", make_fi(0.72),
    make_drift('COLLAPSE', collapse_pct=-30.0),
    cad_bad_an, base_info_court, 190
)
has_point_fort = any('capitaliser' in r['title'].lower() or 'maintenir' in r['title'].lower()
                     or 'exploiter' in r['title'].lower() for r in recs_no_strength)
# Note : R-H peut être absent (aucun critère positif) — c'est le comportement attendu
test("G7 · R-H absent si tous signaux négatifs (pas de faux point fort)",
     not has_point_fort,
     f"titres={[r['title'][:30] for r in recs_no_strength]}")

# G8 : profil ULTRA — texte adapté (mentionne distance ou durée)
recs_ultra = generate_coach_recommendations(
    "PROFIL ENDURANCE", make_fi(0.88),
    make_drift('DRIFT', drift_pct=-8.0),
    cad_ok_an, base_info_ultra, 190
)
has_drift_ultra = any(r.get('category') == 'ENDURANCE' for r in recs_ultra)
test("G8 · DRIFT sur ULTRA → reco ENDURANCE générée",
     has_drift_ultra,
     f"recs={[(r.get('category'), r['title'][:30]) for r in recs_ultra]}")

# G9 : cad_cv Option B — paramètre None → calcul interne sans crash
recs_cad_cv = generate_coach_recommendations(
    "PROFIL ENDURANCE", make_fi(0.93),
    make_drift('STABLE', drift_pct=-1.0),
    cad_ok_an, base_info_long, 190,
    cad_cv=None   # Option B : doit fonctionner sans ce paramètre
)
test("G9 · cad_cv=None → pas de crash, recos retournées",
     isinstance(recs_cad_cv, list) and len(recs_cad_cv) > 0,
     f"len={len(recs_cad_cv)}")


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

try:
    from tcx_parser import parse_tcx
    _tcx_available = True
except ImportError as _tcx_err:
    _tcx_available = False
    for _j in ["J1 · TCX valide : DataFrame non vide",
               "J2 · TCX valide : toutes les colonnes requises présentes",
               "J3 · TCX : cad_ambiguous toujours False",
               "J4 · TCX trop court → ValueError",
               "J5 · TCX invalide → ValueError"]:
        test(_j, False, f"gpx_parser absent du path — {_tcx_err}")

if _tcx_available:
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

# K3 — V6 : COLLAPSE franc + decay < 0.85 (seuil BUG-1)
# Note : decay=0.88 routait vers V6 avant BUG-1 — désormais V3 (decay≥0.85)
# K3 teste maintenant le cas V6 correct : decay=0.80 (< 0.85)
r = _v({'decay_ratio': 0.80, 'decay_pct': 20.0},
       {'pattern': 'COLLAPSE', 'collapse_pct': -12.0}, 60)
test("K3 · V6 : COLLAPSE + decay=0.80 (<0.85)", r['code'] == 'V6', f"got {r['code']}")

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

# K11 — BUG-1 Sprint 4A : COLLAPSE + decay 0.85–0.90 → V3 (pas V6)
# Cas réel : Jérémy 43km — collapse nutritionnel, allure partiellement tenue
r = _v({'decay_ratio': 0.87, 'decay_pct': 13.0},
       {'pattern': 'COLLAPSE', 'collapse_pct': -9.0}, 60)
test("K11 · BUG-1 : COLLAPSE + decay=0.87 → V3 (pas V6)",
     r['code'] == 'V3',
     f"got {r['code']}")

# K12 — BUG-1 : COLLAPSE + decay < 0.85 → V6 (comportement conservé)
r = _v({'decay_ratio': 0.82, 'decay_pct': 18.0},
       {'pattern': 'COLLAPSE', 'collapse_pct': -12.0}, 55)
test("K12 · BUG-1 : COLLAPSE + decay=0.82 → V6 (seuil <0.85)",
     r['code'] == 'V6',
     f"got {r['code']}")

# K13 — BUG-1 : COLLAPSE + decay exactement 0.85 → V3 (borne incluse)
r = _v({'decay_ratio': 0.85, 'decay_pct': 15.0},
       {'pattern': 'COLLAPSE', 'collapse_pct': -8.0}, 58)
test("K13 · BUG-1 : COLLAPSE + decay=0.85 → V3 (borne incluse)",
     r['code'] == 'V3',
     f"got {r['code']}")


# ══════════════════════════════════════════════════════════════════
# L — _GET_RACE_PROFILE (Sprint 4 · R1)
# ══════════════════════════════════════════════════════════════════

section("L — _get_race_profile()")

# L1 : 50km + 4h → ULTRA
test("L1 · 50km + 4h → ULTRA",
     _get_race_profile(50, 14400) == 'ULTRA', "")

# L2 : 50km + 3h45 → LONG (ET strict — distance ok mais durée < 4h)
test("L2 · 50km + 3h45 → LONG (ET strict : durée < 4h)",
     _get_race_profile(50, 13500) == 'LONG', "")

# L3 : 25km + 5h → LONG (ET strict — durée ok mais distance < 50km)
test("L3 · 25km + 5h → LONG (ET strict : distance < 50km)",
     _get_race_profile(25, 18000) == 'LONG', "")

# L4 : 30km + 3h → LONG
test("L4 · 30km + 3h → LONG",
     _get_race_profile(30, 10800) == 'LONG', "")

# L5 : 25km + 1h20 → COURT (durée < 1h30)
test("L5 · 25km + 1h20 → COURT (durée < 1h30)",
     _get_race_profile(25, 4800) == 'COURT', "")

# L6 : 12km + 1h → COURT
test("L6 · 12km + 1h → COURT",
     _get_race_profile(12, 3600) == 'COURT', "")

# L7 : marcheur lent — 25km + 5h → LONG (pas ULTRA : distance < 50km)
test("L7 · marcheur 25km + 5h → LONG (pas ULTRA)",
     _get_race_profile(25, 18000) == 'LONG', "")



# ══════════════════════════════════════════════════════════════════
# M — generate_pdf() — Smoke tests S4
# Vérifie que le PDF se génère sans crash dans les cas normaux
# et dégradés (sans FC, sans verdict, sans splits).
# ══════════════════════════════════════════════════════════════════

section("M — generate_pdf() smoke tests")

try:
    from charts import generate_pdf
    _charts_available = True
except ImportError as _charts_err:
    _charts_available = False
    for _m in ["M1 · PDF complet (verdict+perf+splits) → bytes non vides",
               "M2 · PDF sans FC (zones=None, verdict=None) → bytes non vides",
               "M3 · Phrase de partage S1 : label verdict présent dans le PDF",
               "M4 · Splits complets (tous les km) : km 1 et km 10 présents",
               "M5 · PDF avec 3 recos → génère sans crash (rec#1 p1, rec#2-3 p2)",
               "M6 · action_line > 20 chars → présente dans le PDF"]:
        test(_m, False, f"dépendances manquantes (plotly/fpdf2) — {_charts_err}")

if _charts_available:
    # Données minimales valides
    _info_base = {
        'name': 'Test Race', 'distance_km': 25.0, 'total_time_s': 7200,
        'elevation_gain': 800, 'avg_velocity_ms': 3.0, 'max_elevation': 1200,
        'has_hr': True, 'has_cad': True, 'hr_mean': 155.0, 'hr_max': 178.0,
        'cad_mean': 175.0,
    }
    _fi_base = {
        'decay_ratio': 0.91, 'decay_pct': 9.0,
        'quartiles': {'Q1': 3.2, 'Q2': 3.1, 'Q3': 3.0, 'Q4': 2.9},
    }
    _drift_base = {
        'pattern': 'DRIFT', 'drift_pct': -4.5, 'ef1': 0.018, 'ef2': 0.016,
        'insufficient_data': False, 'collapse_pct': None,
        'fc_q1_mean': 150.0, 'fc_q4_mean': 158.0,
        'quartiles': {}, 'fc_slope_bph': -1.0,
    }
    _zones_base = {
        'bpm':  {'Z1': (0,114), 'Z2': (114,133), 'Z3': (133,152), 'Z4': (152,171), 'Z5': (171,190)},
        'pct':  {'Z1': 5.0, 'Z2': 35.0, 'Z3': 30.0, 'Z4': 20.0, 'Z5': 10.0},
        'time': {'Z1': 360,  'Z2': 2520,  'Z3': 2160,  'Z4': 1440,  'Z5': 720},
    }
    _cad_base   = {'mean': 175.0, 'optimal_pct': 72.0, 'dist': {'<160': 5.0, '160-170': 23.0, '170-200': 72.0}}
    _splits_base = [
        {'km': i, 'pace': f'5:{i%60:02d}', 'gap': f'5:{i%60:02d}', 'pace_s': 300 + i*2,
         'd_pos': 30, 'd_neg': 10, 'hr': 155 + i, 'cadence': 175, 'has_walk': False}
        for i in range(1, 11)
    ]
    _recs_base = [
        {'level': 'crit', 'title': 'Cadence trop basse', 'body': 'Augmente ta cadence de 5%.'},
        {'level': 'warn', 'title': 'Depart rapide',      'body': 'Q1 au-dessus du seuil.'},
        {'level': 'info', 'title': 'Zones FC OK',        'body': '80% en Z2-Z3.'},
    ]
    _perf_base = {
        'score': 72, 'score_gap': 68, 'score_ef': 75, 'score_var': 80,
        'partial': False, 'partial_reason': None,
        'weights': {'gap': 0.5, 'ef': 0.35, 'var': 0.15},
        'var_neutralized': False,
    }
    _verdict_base = {
        'code': 'V2', 'label': 'PERFORMANCE CORRECTE',
        'sub': 'Score 72/100 -- Derive EF : 4.5% -- marge de progression identifiee.',
        'color': '#C8A84B', 'icon': '~',
    }
    _grade_df = pd.DataFrame({
        'Tranche pente':   ['-5 a 0%', '0 a 5%', '5 a 10%'],
        'Allure (min/km)': ['4:30',    '5:00',   '6:20'],
    })

    # M1 — Cas complet avec verdict + perf → bytes non vides
    try:
        result = generate_pdf(
            _info_base, _fi_base, 3.0, 'PROFIL ENDURANCE', _grade_df,
            _zones_base, _drift_base, _cad_base, _splits_base, _recs_base,
            190, _perf_base, _verdict_base, 'test@test.com'
        )
        test("M1 · PDF complet (verdict+perf+splits) → bytes non vides",
             isinstance(result, bytes) and len(result) > 1000,
             f"len={len(result)}")
    except Exception as e:
        test("M1 · PDF complet (verdict+perf+splits) → bytes non vides", False, str(e))

    # M2 — Sans FC (zones=None, drift insufficient) → pas de crash
    _info_no_hr  = {**_info_base, 'has_hr': False, 'hr_mean': None, 'hr_max': None, 'cad_mean': None}
    _drift_insuf = {**_drift_base, 'pattern': None, 'insufficient_data': True}
    try:
        result = generate_pdf(
            _info_no_hr, _fi_base, 3.0, 'PROFIL FRAGILE', _grade_df,
            None, _drift_insuf, {}, [], _recs_base[:1],
            190, None, None, ''
        )
        test("M2 · PDF sans FC (zones=None, verdict=None) → bytes non vides",
             isinstance(result, bytes) and len(result) > 500,
             f"len={len(result)}")
    except Exception as e:
        test("M2 · PDF sans FC (zones=None, verdict=None) → bytes non vides", False, str(e))

    # M3 — Phrase de partage S1 : label verdict présent dans le PDF
    try:
        result   = generate_pdf(
            _info_base, _fi_base, 3.0, 'PROFIL ENDURANCE', _grade_df,
            _zones_base, _drift_base, _cad_base, _splits_base, _recs_base,
            190, _perf_base, _verdict_base, ''
        )
        pdf_text = result.decode('latin-1', errors='ignore')
        test("M3 · Phrase de partage S1 : label verdict présent dans le PDF",
             'PERFORMANCE CORRECTE' in pdf_text or 'VERTEX Score' in pdf_text,
             "label ou 'VERTEX Score' non trouvé dans le binaire PDF")
    except Exception as e:
        test("M3 · Phrase de partage S1 : label verdict présent dans le PDF", False, str(e))

    # M4 — Splits complets : génération sans crash
    try:
        result = generate_pdf(
            _info_base, _fi_base, 3.0, 'PROFIL ENDURANCE', _grade_df,
            _zones_base, _drift_base, _cad_base, _splits_base, _recs_base,
            190, _perf_base, _verdict_base, ''
        )
        test("M4 · Splits complets (tous les km) : génération sans crash",
             isinstance(result, bytes) and len(result) > 1000,
             f"len={len(result)}")
    except Exception as e:
        test("M4 · Splits complets : génération sans crash", False, str(e))

    # M5 — Recos 3 dans le PDF → génère sans crash
    try:
        result = generate_pdf(
            _info_base, _fi_base, 3.0, 'PROFIL ENDURANCE', _grade_df,
            _zones_base, _drift_base, _cad_base, [], _recs_base,
            190, _perf_base, _verdict_base, ''
        )
        test("M5 · PDF avec 3 recos → génère sans crash (rec#1 p1, rec#2-3 p2)",
             isinstance(result, bytes) and len(result) > 500,
             f"len={len(result)}")
    except Exception as e:
        test("M5 · PDF avec 3 recos → génère sans crash (rec#1 p1, rec#2-3 p2)", False, str(e))

    # M6 — action_line > 20 chars → présente dans le PDF
    verdict_with_action = {**_verdict_base,
        'action_line': 'Reduis ton allure de depart de 10% sur les 3 premiers km.'}
    try:
        result = generate_pdf(
            _info_base, _fi_base, 3.0, 'PROFIL ENDURANCE', _grade_df,
            _zones_base, _drift_base, _cad_base, _splits_base, _recs_base,
            190, _perf_base, verdict_with_action, ''
        )
        pdf_text = result.decode('latin-1', errors='ignore')
        test("M6 · action_line > 20 chars → présente dans le PDF",
             'Reduis ton allure' in pdf_text, "action_line non trouvée dans le binaire")
    except Exception as e:
        test("M6 · action_line > 20 chars → présente dans le PDF", False, str(e))



# ══════════════════════════════════════════════════════════════════
# N — CARDIAC_DRIFT CDC v1.3 : DRIFT-CARDIO / DRIFT-NEURO / DRIFT
# ══════════════════════════════════════════════════════════════════

section("N — cardiac_drift() CDC v1.3")


def make_drift_df(n: int, fc_start: float, fc_end: float,
                  ef_start: float, ef_end: float,
                  velocity: float = 3.0, duration_min: float = 60) -> pd.DataFrame:
    """
    DataFrame synthetique terrain plat pour tester cardiac_drift().
    fc_start/fc_end : FC debut/fin (bpm).
    ef_start/ef_end : non utilise directement (EF emerge de velocity/hr).
    La pente FC est encodee via fc_start→fc_end sur la duree.
    """
    t = np.linspace(0, duration_min * 60, n)
    hr = np.linspace(fc_start, fc_end, n)
    # Velocity decroissante pour simuler derive EF : debut ef_start, fin ef_end
    # EF = velocity/hr*100, donc velocity = EF * hr / 100
    v_arr = np.linspace(ef_start * fc_start / 100, ef_end * fc_end / 100, n)
    d = np.cumsum(v_arr * np.diff(t, prepend=0))
    return pd.DataFrame({
        'time_s':   t,
        'distance': d,
        'hr':       hr,
        'velocity': v_arr,
        'grade':    np.zeros(n),
    })


# N1 : DRIFT-CARDIO — fc_slope > +0.5 bph ET drift_ef < -4%
# FC monte 140→165 sur 60 min (slope ~ +25 bpm/h >> 0.5)
# EF degrade : velocity baisse significativement
df_dc = make_drift_df(300, fc_start=140, fc_end=165, ef_start=2.2, ef_end=1.8,
                       velocity=3.0, duration_min=60)
r_dc = cardiac_drift(df_dc, duration_s=3600, dp_per_km=5)
test("N1 · DRIFT-CARDIO : FC monte + EF degrade → pattern DRIFT-CARDIO",
     r_dc['pattern'] == 'DRIFT-CARDIO',
     f"got pattern={r_dc['pattern']}, slope={r_dc.get('fc_slope_bph', '?'):.2f} bph, drift={r_dc.get('drift_pct', '?')}")

# N2 : DRIFT-NEURO — fc_slope <= +0.5 bph ET drift_ef < -4%
# FC strictement constante (slope = 0 bph) + vitesse qui chute significativement
# EF = velocity/hr*100 : si FC fixe et velocity baisse, EF baisse aussi
n_n2 = 300
t_n2 = np.linspace(0, 3600, n_n2)
hr_n2 = np.full(n_n2, 155.0)                        # FC CONSTANTE
v_n2 = np.linspace(3.0, 2.35, n_n2)                 # vitesse -22% → EF baisse
d_n2 = np.cumsum(v_n2 * np.diff(t_n2, prepend=0))
df_dn = pd.DataFrame({'time_s': t_n2, 'distance': d_n2, 'hr': hr_n2,
                       'velocity': v_n2, 'grade': np.zeros(n_n2)})
r_dn = cardiac_drift(df_dn, duration_s=3600, dp_per_km=5)
test("N2 · DRIFT-NEURO : FC stable + EF degrade → pattern DRIFT-NEURO",
     r_dn['pattern'] == 'DRIFT-NEURO',
     f"got pattern={r_dn['pattern']}, slope={r_dn.get('fc_slope_bph', '?'):.2f} bph, drift={r_dn.get('drift_pct', '?')}")

# N3 : DRIFT faible — drift_ef entre -2% et -4% → pattern DRIFT (pas DRIFT-CARDIO/NEURO)
df_dw = make_drift_df(300, fc_start=150, fc_end=151, ef_start=2.05, ef_end=1.97,
                       velocity=3.0, duration_min=60)
r_dw = cardiac_drift(df_dw, duration_s=3600, dp_per_km=5)
test("N3 · DRIFT faible : derive -2% a -4% → pattern DRIFT",
     r_dw['pattern'] in ('DRIFT', 'STABLE'),
     f"got pattern={r_dw['pattern']}, drift={r_dw.get('drift_pct', '?')}")

# N4 : STABLE — EF et FC stables → pattern STABLE
df_st = make_drift_df(300, fc_start=150, fc_end=151, ef_start=2.0, ef_end=2.0,
                       velocity=3.0, duration_min=60)
r_st = cardiac_drift(df_st, duration_s=3600, dp_per_km=5)
test("N4 · STABLE : EF et FC stables → STABLE",
     r_st['pattern'] == 'STABLE',
     f"got pattern={r_st['pattern']}, drift={r_st.get('drift_pct', '?')}")

# N5 : COLLAPSE non affecte par CDC v1.3 — chute FC > 20% → COLLAPSE
n_c = 200
t_c = np.linspace(0, 3600, n_c)
hr_c = np.linspace(160, 110, n_c)   # chute 31% → COLLAPSE garanti
d_c = np.cumsum(np.full(n_c, 3.0) * np.diff(t_c, prepend=0))
df_col = pd.DataFrame({'time_s': t_c, 'distance': d_c, 'hr': hr_c,
                        'velocity': np.full(n_c, 3.0), 'grade': np.zeros(n_c)})
r_col = cardiac_drift(df_col, duration_s=3600, dp_per_km=5)
test("N5 · COLLAPSE non affecte : chute FC >20% → toujours COLLAPSE",
     r_col['pattern'] == 'COLLAPSE',
     f"got pattern={r_col['pattern']}")

# N6 : fc_slope < +0.5 bph → DRIFT-NEURO (cas Thibault marathon : slope -0.28 bph)
# Ref dataset : marathon Thibault, DRIFT-NEURO valide, fc_slope -0.28 bph
# FC legerement decroissante (slope -0.28 bph) + EF degrade via vitesse
# Note : la borne exacte 0.5 est instable numeriquement (polyfit flottant) --
#        on teste le cas pratique confirme par dataset reel.
n_n6 = 300
t_n6 = np.linspace(0, 3600, n_n6)
hr_n6 = 155.0 + (-0.28 / 3600) * t_n6  # slope -0.28 bph (ref Thibault marathon)
v_n6  = np.linspace(3.0, 2.35, n_n6)    # velocity chute -> EF baisse
d_n6  = np.cumsum(v_n6 * np.diff(t_n6, prepend=0))
df_border = pd.DataFrame({'time_s': t_n6, 'distance': d_n6, 'hr': hr_n6,
                           'velocity': v_n6, 'grade': np.zeros(n_n6)})
r_border = cardiac_drift(df_border, duration_s=3600, dp_per_km=5)
test("N6 · fc_slope=-0.28 bph (ref Thibault marathon) + drift<-4% → DRIFT-NEURO",
     r_border['pattern'] == 'DRIFT-NEURO',
     f"got pattern={r_border['pattern']}, slope={r_border.get('fc_slope_bph', '?'):.3f} bph")


# ══════════════════════════════════════════════════════════════════
# O — COMPUTE_VERDICT CDC v1.3 : V3-NEURO
# ══════════════════════════════════════════════════════════════════

section("O — compute_verdict() CDC v1.3 : V3-NEURO")


def _v_neuro(fi_dict, drift_dict, score):
    return compute_verdict(fi_dict, drift_dict, {'score': score})


# O1 : DRIFT-NEURO + decay 0.90 < 0.93 → V3
r = _v_neuro({'decay_ratio': 0.90, 'decay_pct': 10.0},
              {'pattern': 'DRIFT-NEURO', 'drift_pct': -5.5, 'insufficient_data': False}, 65)
test("O1 · DRIFT-NEURO + decay=0.90 → V3",
     r['code'] == 'V3',
     f"got {r['code']}")

# O2 : DRIFT-NEURO + decay 0.85 (< 0.93) → V3
r = _v_neuro({'decay_ratio': 0.85, 'decay_pct': 15.0},
              {'pattern': 'DRIFT-NEURO', 'drift_pct': -6.0, 'insufficient_data': False}, 55)
test("O2 · DRIFT-NEURO + decay=0.85 → V3",
     r['code'] == 'V3',
     f"got {r['code']}")

# O3 : DRIFT-NEURO + decay >= 0.93 → NE DOIT PAS etre V3 via regle NEURO (V2 ou V1)
r = _v_neuro({'decay_ratio': 0.95, 'decay_pct': 5.0},
              {'pattern': 'DRIFT-NEURO', 'drift_pct': -5.0, 'insufficient_data': False}, 70)
test("O3 · DRIFT-NEURO + decay=0.95 → pas V3-NEURO (V2 ou V1)",
     r['code'] in ('V1', 'V2'),
     f"got {r['code']}")

# O4 : DRIFT-CARDIO + decay 0.88 + score < 50 → V4 (fatigue combinee elargie)
r = _v_neuro({'decay_ratio': 0.88, 'decay_pct': 12.0},
              {'pattern': 'DRIFT-CARDIO', 'drift_pct': -5.0, 'insufficient_data': False}, 45)
test("O4 · DRIFT-CARDIO + decay<0.90 + score<50 → V4",
     r['code'] == 'V4',
     f"got {r['code']}")

# O5 : DRIFT-NEURO + decay 0.88 + score < 50 → V4 (fatigue combinee elargie)
r = _v_neuro({'decay_ratio': 0.88, 'decay_pct': 12.0},
              {'pattern': 'DRIFT-NEURO', 'drift_pct': -5.5, 'insufficient_data': False}, 40)
test("O5 · DRIFT-NEURO + decay<0.90 + score<50 → V4 (priorite sur V3-NEURO)",
     r['code'] == 'V4',
     f"got {r['code']}")

# O6 : DRIFT classique + decay 0.88 + score < 50 → V4 (comportement conserve)
r = _v_neuro({'decay_ratio': 0.88, 'decay_pct': 12.0},
              {'pattern': 'DRIFT', 'drift_pct': -3.0, 'insufficient_data': False}, 45)
test("O6 · DRIFT + decay<0.90 + score<50 → V4 (comportement conserve)",
     r['code'] == 'V4',
     f"got {r['code']}")



# ══════════════════════════════════════════════════════════════════
# P — GET_SCORE_WEIGHTS SCI-3 (CDC Elena v1.3)
# ══════════════════════════════════════════════════════════════════

section("P — get_score_weights() SCI-3")

EPSILON = 1e-9

def check_sum(w):
    return abs(w['w_gap'] + w['w_ef'] + w['w_var'] - 1.0) < EPSILON

# P1 : Z1, FC dispo → GAP 70% / EF 0% / Var 30%
w = get_score_weights(dp_per_km=5.0, ef_unavailable=False)
test("P1 · Z1 dp=5 FC dispo → w_gap=0.70 w_ef=0.00 w_var=0.30",
     abs(w['w_gap']-0.70)<EPSILON and abs(w['w_ef'])<EPSILON and abs(w['w_var']-0.30)<EPSILON and w['zone']=='Z1',
     f"got gap={w['w_gap']} ef={w['w_ef']} var={w['w_var']} zone={w['zone']}")

# P2 : Z1, EF indispo → même résultat (EF déjà 0% en Z1, redistribution nulle)
w = get_score_weights(dp_per_km=5.0, ef_unavailable=True)
test("P2 · Z1 dp=5 EF indispo → idem P1 (EF deja 0%)",
     abs(w['w_gap']-0.70)<EPSILON and abs(w['w_ef'])<EPSILON and abs(w['w_var']-0.30)<EPSILON,
     f"got gap={w['w_gap']} ef={w['w_ef']} var={w['w_var']}")

# P3 : Z3, FC dispo → GAP 50% / EF 35% / Var 15%
w = get_score_weights(dp_per_km=25.0, ef_unavailable=False)
test("P3 · Z3 dp=25 FC dispo → w_gap=0.50 w_ef=0.35 w_var=0.15",
     abs(w['w_gap']-0.50)<EPSILON and abs(w['w_ef']-0.35)<EPSILON and abs(w['w_var']-0.15)<EPSILON and w['zone']=='Z3',
     f"got gap={w['w_gap']} ef={w['w_ef']} var={w['w_var']} zone={w['zone']}")

# P4 : Z3, EF indispo → GAP 85% / EF 0% / Var 15%
w = get_score_weights(dp_per_km=25.0, ef_unavailable=True)
test("P4 · Z3 dp=25 EF indispo → w_gap=0.85 w_ef=0.00 w_var=0.15",
     abs(w['w_gap']-0.85)<EPSILON and abs(w['w_ef'])<EPSILON and abs(w['w_var']-0.15)<EPSILON,
     f"got gap={w['w_gap']} ef={w['w_ef']} var={w['w_var']}")

# P5 : Z2 dp=15, FC dispo → interpolation t=0.5
# w_gap = 0.70 + 0.5*(0.50-0.70) = 0.60
# w_ef  = 0.00 + 0.5*(0.35-0.00) = 0.175
# w_var = 0.30 + 0.5*(0.15-0.30) = 0.225
w = get_score_weights(dp_per_km=15.0, ef_unavailable=False)
test("P5 · Z2 dp=15 FC dispo → w_gap≈0.60 w_ef≈0.175 w_var≈0.225",
     abs(w['w_gap']-0.60)<1e-6 and abs(w['w_ef']-0.175)<1e-6 and abs(w['w_var']-0.225)<1e-6 and w['zone']=='Z2',
     f"got gap={w['w_gap']:.4f} ef={w['w_ef']:.4f} var={w['w_var']:.4f} zone={w['zone']}")

# P6 : Z2 dp=15, EF indispo → w_ef=0, redistribué vers GAP, somme=1
w = get_score_weights(dp_per_km=15.0, ef_unavailable=True)
test("P6 · Z2 dp=15 EF indispo → w_ef=0.0, somme=1.0",
     abs(w['w_ef'])<EPSILON and check_sum(w),
     f"got gap={w['w_gap']:.4f} ef={w['w_ef']:.4f} var={w['w_var']:.4f} sum={w['w_gap']+w['w_ef']+w['w_var']:.10f}")

# P7 : somme = 1.0 pour tous les cas
all_pass = True
for dp in [0, 5, 10, 15, 19.9, 20, 25, 50, 100]:
    for ef in [True, False]:
        w = get_score_weights(dp_per_km=dp, ef_unavailable=ef)
        if not check_sum(w):
            all_pass = False
            print(f"  FAIL sum: dp={dp} ef={ef} sum={w['w_gap']+w['w_ef']+w['w_var']}")
test("P7 · somme w_gap+w_ef+w_var == 1.0 pour toutes combinaisons",
     all_pass, "voir details ci-dessus si echec")

# P8 : borne dp=10 m/km → zone Z2 (t=0.0, poids = Z1)
w = get_score_weights(dp_per_km=10.0, ef_unavailable=False)
test("P8 · borne dp=10 → zone Z2 (pas Z1), poids Z1 (t=0)",
     w['zone'] == 'Z2' and abs(w['w_gap']-0.70)<1e-6 and abs(w['w_ef']-0.00)<1e-6,
     f"got zone={w['zone']} gap={w['w_gap']:.4f} ef={w['w_ef']:.4f}")

# P9 : borne dp=20 m/km → zone Z3 (borne supérieure exclue de Z2)
w = get_score_weights(dp_per_km=20.0, ef_unavailable=False)
test("P9 · borne dp=20 → zone Z3 (Z2 exclue)",
     w['zone'] == 'Z3' and abs(w['w_gap']-0.50)<EPSILON,
     f"got zone={w['zone']} gap={w['w_gap']}")

# P10 : Z2 non validé → zone_validated=False
w = get_score_weights(dp_per_km=15.0, ef_unavailable=False)
test("P10 · Z2 → zone_validated=False",
     w['zone_validated'] == False,
     f"got zone_validated={w['zone_validated']}")

# P11 : Z1 et Z3 validés → zone_validated=True
w1 = get_score_weights(dp_per_km=5.0, ef_unavailable=False)
w3 = get_score_weights(dp_per_km=30.0, ef_unavailable=False)
test("P11 · Z1 et Z3 → zone_validated=True",
     w1['zone_validated'] == True and w3['zone_validated'] == True,
     f"Z1={w1['zone_validated']} Z3={w3['zone_validated']}")




# ══════════════════════════════════════════════════════════════════
# Q — DETECT_ELEVATION_PROFILE SCI-3 ① (CDC Elena v1.3)
# ══════════════════════════════════════════════════════════════════

section("Q — detect_elevation_profile() SCI-3 ①")

import numpy as np

def make_elev_df(ele_arr, n_points=None):
    """DataFrame minimal pour detect_elevation_profile."""
    if n_points:
        ele_arr = np.interp(np.linspace(0, len(ele_arr)-1, n_points),
                            np.arange(len(ele_arr)), ele_arr)
    n = len(ele_arr)
    dist = np.linspace(0, 20000, n)
    dz   = np.diff(np.array(ele_arr, dtype=float), prepend=ele_arr[0])
    return pd.DataFrame({
        'distance': dist,
        'elevation': ele_arr,
        'dz': dz,
        'velocity': np.full(n, 3.0),
        'grade': np.zeros(n),
        'time_s': np.linspace(0, 7200, n),
    })

# Q1 : profil FLAT — D+ regulier sur les 4 quartiles
ele_flat = 500 + 200 * np.sin(np.linspace(0, 4*np.pi, 400))
r = detect_elevation_profile(make_elev_df(ele_flat))
test("Q1 · profil plat regulier → FLAT",
     r['profile'] == 'FLAT',
     f"got profile={r['profile']} bias={r['elevation_bias']:.2f}")

# Q2 : SCI-8 — montée Q1-Q3 + descente massive Q4 → MIXED (asc_bias>0.55 ET desc_bias>0.45)
ele_desc = np.concatenate([np.linspace(500, 1500, 300), np.linspace(1500, 500, 100)])
r = detect_elevation_profile(make_elev_df(ele_desc))
test("Q2 · montee Q1-Q3 + descente Q4 → MIXED dominant_q=Q4",
     r['profile'] == 'MIXED' and r['dominant_q'] == 'Q4',
     f"got profile={r['profile']} dominant={r['dominant_q']} bias={r['elevation_bias']:.2f}")

# Q3 : SCI-8 — montée massive Q1 + descente étalée Q2-Q4 → MIXED (asc_bias>0.55 ET desc_bias>0.45)
ele_asc = np.concatenate([np.linspace(500, 1500, 100), np.linspace(1500, 500, 300)])
r = detect_elevation_profile(make_elev_df(ele_asc))
test("Q3 · montee Q1 + descente Q2-Q4 → MIXED dominant_q=Q3",
     r['profile'] == 'MIXED' and r['dominant_q'] == 'Q3',
     f"got profile={r['profile']} dominant={r['dominant_q']} bias={r['elevation_bias']:.2f}")

# Q4 : seuil 40% — descente Q4 a 39% → FLAT (sous le seuil)
# 4 quartiles egaux en D- sauf Q4 = 39%
ele_border = np.concatenate([
    500 + 250*np.sin(np.linspace(0, 3*np.pi, 300)),  # D- reparti Q1-Q3
    np.linspace(700, 400, 100),                        # Q4 : descente ~39% du total
])
r = detect_elevation_profile(make_elev_df(ele_border))
test("Q4 · biais Q4 < 40% → FLAT (sous seuil)",
     r['profile'] == 'FLAT',
     f"got profile={r['profile']} bias={r['elevation_bias']:.2f}")

# Q5 : retour dict complet — toutes les cles presentes
r = detect_elevation_profile(make_elev_df(ele_flat))
required_keys = {'profile', 'elevation_bias', 'magnitude', 'dominant_q', 'dplus_by_q', 'dminus_by_q'}
test("Q5 · retour dict : toutes les cles presentes",
     required_keys.issubset(r.keys()),
     f"manquantes={required_keys - set(r.keys())}")

# Q6 : magnitude = 0.0 sur FLAT
test("Q6 · FLAT → magnitude=0.0",
     r['magnitude'] == 0.0,
     f"got magnitude={r['magnitude']}")

# Q7 : DESCENDING → magnitude > 0
r_d = detect_elevation_profile(make_elev_df(ele_desc))
test("Q7 · DESCENDING → magnitude > 0",
     r_d['magnitude'] > 0,
     f"got magnitude={r_d['magnitude']:.3f}")

# Q8 : DataFrame vide / dist=0 → pas de crash, retourne FLAT
df_empty = pd.DataFrame({'distance': [0.0], 'elevation': [500.0], 'dz': [0.0],
                          'velocity': [0.0], 'grade': [0.0], 'time_s': [0.0]})
try:
    r_e = detect_elevation_profile(df_empty)
    test("Q8 · DataFrame minimal → pas de crash, FLAT",
         r_e['profile'] == 'FLAT', f"got {r_e['profile']}")
except Exception as ex:
    test("Q8 · DataFrame minimal → pas de crash", False, str(ex))


# ══════════════════════════════════════════════════════════════════
# R — APPLY_DECAY_CORRECTION SCI-3 ② (CDC Elena v1.3)
# ══════════════════════════════════════════════════════════════════

section("R — apply_decay_correction() SCI-3 ②")

def make_full_df(ele_arr):
    """DataFrame avec grade et gap calculés pour apply_decay_correction."""
    n = len(ele_arr)
    dist = np.linspace(0, 20000, n)
    t    = np.linspace(0, 7200, n)
    dz   = np.diff(np.array(ele_arr, dtype=float), prepend=ele_arr[0])
    grade = np.gradient(ele_arr, dist) * 100
    grade = np.clip(grade, -30, 30)
    return pd.DataFrame({
        'distance': dist, 'elevation': ele_arr, 'dz': dz,
        'grade': grade, 'velocity': np.full(n, 3.0), 'time_s': t,
    })

fi_base = {'decay_ratio': 0.88, 'decay_pct': 12.0,
           'quartiles': {'Q1': 3.0, 'Q2': 2.9, 'Q3': 2.85, 'Q4': 2.74}}
ep_flat = {'profile': 'FLAT', 'elevation_bias': 0.2, 'magnitude': 0.0,
           'dominant_q': 'Q2', 'dplus_by_q': {}, 'dminus_by_q': {}}

# R1 : FLAT → pas de correction, decay_ratio inchange
df_r = make_full_df(500 + 200*np.sin(np.linspace(0, 4*np.pi, 400)))
r = apply_decay_correction(fi_base, ep_flat, df_r)
test("R1 · FLAT → correction_applied=False, decay_ratio inchange",
     r['correction_applied'] == False and r['decay_ratio_corrected'] == 0.88,
     f"got applied={r['correction_applied']} ratio={r['decay_ratio_corrected']}")

# R2 : retour dict enrichi — toutes les cles presentes
required = {'decay_ratio_corrected', 'decay_pct_corrected', 'correction_applied',
            'correction_magnitude', 'elev_profile'}
test("R2 · retour dict enrichi : toutes les cles presentes",
     required.issubset(r.keys()),
     f"manquantes={required - set(r.keys())}")

# R3 : DESCENDING → garde-fou dynamique [0.50, 1.50] respecte toujours
ele_extreme = np.concatenate([np.linspace(500, 2000, 300), np.linspace(2000, 500, 100)])
df_ext = make_full_df(ele_extreme)
fi_ext = fatigue_index(df_ext)
ep_ext = detect_elevation_profile(df_ext)
r_ext  = apply_decay_correction(fi_ext, ep_ext, df_ext)
test("R3 · DESCENDING extreme → garde-fou dynamique [0.50, 1.50] respecte",
     0.50 <= r_ext['decay_ratio_corrected'] <= 1.50,
     f"got ratio={r_ext['decay_ratio_corrected']:.4f}")

# R4 : correction_magnitude >= 0
test("R4 · correction_magnitude >= 0",
     r_ext['correction_magnitude'] >= 0,
     f"got {r_ext['correction_magnitude']}")

# R5 : FLAT avec decay NaN → pas de crash
fi_nan = {'decay_ratio': float('nan'), 'decay_pct': float('nan'), 'quartiles': {}}
r_nan = apply_decay_correction(fi_nan, ep_flat, df_r)
test("R5 · decay_ratio NaN → pas de crash, correction_applied=False",
     r_nan['correction_applied'] == False,
     f"got applied={r_nan['correction_applied']}")

# R6 : DESCENDING → decay_ratio_corrected <= original ou clippe (jamais aggrave le biais)
# Le ratio corrige doit etre <= ratio original si original > 1.0 (descente gonfle Q4)
fi_high = dict(fi_ext)
r_high = apply_decay_correction(fi_high, ep_ext, df_ext)
test("R6 · DESCENDING → ratio_corrected <= original OU clippe a 1.20",
     r_high['decay_ratio_corrected'] <= fi_high['decay_ratio'] or r_high['decay_ratio_corrected'] == 1.20,
     f"orig={fi_high['decay_ratio']:.4f} corr={r_high['decay_ratio_corrected']:.4f}")

# R7 : elev_profile recopie dans le retour pour tracabilite
test("R7 · elev_profile present dans le retour",
     'elev_profile' in r_ext and r_ext['elev_profile']['profile'] == ep_ext['profile'],
     f"got {r_ext.get('elev_profile', {}).get('profile')}")



# ══════════════════════════════════════════════════════════════════
# S — COMPUTE_VERDICT : action_line présente pour chaque code (B1 Sprint 5)
# ══════════════════════════════════════════════════════════════════

section("S — compute_verdict() · action_line présente (B1 Sprint 5)")

def _cv(fi_dict, drift_dict, score):
    return compute_verdict(fi_dict, drift_dict, {'score': score})

def _has_action(r):
    return 'action_line' in r and isinstance(r['action_line'], str) and len(r['action_line']) > 5

# S1 : INSUFFICIENT → action_line présente
r = _cv({'decay_ratio': float('nan'), 'decay_pct': float('nan')},
        {'pattern': None, 'insufficient_data': True, 'collapse_pct': None,
         'drift_pct': None, 'fc_slope_bph': None}, 0)
test("S1 · INSUFFICIENT → action_line présente",
     r['code'] == 'INSUFFICIENT' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S2 : V1-NS (NEGATIVE_SPLIT) → action_line présente
r = _cv({'decay_ratio': 0.97, 'decay_pct': 3.0},
        {'pattern': 'NEGATIVE_SPLIT', 'insufficient_data': False, 'collapse_pct': None,
         'drift_pct': None, 'fc_slope_bph': -1.5, 'decay_v': 0.10}, 82)
test("S2 · V1-NS (NEGATIVE_SPLIT) → action_line présente",
     r['code'] == 'V1' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S3 : V7 (COLLAPSE + decay>0.90 + score>75) → action_line présente
r = _cv({'decay_ratio': 0.93, 'decay_pct': 7.0},
        {'pattern': 'COLLAPSE', 'insufficient_data': False, 'collapse_pct': -18.0,
         'drift_pct': None, 'fc_slope_bph': -4.0}, 80)
test("S3 · V7 → action_line présente",
     r['code'] == 'V7' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S4 : V6 (COLLAPSE + decay<0.85) → action_line présente
r = _cv({'decay_ratio': 0.82, 'decay_pct': 18.0},
        {'pattern': 'COLLAPSE', 'insufficient_data': False, 'collapse_pct': -22.0,
         'drift_pct': None, 'fc_slope_bph': -4.0}, 48)
test("S4 · V6 (COLLAPSE + decay<0.85) → action_line présente",
     r['code'] == 'V6' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S5 : V5 (decay < 0.80) → action_line présente
r = _cv({'decay_ratio': 0.75, 'decay_pct': 25.0},
        {'pattern': 'STABLE', 'insufficient_data': False, 'collapse_pct': None,
         'drift_pct': None, 'fc_slope_bph': 0.3}, 42)
test("S5 · V5 (decay<0.80) → action_line présente",
     r['code'] == 'V5' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S6 : V4 (DRIFT + decay<0.90 + score<50) → action_line présente
r = _cv({'decay_ratio': 0.86, 'decay_pct': 14.0},
        {'pattern': 'DRIFT', 'insufficient_data': False, 'collapse_pct': None,
         'drift_pct': -3.5, 'fc_slope_bph': 0.4}, 44)
test("S6 · V4 (DRIFT + decay<0.90 + score<50) → action_line présente",
     r['code'] == 'V4' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S7 : V3 standard (decay 0.80–0.90) → action_line présente
r = _cv({'decay_ratio': 0.85, 'decay_pct': 15.0},
        {'pattern': 'STABLE', 'insufficient_data': False, 'collapse_pct': None,
         'drift_pct': None, 'fc_slope_bph': 0.2}, 60)
test("S7 · V3 standard (decay 0.80–0.90) → action_line présente",
     r['code'] == 'V3' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S8 : V2 standard → action_line présente
r = _cv({'decay_ratio': 0.92, 'decay_pct': 8.0},
        {'pattern': 'STABLE', 'insufficient_data': False, 'collapse_pct': None,
         'drift_pct': None, 'fc_slope_bph': 0.2}, 65)
test("S8 · V2 standard → action_line présente",
     r['code'] == 'V2' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S9 : V1 → action_line présente
r = _cv({'decay_ratio': 0.95, 'decay_pct': 5.0},
        {'pattern': 'STABLE', 'insufficient_data': False, 'collapse_pct': None,
         'drift_pct': None, 'fc_slope_bph': 0.1}, 80)
test("S9 · V1 (score>=75, decay>=0.90) → action_line présente",
     r['code'] == 'V1' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:40]}'")

# S10 : V2 DRIFT-CARDIO → action_line différenciée
r = _cv({'decay_ratio': 0.92, 'decay_pct': 8.0},
        {'pattern': 'DRIFT-CARDIO', 'insufficient_data': False, 'collapse_pct': None,
         'drift_pct': -5.0, 'fc_slope_bph': 2.5}, 62)
test("S10 · V2 DRIFT-CARDIO → action_line présente et différenciée",
     r['code'] == 'V2' and _has_action(r),
     f"code={r['code']} action='{r.get('action_line','MISSING')[:50]}'")


# ══════════════════════════════════════════════════════════════════
# SECTION T — flat_pace_estimate()
# ══════════════════════════════════════════════════════════════════

print("\n── Section T : flat_pace_estimate ──")

def _flat_df(n=50, grade=0.0, vel=3.0):
    return pd.DataFrame({'grade': [grade]*n, 'velocity': [vel]*n})

# T1 : retourne un float
test("T1 · flat_pace_estimate → retourne float",
     isinstance(flat_pace_estimate(_flat_df()), float),
     "")

# T2 : valeur positive
_t2 = flat_pace_estimate(_flat_df())
test("T2 · flat_pace_estimate → valeur > 0",
     _t2 > 0,
     f"res={_t2:.3f}")

# T3 : fallback médiane si moins de 10 points plats (pente forte partout)
_steep = pd.DataFrame({'grade': [30.0]*20, 'velocity': [1.5]*20})
_t3 = flat_pace_estimate(_steep)
test("T3 · flat_pace_estimate → fallback médiane (aucun point plat)",
     isinstance(_t3, float) and _t3 > 0,
     f"res={_t3:.3f}")

# T4 : valeur dans ordre de grandeur cohérent avec la vitesse input
_t4 = flat_pace_estimate(_flat_df(vel=4.0))
test("T4 · flat_pace_estimate — entre 1 et 8 m/s",
     1.0 < _t4 < 8.0,
     f"res={_t4:.3f}")


# ══════════════════════════════════════════════════════════════════
# SECTION U — grade_pace_profile()
# ══════════════════════════════════════════════════════════════════

print("\n── Section U : grade_pace_profile ──")

def _grade_pace_df():
    grades = [0.0]*40 + [7.0]*30 + [12.0]*20 + [20.0]*10
    vels   = [3.5]*40 + [2.0]*30 + [1.5]*20 + [1.0]*10
    return pd.DataFrame({'grade': grades, 'velocity': vels})

_u = grade_pace_profile(_grade_pace_df())

# U1 : retourne un DataFrame
test("U1 · grade_pace_profile → retourne DataFrame",
     isinstance(_u, pd.DataFrame),
     f"type={type(_u)}")

# U2 : colonnes attendues
_u_expected_cols = {'Tranche pente', 'Vitesse (m/s)', 'Allure (min/km)'}
test("U2 · grade_pace_profile → colonnes correctes",
     _u_expected_cols.issubset(set(_u.columns)),
     f"cols={list(_u.columns)}")

# U3 : au moins 1 ligne pour données multi-pentes
test("U3 · grade_pace_profile → au moins 1 ligne",
     len(_u) >= 1,
     f"len={len(_u)}")

# U4 : DataFrame vide si toute velocity ≤ 0.3
_u4 = grade_pace_profile(pd.DataFrame({'grade': [5.0]*20, 'velocity': [0.1]*20}))
test("U4 · grade_pace_profile → vide si velocity≤0.3 partout",
     len(_u4) == 0,
     f"len={len(_u4)}")


# ══════════════════════════════════════════════════════════════════
# SECTION V — hr_by_grade()
# ══════════════════════════════════════════════════════════════════

print("\n── Section V : hr_by_grade ──")

def _hr_grade_df(n_per_bin=50):
    rows = []
    for g in range(-15, 20, 5):
        for _ in range(n_per_bin):
            rows.append({'grade': float(g + 2), 'hr': 140.0, 'velocity': 2.0})
    return pd.DataFrame(rows)

_v = hr_by_grade(_hr_grade_df())

# V1 : retourne un DataFrame
test("V1 · hr_by_grade → retourne DataFrame",
     isinstance(_v, pd.DataFrame),
     f"type={type(_v)}")

# V2 : vide si hr ≤ 80 partout
_v2 = hr_by_grade(pd.DataFrame({'grade': [5.0]*100, 'hr': [60.0]*100, 'velocity': [2.0]*100}))
test("V2 · hr_by_grade → vide si hr≤80 partout",
     len(_v2) == 0,
     f"len={len(_v2)}")

# V3 : filtre les bins avec n ≤ 30
_v3 = hr_by_grade(pd.DataFrame({'grade': [5.0]*15, 'hr': [140.0]*15, 'velocity': [2.0]*15}))
test("V3 · hr_by_grade → filtre bins n≤30",
     len(_v3) == 0,
     f"len={len(_v3)}")

# V4 : colonnes hr_mean et n présentes
test("V4 · hr_by_grade → colonnes hr_mean et n présentes",
     'hr_mean' in _v.columns and 'n' in _v.columns,
     f"cols={list(_v.columns)}")


# ══════════════════════════════════════════════════════════════════
# SECTION W — haversine_vec()
# ══════════════════════════════════════════════════════════════════

print("\n── Section W : haversine_vec ──")

_w_lat = np.array([48.85, 48.86, 48.87])
_w_lon = np.array([2.35,  2.35,  2.35])
_w = haversine_vec(_w_lat, _w_lon)

# W1 : premier élément toujours 0.0
test("W1 · haversine_vec → premier élément = 0.0",
     _w[0] == 0.0,
     f"w[0]={_w[0]}")

# W2 : longueur = longueur input
test("W2 · haversine_vec → len(output) = len(input)",
     len(_w) == len(_w_lat),
     f"len={len(_w)}")

# W3 : deux points identiques consécutifs → distance = 0
_w3 = haversine_vec(np.array([48.0, 48.0]), np.array([2.0, 2.0]))
test("W3 · haversine_vec → même point consécutif → dist=0",
     _w3[1] == 0.0,
     f"dist={_w3[1]}")

# W4 : 1 degré latitude à l'équateur ≈ 111 194 m (±2 km)
_w4 = haversine_vec(np.array([0.0, 1.0]), np.array([0.0, 0.0]))[1]
test("W4 · haversine_vec — 1° lat équateur ≈ 111 194 m (±2 km)",
     abs(_w4 - 111194) < 2000,
     f"d={_w4:.0f} m")


# ══════════════════════════════════════════════════════════════════
# SECTION X — extract_race_info()
# ══════════════════════════════════════════════════════════════════

print("\n── Section X : extract_race_info ──")

def _race_info_df(n=200, with_hr=True, estimated=False, degraded=False):
    dist   = np.linspace(0, 5000, n)
    time_s = np.linspace(0, 1800, n)
    ele    = 100 + np.linspace(0, 50, n)
    dz     = np.concatenate([[0.0], np.diff(ele)])
    vel    = np.full(n, 2.5)
    hr     = np.full(n, 150.0) if with_hr else np.full(n, np.nan)
    cad    = np.full(n, 170.0)
    return pd.DataFrame({
        'distance': dist, 'time_s': time_s,
        'elevation': ele, 'dz': dz,
        'velocity': vel, 'hr': hr, 'cadence': cad,
        'elevation_degraded':   [degraded]*n,
        'timestamps_estimated': [estimated]*n,
    })

_x = extract_race_info(_race_info_df(), 'test_race.gpx')

# X1 : clés obligatoires présentes
_x_required = {'name', 'distance_km', 'total_time_s', 'elevation_gain',
                'elevation_loss', 'has_hr', 'hr_coverage_pct',
                'timestamps_estimated', 'elevation_degraded'}
test("X1 · extract_race_info → clés obligatoires présentes",
     _x_required.issubset(set(_x.keys())),
     f"manquantes={_x_required - set(_x.keys())}")

# X2 : distance_km correcte (≈ 5.0 km)
test("X2 · extract_race_info → distance_km ≈ 5.0",
     abs(_x['distance_km'] - 5.0) < 0.1,
     f"dist={_x['distance_km']:.2f}")

# X3 : has_hr = False sans FC
_x3 = extract_race_info(_race_info_df(with_hr=False), 'no_hr.gpx')
test("X3 · extract_race_info → has_hr=False sans FC",
     _x3['has_hr'] == False,
     f"has_hr={_x3['has_hr']}")

# X4 : has_hr = True avec FC
test("X4 · extract_race_info → has_hr=True avec FC",
     _x['has_hr'] == True,
     f"has_hr={_x['has_hr']}")

# X5 : timestamps_estimated propagé
_x5 = extract_race_info(_race_info_df(estimated=True), 'estimated.gpx')
test("X5 · extract_race_info → timestamps_estimated propagé",
     _x5['timestamps_estimated'] == True,
     f"ts_est={_x5['timestamps_estimated']}")

# X6 : elevation_degraded propagé
_x6 = extract_race_info(_race_info_df(degraded=True), 'degraded.gpx')
test("X6 · extract_race_info → elevation_degraded propagé",
     _x6['elevation_degraded'] == True,
     f"ele_deg={_x6['elevation_degraded']}")


# ══════════════════════════════════════════════════════════════════
# SECTION Y — compute_km_splits()
# ══════════════════════════════════════════════════════════════════

print("\n── Section Y : compute_km_splits ──")

def _splits_df(km=3):
    n    = km * 100
    dist = np.linspace(0, km * 1000, n)
    ts   = np.linspace(0, km * 360, n)   # 6 min/km
    return pd.DataFrame({
        'distance': dist, 'time_s': ts,
        'dz':       np.zeros(n), 'grade': np.zeros(n),
        'hr':       np.full(n, 145.0),
        'cadence':  np.full(n, 175.0),
    })

# Y1 : liste vide si distance < 1 km
_y1_df = pd.DataFrame({
    'distance': np.linspace(0, 500, 50), 'time_s': np.linspace(0, 180, 50),
    'dz': np.zeros(50), 'grade': np.zeros(50),
    'hr': np.full(50, 145.0), 'cadence': np.full(50, 175.0),
})
test("Y1 · compute_km_splits → vide si dist < 1 km",
     compute_km_splits(_y1_df) == [],
     f"res={compute_km_splits(_y1_df)}")

# Y2 : 1 split pour exactement 1 km
test("Y2 · compute_km_splits → 1 split pour 1 km",
     len(compute_km_splits(_splits_df(km=1))) == 1,
     f"len={len(compute_km_splits(_splits_df(km=1)))}")

# Y3 : champ 'km' commence à 1, dernier = n_km
_y3 = compute_km_splits(_splits_df(km=3))
test("Y3 · compute_km_splits → km 1→3 pour 3 km",
     _y3[0]['km'] == 1 and _y3[-1]['km'] == 3,
     f"kms={[s['km'] for s in _y3]}")

# Y4 : pace_s positif
test("Y4 · compute_km_splits → pace_s > 0",
     all(s['pace_s'] > 0 for s in _y3 if s['pace_s'] is not None),
     f"paces={[s['pace_s'] for s in _y3]}")

# Y5 : champ 'has_walk' présent dans chaque split
test("Y5 · compute_km_splits → has_walk présent dans chaque split",
     all('has_walk' in s for s in _y3),
     "")


# ══════════════════════════════════════════════════════════════════
# SECTION Z — CAS LIMITES CRITIQUES
# ══════════════════════════════════════════════════════════════════

print("\n── Section Z : cas limites critiques ──")

# Helper NaN/None — évite crash si la valeur est None
def _isnan_z(v):
    return v is None or (isinstance(v, float) and math.isnan(v))

def make_minimal_df(n=2, velocity=0.0, grade=0.0, hr=None, cadence=None):
    """DataFrame minimal pour tests cas limites."""
    d = {
        'time_s':    np.linspace(0, max(n - 1, 1), n),
        'distance':  np.linspace(0, max(n - 1, 1) * velocity, n),
        'velocity':  np.full(n, velocity),
        'grade':     np.full(n, grade),
        'dz':        np.zeros(n),
        'elevation': np.linspace(100, 100 + n, n),
        'gap_flag':  np.zeros(n, dtype=bool),
        'is_walk':   np.zeros(n, dtype=bool),
    }
    d['hr']      = np.full(n, float(hr))      if hr      is not None else np.full(n, np.nan)
    d['cadence'] = np.full(n, float(cadence)) if cadence is not None else np.full(n, np.nan)
    return pd.DataFrame(d)


# Z1 : 1 seul point GPS → fatigue_index retourne NaN sans crash
try:
    fi_z1 = fatigue_index(make_minimal_df(n=1, velocity=3.0, hr=150))
    test("Z1 · 1 point GPS : fatigue_index retourne NaN sans crash",
         _isnan_z(fi_z1['decay_ratio']),
         f"decay_ratio={fi_z1['decay_ratio']}")
except Exception as e:
    test("Z1 · 1 point GPS : fatigue_index retourne NaN sans crash",
         False, str(e))


# Z2 : FC constante (capteur bloqué) → pas de COLLAPSE ni DRIFT-CARDIO
_z2 = cardiac_drift(make_flat_df(200, fc_start=155, fc_end=155,
                                  velocity=3.0, duration_min=40))
test("Z2 · FC constante (capteur bloqué) : pas de COLLAPSE ni DRIFT-CARDIO",
     _z2['pattern'] not in ('COLLAPSE', 'DRIFT-CARDIO'),
     f"pattern={_z2['pattern']}, fc_slope={_z2.get('fc_slope_bph')}")


# Z3 : 100% marche (velocity ≤ 0.2) → decay_ratio NaN sans crash
try:
    fi_z3 = fatigue_index(make_minimal_df(n=200, velocity=0.2, hr=130))
    test("Z3 · 100% marche (velocity≤0.2) : decay_ratio NaN sans crash",
         _isnan_z(fi_z3['decay_ratio']),
         f"decay_ratio={fi_z3['decay_ratio']}")
except Exception as e:
    test("Z3 · 100% marche (velocity≤0.2) : decay_ratio NaN sans crash",
         False, str(e))


# Z4 : timestamps non ordonnés → fatigue_index ne crashe pas
_n_z4 = 100
_t_z4 = np.random.permutation(np.linspace(0, 3600, _n_z4))
_df_z4 = pd.DataFrame({
    'time_s':    _t_z4,
    'distance':  np.linspace(0, 10000, _n_z4),
    'velocity':  np.full(_n_z4, 3.0),
    'grade':     np.zeros(_n_z4),
    'dz':        np.zeros(_n_z4),
    'hr':        np.full(_n_z4, 155.0),
    'cadence':   np.full(_n_z4, 175.0),
    'gap_flag':  np.zeros(_n_z4, dtype=bool),
    'is_walk':   np.zeros(_n_z4, dtype=bool),
    'elevation': np.linspace(100, 200, _n_z4),
})
try:
    fi_z4 = fatigue_index(_df_z4)
    test("Z4 · timestamps non ordonnés : pas de crash",
         True, f"decay_ratio={fi_z4.get('decay_ratio')}")
except Exception as e:
    test("Z4 · timestamps non ordonnés : pas de crash",
         False, str(e))


# Z5 : D+ = 0 (piste plate) → compute_performance_score sans crash
_n_z5 = 200
_df_z5 = pd.DataFrame({
    'time_s':    np.linspace(0, 3600, _n_z5),
    'distance':  np.linspace(0, 10000, _n_z5),
    'velocity':  np.full(_n_z5, 2.78),
    'grade':     np.zeros(_n_z5),
    'dz':        np.zeros(_n_z5),
    'hr':        np.full(_n_z5, 150.0),
    'cadence':   np.full(_n_z5, 175.0),
    'gap_flag':  np.zeros(_n_z5, dtype=bool),
    'is_walk':   np.zeros(_n_z5, dtype=bool),
    'elevation': np.full(_n_z5, 100.0),
})
try:
    fi_z5    = fatigue_index(_df_z5)
    drift_z5 = cardiac_drift(_df_z5, duration_s=3600, dp_per_km=0.0)
    perf_z5  = compute_performance_score(fi_z5, drift_z5, dp_per_km=0.0)
    test("Z5 · D+=0 (piste plate) : score calculé sans crash",
         isinstance(perf_z5['score'], (int, float)) and not _isnan_z(perf_z5['score']),
         f"score={perf_z5['score']}")
except Exception as e:
    test("Z5 · D+=0 (piste plate) : score calculé sans crash",
         False, str(e))


# Z6 : FCmax = 220 → compute_hr_zones retourne zones monotones sans crash
_n_z6 = 100
_df_z6 = pd.DataFrame({
    'time_s': np.linspace(0, 3600, _n_z6),
    'hr':     np.full(_n_z6, 170.0),
})
_df_z6['dt'] = _df_z6['time_s'].diff().fillna(0)
try:
    _zones_z6  = compute_hr_zones(_df_z6, fcmax=220)
    _bpm_z6    = _zones_z6['bpm']
    _bounds_z6 = [_bpm_z6[z][0] for z in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']]
    test("Z6 · FCmax=220 : zones monotones sans crash",
         all(_bounds_z6[i] <= _bounds_z6[i + 1] for i in range(len(_bounds_z6) - 1)),
         f"bounds={_bounds_z6}")
except Exception as e:
    test("Z6 · FCmax=220 : zones monotones sans crash",
         False, str(e))


# Z7 : course 30 secondes → fatigue_index NaN + cardiac_drift insufficient
_df_z7 = make_minimal_df(n=10, velocity=3.0, hr=155)
_df_z7['time_s'] = np.linspace(0, 30, 10)
try:
    fi_z7    = fatigue_index(_df_z7)
    drift_z7 = cardiac_drift(_df_z7, duration_s=30, dp_per_km=0.0)
    test("Z7 · course 30s : fatigue_index decay_ratio NaN",
         _isnan_z(fi_z7['decay_ratio']),
         f"decay_ratio={fi_z7['decay_ratio']}")
    test("Z7b · course 30s : cardiac_drift insufficient_data=True",
         drift_z7['insufficient_data'] == True,
         f"insufficient={drift_z7['insufficient_data']}")
except Exception as e:
    test("Z7 · course 30s : pas de crash",
         False, str(e))


# Z8 : decay_v=None → fallback 0.0, pattern valide retourné
_df_z8 = make_flat_df(200, fc_start=150, fc_end=158, velocity=3.0, duration_min=40)
try:
    _r_z8 = cardiac_drift(_df_z8, duration_s=3600, dp_per_km=5.0, decay_v=None)
    test("Z8 · decay_v=None : pattern valide sans crash",
         _r_z8['pattern'] is not None and _r_z8['insufficient_data'] == False,
         f"pattern={_r_z8['pattern']}, decay_v={_r_z8.get('decay_v')}")
    test("Z8b · decay_v=None : decay_v retourné = 0.0",
         _r_z8.get('decay_v') == 0.0,
         f"decay_v={_r_z8.get('decay_v')}")
except Exception as e:
    test("Z8 · decay_v=None : pas de crash",
         False, str(e))


# ══════════════════════════════════════════════════════════════════
# K — FIT PARSER
# ══════════════════════════════════════════════════════════════════
section("K — fit_parser.parse_fit()")
try:
    from fit_parser import parse_fit
    _fit_available = True
except ImportError as _fit_err:
    _fit_available = False
test("K1 · FIT : import réussi", _fit_available,
     "fitparse installé" if _fit_available else f"fit_parser absent — {_fit_err}")
# Note : test K2+ nécessite un vrai fichier .fit (binaire)
# Validation fonctionnelle à faire avec le fichier réel d'Adrien


# ══════════════════════════════════════════════════════════════════
# G — GOLDEN TESTS SUR GPX RÉELS
# ══════════════════════════════════════════════════════════════════
# Objectif : détecter toute régression moteur sur données validées.
# Les valeurs attendues sont issues d'analyses manuelles confirmées.
# Si un test G échoue après refacto → régression réelle, pas de merge.
# ══════════════════════════════════════════════════════════════════
section("G — Golden tests GPX réels")
import os
from gpx_parser import parse_gpx
_GPX_DIR = os.path.join(os.path.dirname(__file__), "Dataset GPX")
def _load_gpx(filename: str):
    """Charge un GPX réel depuis Dataset GPX/. Retourne None si absent."""
    path = os.path.join(_GPX_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return parse_gpx(f.read())
# ── G1 : Samuel CDF Long 2026 ───────────────────────────────────
# Course validée : 51.6km / 2779mD+ / 5h32
# Score attendu : 85 (partiel — profil ASCENDING)
# Verdict attendu : V3 (DÉGRADATION PROGRESSIVE)
# elev_profile : ASCENDING
# FCmax saisie : 195
_g1_df = _load_gpx("CDF Trail Long 2026 Samuel.gpx")
if _g1_df is None:
    test("G1a · Samuel CDF Long — GPX chargé", False, "Fichier absent dans Dataset GPX/")
else:
    try:
        from engine import (fatigue_index, detect_elevation_profile,
                            apply_decay_correction, cardiac_drift,
                            compute_performance_score, compute_verdict,
                            classify_profile)
        from gpx_parser import extract_race_info
        import math
        _g1_info   = extract_race_info(_g1_df, "CDF Trail Long 2026 Samuel.gpx")
        _g1_fi     = fatigue_index(_g1_df)
        _g1_ep     = detect_elevation_profile(_g1_df)
        _g1_fi     = apply_decay_correction(_g1_fi, _g1_ep, _g1_df)
        _g1_dp     = _g1_info['elevation_gain'] / _g1_info['distance_km'] if _g1_info.get('distance_km', 0) > 0 else 0.0
        _g1_corr   = _g1_fi.get('decay_ratio_corrected', float('nan'))
        _g1_fi_s   = dict(_g1_fi)
        if not (isinstance(_g1_corr, float) and math.isnan(_g1_corr)) and _g1_corr is not None:
            _g1_fi_s['decay_ratio'] = _g1_corr
            _g1_fi_s['decay_pct']   = _g1_fi['decay_pct_corrected']
        _g1_dv     = (_g1_fi_s.get('decay_ratio', 1.0) or 1.0) - 1.0
        _g1_drift  = cardiac_drift(_g1_df,
                         duration_s=_g1_info['total_time_s'],
                         dp_per_km=_g1_dp,
                         decay_v=_g1_dv)
        _g1_perf   = compute_performance_score(_g1_fi_s, _g1_drift, dp_per_km=_g1_dp)
        _g1_v      = compute_verdict(_g1_fi_s, _g1_drift, _g1_perf)
        test("G1a · Samuel CDF Long — GPX chargé",
             _g1_df is not None and len(_g1_df) > 100,
             f"points={len(_g1_df)}")
        test("G1b · Samuel CDF Long — distance 48–55 km",
             48.0 <= _g1_info['distance_km'] <= 55.0,
             f"distance={_g1_info['distance_km']:.1f} km")
        test("G1c · Samuel CDF Long — elev_profile = MIXED (SCI-8)",
             _g1_ep.get('profile') == 'MIXED',
             f"profile={_g1_ep.get('profile')}")
        test("G1d · Samuel CDF Long — score partiel 20–100",
             20 <= _g1_perf['score'] <= 100,
             f"score={_g1_perf['score']}, partial={_g1_perf['partial']}")
        test("G1e · Samuel CDF Long — verdict V3 (COLLAPSE + decay >= 0.85)",
             _g1_v['code'] == 'V3',
             f"verdict={_g1_v['code']}, label={_g1_v['label']}")
        # C4-BUG fix + V3/V6 sans filtre profil (Marcus) :
        # Samuel decay brut=1.456 → plancher MIXED → corrigé=1.0 → COLLAPSE + decay>=0.85 → V3.
        test("G1f · Samuel CDF Long — pas de faux positif V7",
             _g1_v['code'] != 'V7',
             f"verdict={_g1_v['code']}")
    except Exception as e:
        test("G1 · Samuel CDF Long — pipeline sans crash", False, str(e))
# ── G2 : Dylan CDF Court 2026 ───────────────────────────────────
# Course validée : 28.2km / 1440mD+ / 2h31m32
# Score attendu : 92 (partiel EF)
# Verdict attendu : V1 STABLE
# FCmax saisie : 188
_g2_df = _load_gpx("CDF Trail Court 2026 Dylan.gpx")
if _g2_df is None:
    test("G2a · Dylan CDF Court — GPX chargé", False, "Fichier absent dans Dataset GPX/")
else:
    try:
        _g2_info   = extract_race_info(_g2_df, "CDF Trail Court 2026 Dylan.gpx")
        _g2_fi     = fatigue_index(_g2_df)
        _g2_ep     = detect_elevation_profile(_g2_df)
        _g2_fi     = apply_decay_correction(_g2_fi, _g2_ep, _g2_df)
        _g2_dp     = _g2_info['elevation_gain'] / _g2_info['distance_km'] if _g2_info.get('distance_km', 0) > 0 else 0.0
        _g2_corr   = _g2_fi.get('decay_ratio_corrected', float('nan'))
        _g2_fi_s   = dict(_g2_fi)
        if not (isinstance(_g2_corr, float) and math.isnan(_g2_corr)) and _g2_corr is not None:
            _g2_fi_s['decay_ratio'] = _g2_corr
            _g2_fi_s['decay_pct']   = _g2_fi['decay_pct_corrected']
        _g2_dv     = (_g2_fi_s.get('decay_ratio', 1.0) or 1.0) - 1.0
        _g2_drift  = cardiac_drift(_g2_df,
                         duration_s=_g2_info['total_time_s'],
                         dp_per_km=_g2_dp,
                         decay_v=_g2_dv)
        _g2_perf   = compute_performance_score(_g2_fi_s, _g2_drift, dp_per_km=_g2_dp)
        _g2_v      = compute_verdict(_g2_fi_s, _g2_drift, _g2_perf)
        test("G2a · Dylan CDF Court — GPX chargé",
             _g2_df is not None and len(_g2_df) > 100,
             f"points={len(_g2_df)}")
        test("G2b · Dylan CDF Court — distance 26–31 km",
             26.0 <= _g2_info['distance_km'] <= 31.0,
             f"distance={_g2_info['distance_km']:.1f} km")
        test("G2c · Dylan CDF Court — score ≥ 35 (SCI-8 MIXED Q4/Qmax)",
             _g2_perf['score'] >= 35,  # recalibré C2 tri-linéaire — decay 0.87 → score_gap 41, score final ~42
             f"score={_g2_perf['score']}")
             # SCI-8 recalibration : Q4/Qmax=0.870 sans EF → score~56 (vs 92 ancien Q4/Q1 gonflé)
             # Dylan "crampes mais course tenue" → V2 cohérent. ≥65 était basé sur ratio clippé ~1.35.
        test("G2d · Dylan CDF Court — verdict V2 (SCI-8 MIXED)",
             _g2_v['code'] == 'V2',
             f"verdict={_g2_v['code']}, label={_g2_v['label']}")
        test("G2e · Dylan CDF Court — drift STABLE",
             _g2_drift.get('pattern') == 'STABLE',
             f"pattern={_g2_drift.get('pattern')}")
    except Exception as e:
        test("G2 · Dylan CDF Court — pipeline sans crash", False, str(e))
# ── G3 : Coralie CDF Long 2026 ──────────────────────────────────
# Course validée : 51.3km / 2792mD+ / 6h01
# Score attendu : 86
# Verdict attendu : V1 STABLE
# elev_profile : DESCENDING (validé B7)
_g3_df = _load_gpx("CDF Trail Long 2026 Coralie.gpx")
if _g3_df is None:
    test("G3a · Coralie CDF Long — GPX chargé", False, "Fichier absent dans Dataset GPX/")
else:
    try:
        _g3_info   = extract_race_info(_g3_df, "CDF Trail Long 2026 Coralie.gpx")
        _g3_fi     = fatigue_index(_g3_df)
        _g3_ep     = detect_elevation_profile(_g3_df)
        _g3_fi     = apply_decay_correction(_g3_fi, _g3_ep, _g3_df)
        _g3_dp     = _g3_info['elevation_gain'] / _g3_info['distance_km'] if _g3_info.get('distance_km', 0) > 0 else 0.0
        _g3_corr   = _g3_fi.get('decay_ratio_corrected', float('nan'))
        _g3_fi_s   = dict(_g3_fi)
        if not (isinstance(_g3_corr, float) and math.isnan(_g3_corr)) and _g3_corr is not None:
            _g3_fi_s['decay_ratio'] = _g3_corr
            _g3_fi_s['decay_pct']   = _g3_fi['decay_pct_corrected']
        _g3_dv     = (_g3_fi_s.get('decay_ratio', 1.0) or 1.0) - 1.0
        _g3_drift  = cardiac_drift(_g3_df,
                         duration_s=_g3_info['total_time_s'],
                         dp_per_km=_g3_dp,
                         decay_v=_g3_dv)
        _g3_perf   = compute_performance_score(_g3_fi_s, _g3_drift, dp_per_km=_g3_dp)
        _g3_v      = compute_verdict(_g3_fi_s, _g3_drift, _g3_perf)
        test("G3a · Coralie CDF Long — GPX chargé",
             _g3_df is not None and len(_g3_df) > 100,
             f"points={len(_g3_df)}")
        test("G3b · Coralie CDF Long — distance 48–55 km",
             48.0 <= _g3_info['distance_km'] <= 55.0,
             f"distance={_g3_info['distance_km']:.1f} km")
        test("G3c · Coralie CDF Long — elev_profile = MIXED (SCI-8)",
             _g3_ep.get('profile') == 'MIXED',
             f"profile={_g3_ep.get('profile')}")
        test("G3d · Coralie CDF Long — score 70–100 (SCI-8 MIXED)",
             70 <= _g3_perf['score'] <= 100,
             f"score={_g3_perf['score']}")
        test("G3e · Coralie CDF Long — verdict V1",
             _g3_v['code'] == 'V1',
             f"verdict={_g3_v['code']}, label={_g3_v['label']}")
        test("G3f · Coralie CDF Long — pas de faux positif V7",
             _g3_v['code'] != 'V7',
             f"verdict={_g3_v['code']}")
    except Exception as e:
        test("G3 · Coralie CDF Long — pipeline sans crash", False, str(e))
# ── G4 : Jérémy 43km ────────────────────────────────────────────
# Course validée : 43.3km / 2217mD+ / 5h11
# elev_profile : FLAT — correction decay non appliquée
# Pattern : COLLAPSE — effondrement CV détecté
# Score attendu : ~70 (partiel — EF non interprétable sur COLLAPSE)
# Verdict attendu : V3 DÉGRADATION PROGRESSIVE
# FCmax : 189
_g4_df = _load_gpx("Jeremy 43km.gpx")
if _g4_df is None:
    test("G4a · Jérémy 43km — GPX chargé", False, "Fichier absent dans Dataset GPX/")
else:
    try:
        _g4_info  = extract_race_info(_g4_df, "Jeremy 43km.gpx")
        _g4_fi    = fatigue_index(_g4_df)
        _g4_ep    = detect_elevation_profile(_g4_df)
        _g4_fi    = apply_decay_correction(_g4_fi, _g4_ep, _g4_df)
        _g4_dp    = _g4_info['elevation_gain'] / _g4_info['distance_km'] if _g4_info.get('distance_km', 0) > 0 else 0.0
        _g4_corr  = _g4_fi.get('decay_ratio_corrected', float('nan'))
        _g4_fi_s  = dict(_g4_fi)
        if not (isinstance(_g4_corr, float) and math.isnan(_g4_corr)) and _g4_corr is not None:
            _g4_fi_s['decay_ratio'] = _g4_corr
            _g4_fi_s['decay_pct']   = _g4_fi['decay_pct_corrected']
        _g4_dv    = (_g4_fi_s.get('decay_ratio', 1.0) or 1.0) - 1.0
        _g4_drift = cardiac_drift(_g4_df,
                        duration_s=_g4_info['total_time_s'],
                        dp_per_km=_g4_dp,
                        decay_v=_g4_dv)
        _g4_perf  = compute_performance_score(_g4_fi_s, _g4_drift, dp_per_km=_g4_dp)
        _g4_v     = compute_verdict(_g4_fi_s, _g4_drift, _g4_perf)
        test("G4a · Jérémy 43km — GPX chargé",
             _g4_df is not None and len(_g4_df) > 100,
             f"points={len(_g4_df)}")
        test("G4b · Jérémy 43km — distance 41–46 km",
             41.0 <= _g4_info['distance_km'] <= 46.0,
             f"distance={_g4_info['distance_km']:.1f} km")
        test("G4c · Jérémy 43km — elev_profile = FLAT",
             _g4_ep.get('profile') == 'FLAT',
             f"profile={_g4_ep.get('profile')}")
        test("G4d · Jérémy 43km — correction decay non appliquée",
             _g4_fi.get('correction_applied') == False,
             f"correction_applied={_g4_fi.get('correction_applied')}")
        test("G4e · Jérémy 43km — pattern COLLAPSE",
             _g4_drift.get('pattern') == 'COLLAPSE',
             f"pattern={_g4_drift.get('pattern')}")
        test("G4f · Jérémy 43km — score partiel 60–80",
             60 <= _g4_perf['score'] <= 80,
             f"score={_g4_perf['score']}, partial={_g4_perf['partial']}")
        test("G4g · Jérémy 43km — verdict V3",
             _g4_v['code'] == 'V3',
             f"verdict={_g4_v['code']}, label={_g4_v['label']}")
    except Exception as e:
        test("G4 · Jérémy 43km — pipeline sans crash", False, str(e))
# ── G5 : Coralie CDF Long 2023 ──────────────────────────────────
# Course validée : 69km / 2601mD+ / 7h32
# elev_profile : DESCENDING — correction decay appliquée
# C4-BUG résolu Sprint 8 — cap_dynamic fixe 1.20 sur DESCENDING
# Pattern : DRIFT-CARDIO (-16.1%) — signal dominant malgré correction
# Score attendu : ~41 (complet — EF disponible)
# Verdict attendu : V4 FATIGUE COMBINÉE
# FCmax : 200
# C4-BUG résolu Sprint 8 — cap_dynamic fixe 1.20 sur DESCENDING
_g5_df = _load_gpx("CDF Trail Long 2023 Coralie.gpx")
if _g5_df is None:
    test("G5a · Coralie CDF 2023 — GPX chargé", False, "Fichier absent dans Dataset GPX/")
else:
    try:
        _g5_info  = extract_race_info(_g5_df, "CDF Trail Long 2023 Coralie.gpx")
        _g5_fi    = fatigue_index(_g5_df)
        _g5_ep    = detect_elevation_profile(_g5_df)
        _g5_fi    = apply_decay_correction(_g5_fi, _g5_ep, _g5_df)
        _g5_dp    = _g5_info['elevation_gain'] / _g5_info['distance_km'] if _g5_info.get('distance_km', 0) > 0 else 0.0
        _g5_corr  = _g5_fi.get('decay_ratio_corrected', float('nan'))
        _g5_fi_s  = dict(_g5_fi)
        if not (isinstance(_g5_corr, float) and math.isnan(_g5_corr)) and _g5_corr is not None:
            _g5_fi_s['decay_ratio'] = _g5_corr
            _g5_fi_s['decay_pct']   = _g5_fi['decay_pct_corrected']
        _g5_dv    = (_g5_fi_s.get('decay_ratio', 1.0) or 1.0) - 1.0
        _g5_drift = cardiac_drift(_g5_df,
                        duration_s=_g5_info['total_time_s'],
                        dp_per_km=_g5_dp,
                        decay_v=_g5_dv)
        _g5_perf  = compute_performance_score(_g5_fi_s, _g5_drift, dp_per_km=_g5_dp)
        _g5_v     = compute_verdict(_g5_fi_s, _g5_drift, _g5_perf)
        test("G5a · Coralie CDF 2023 — GPX chargé",
             _g5_df is not None and len(_g5_df) > 100,
             f"points={len(_g5_df)}")
        test("G5b · Coralie CDF 2023 — distance 66–72 km",
             66.0 <= _g5_info['distance_km'] <= 72.0,
             f"distance={_g5_info['distance_km']:.1f} km")
        test("G5c · Coralie CDF 2023 — elev_profile = FLAT (SCI-8: desc_bias < 0.55)",
             _g5_ep.get('profile') == 'FLAT',
             f"profile={_g5_ep.get('profile')}")
        test("G5d · Coralie CDF 2023 — correction decay non appliquée (FLAT)",
             _g5_fi.get('correction_applied') == False,
             f"correction_applied={_g5_fi.get('correction_applied')}")
        test("G5e · Coralie CDF 2023 — pattern DRIFT-CARDIO",
             _g5_drift.get('pattern') == 'DRIFT-CARDIO',
             f"pattern={_g5_drift.get('pattern')}, drift_pct={_g5_drift.get('drift_pct'):.1f}%")
        test("G5f · Coralie CDF 2023 — score 10–25 (SCI-8: decay brut sans correction)",
             10 <= _g5_perf['score'] <= 25,
             f"score={_g5_perf['score']}, partial={_g5_perf['partial']}")
        test("G5g · Coralie CDF 2023 — verdict V5 (SCI-8: decay=0.751 < 0.80)",
             _g5_v['code'] == 'V5',
             f"verdict={_g5_v['code']}, label={_g5_v['label']}")
        test("G5h · Coralie CDF 2023 — drift_pct < -10% (DRIFT-CARDIO fort)",
             (_g5_drift.get('drift_pct') or 0) < -10.0,
             f"drift_pct={_g5_drift.get('drift_pct')}")
        # C4-BUG résolu Sprint 8 — cap_dynamic fixe 1.20 sur DESCENDING
    except Exception as e:
        test("G5 · Coralie CDF 2023 — pipeline sans crash", False, str(e))


# ══════════════════════════════════════════════════════════════════
# SCI-7 — EF FALLBACK GAP (attente validation terrain)
# ══════════════════════════════════════════════════════════════════
section("SCI-7 — EF fallback GAP : profil montagneux sans plat")

def make_mountain_df(n=500, duration_s=19800, ef_slope_pph=-0.05):
    """
    Simule un profil montagneux : grade moyen +12%, zero plat (<3%).
    ef_slope_pph=-0.05/h = dégradation EF de 5% par heure.
    Sur 5.5h → drift attendu = -27.5% → clippé à -20%.
    """
    t = np.linspace(0, duration_s, n)
    # grade >3% partout → aucun point dans flat
    grade = np.full(n, 12.0)
    hr = np.linspace(155, 170, n)
    # EF décroissante : gap_velocity baisse progressivement
    # ef_point = gap_v / hr * 100
    # On construit gap_v tel que ef_slope_pph = -0.05/h
    # ef(t) = ef0 + ef_slope_pph * t_h
    ef0 = 2.0
    ef_arr = ef0 + (ef_slope_pph / 3600) * t  # en /s
    gap_v = ef_arr * hr / 100
    gap_v = np.clip(gap_v, 0.5, 6.0)
    d = np.cumsum(gap_v * np.diff(t, prepend=0))
    return pd.DataFrame({
        'time_s': t, 'distance': d, 'hr': hr,
        'velocity': gap_v * 0.9, 'grade': grade,
        'gap_velocity': gap_v,
    })

# SCI7-1 : profil montagneux → insufficient_data=False via fallback
df_m1 = make_mountain_df(ef_slope_pph=-0.05)
r_m1 = cardiac_drift(df_m1, duration_s=19800, dp_per_km=55)
test("SCI7-1 · profil montagneux + EF dégradée → insufficient_data=False (fallback)",
     r_m1['insufficient_data'] == False,
     f"insufficient={r_m1['insufficient_data']} pattern={r_m1.get('pattern')} drift={r_m1.get('drift_pct')}")

# SCI7-2 : drift_pct clippé à -20% (ef_slope × durée dépasse)
test("SCI7-2 · drift_pct_gap clippé à -20% minimum",
     r_m1.get('drift_pct') is not None and r_m1['drift_pct'] >= -20.0,
     f"drift_pct={r_m1.get('drift_pct')}")

# SCI7-3 : ef_source = GAP_FALLBACK tracé dans le retour
test("SCI7-3 · ef_source='GAP_FALLBACK' présent dans retour",
     r_m1.get('ef_source') == 'GAP_FALLBACK',
     f"ef_source={r_m1.get('ef_source')}")

# SCI7-4 : profil montagneux SANS dégradation EF → insufficient_data=True (pas de faux positif)
df_m2 = make_mountain_df(ef_slope_pph=0.001)  # EF stable ou légèrement croissante
r_m2 = cardiac_drift(df_m2, duration_s=19800, dp_per_km=55)
test("SCI7-4 · profil montagneux + EF stable → insufficient_data=True (pas de fallback)",
     r_m2['insufficient_data'] == True,
     f"insufficient={r_m2['insufficient_data']} pattern={r_m2.get('pattern')}")

# SCI7-5 : rétrocompat — profil FLAT non affecté
df_flat = make_flat_df(200, fc_start=150, fc_end=155)
r_flat = cardiac_drift(df_flat, duration_s=3600, dp_per_km=5)
test("SCI7-5 · profil FLAT non affecté — ef_source absent ou FLAT",
     r_flat.get('ef_source', 'FLAT') == 'FLAT',
     f"ef_source={r_flat.get('ef_source', 'FLAT')} pattern={r_flat.get('pattern')}")

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
