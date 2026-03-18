"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — engine.py                                       ║
║         GAP · Fatigue · FC · Cadence · Recommandations · v3.4   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════
# GAP ENGINE — Minetti 2002
# ══════════════════════════════════════════════════════════════════

def gap_correction(velocity_ms: float, grade_pct: float) -> float:
    """Version scalaire — conservée pour les appels ponctuels (splits, charts)."""
    g = grade_pct / 100.0
    energy_flat  = 3.6
    energy_slope = (155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6)
    correction   = max(0.5, min(2.5, energy_slope / energy_flat))
    return velocity_ms / correction if correction > 0 else velocity_ms


def gap_correction_vec(velocity: np.ndarray, grade_pct: np.ndarray) -> np.ndarray:
    """Version vectorisée numpy — ×50 plus rapide sur DataFrame complet."""
    g = grade_pct / 100.0
    energy_slope = (155.4*g**5 - 30.4*g**4 - 43.3*g**3 + 46.3*g**2 + 19.5*g + 3.6)
    correction = np.clip(energy_slope / 3.6, 0.5, 2.5)
    return np.where(correction > 0, velocity / correction, velocity)


def v_to_pace(v: float) -> str:
    if not v or v <= 0.1:
        return "--:--"
    s = 1000 / v
    return f"{int(s//60)}:{int(s%60):02d}"


# ══════════════════════════════════════════════════════════════════
# PROFIL PENTE
# ══════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════
# FATIGUE INDEX
# ══════════════════════════════════════════════════════════════════

def fatigue_index(df: pd.DataFrame) -> dict:
    df = df.copy()
    # v3.3 : vectorisé numpy
    df['gap'] = gap_correction_vec(df['velocity'].to_numpy(), df['grade'].to_numpy())
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
    return {
        'quartiles': quartiles,
        'decay_ratio': ratio,
        'decay_pct': (1 - ratio)*100 if not math.isnan(ratio) else float('nan'),
    }


def flat_pace_estimate(df: pd.DataFrame) -> float:
    flat_mask = (df['grade'].abs() < 3) & (df['velocity'] > 0.3)
    fdf = df[flat_mask]
    if len(fdf) < 10:
        return df[df['velocity'] > 0.3]['velocity'].median()
    # v3.3 : vectorisé numpy
    return float(np.median(gap_correction_vec(
        fdf['velocity'].to_numpy(), fdf['grade'].to_numpy()
    )))


def classify_profile(decay_ratio: float, flat_v: float) -> str:
    if math.isnan(decay_ratio):
        return "PROFIL INCONNU"
    if decay_ratio >= 0.93:
        return "PROFIL ENDURANCE"
    if decay_ratio >= 0.85:
        return "PROFIL EXPLOSIF"
    return "PROFIL FRAGILE"


# ══════════════════════════════════════════════════════════════════
# ZONES FC
# ══════════════════════════════════════════════════════════════════

ZONE_NAMES = {
    'Z1': 'Récupération',
    'Z2': 'Endurance fondamentale',
    'Z3': 'Tempo / Seuil aérobie',
    'Z4': 'Seuil lactate',
    'Z5': 'VO2max / Anaérobie',
}


def compute_hr_zones(df: pd.DataFrame, fcmax: int, custom_zones: dict = None) -> dict:
    valid = df[df['hr'] > 50].copy()
    valid['dt'] = valid['time_s'].diff().fillna(0).clip(0, 30)

    if custom_zones:
        zone_bpm = {z: (int(v[0]), int(v[1])) for z, v in custom_zones.items()}
        bins   = [zone_bpm[z][0] for z in ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']] + [zone_bpm['Z5'][1] + 1]
        labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
        valid['zone'] = pd.cut(valid['hr'], bins=bins, labels=labels, right=False)
        zone_time = valid.groupby('zone', observed=True)['dt'].sum().reindex(labels, fill_value=0).to_dict()
        total = sum(zone_time.values())
        zone_pct = {z: (t/total*100 if total > 0 else 0) for z, t in zone_time.items()}
        return {'time': zone_time, 'pct': zone_pct, 'bpm': zone_bpm, 'fcmax': fcmax, 'mode': 'manual'}

    thresholds = {
        'Z1': (0,    0.60),
        'Z2': (0.60, 0.70),
        'Z3': (0.70, 0.80),
        'Z4': (0.80, 0.90),
        'Z5': (0.90, 1.01),
    }
    zone_bpm  = {z: (int(lo*fcmax), int(hi*fcmax)) for z, (lo, hi) in thresholds.items()}
    bins   = [lo * fcmax for lo, _ in thresholds.values()] + [250]
    labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z5']
    valid['zone'] = pd.cut(valid['hr'], bins=bins, labels=labels, right=False)
    zone_time = valid.groupby('zone', observed=True)['dt'].sum().reindex(labels, fill_value=0).to_dict()
    total     = sum(zone_time.values())
    zone_pct  = {z: (t/total*100 if total > 0 else 0) for z, t in zone_time.items()}
    return {'time': zone_time, 'pct': zone_pct, 'bpm': zone_bpm, 'fcmax': fcmax, 'mode': 'auto'}


# ══════════════════════════════════════════════════════════════════
# CARDIAC DRIFT — v3.4 (Sprint 2 ⑤ + ⑥)
# ══════════════════════════════════════════════════════════════════

def cardiac_drift(df: pd.DataFrame) -> dict:
    """
    v3.4 — Détection 3 patterns : STABLE / DRIFT / COLLAPSE
    COLLAPSE = FC qui s'effondre en fin de course (cardiac drift inversé)
    Sprint 2 item ⑤ : seuil minimum 10 min terrain plat
    Sprint 2 item ⑥ : CDC Elena v1.0 — régression linéaire FC + fc_delta Q1/Q4
    """
    _empty = {
        'ef1': None, 'ef2': None, 'drift_pct': None, 'quartiles': {},
        'pattern': None, 'collapse_pct': None, 'fc_slope_bph': None,
        'fc_q1_mean': None, 'fc_q4_mean': None, 'insufficient_data': True,
    }

    flat = df[
        (df['grade'].abs() < 3) &
        (df['velocity'] > 0.3) &
        (df['hr'] > 80)
    ].copy()

    if len(flat) < 20:
        return _empty

    flat = flat.sort_values('distance').reset_index(drop=True)

    # Item ⑤ : seuil minimum 10 min de terrain plat
    flat_duration_min = (flat['time_s'].max() - flat['time_s'].min()) / 60
    if flat_duration_min < 10:
        return _empty

    # EF par demi-course (rétrocompat)
    mid = len(flat) // 2

    def ef(sub):
        if len(sub) == 0:
            return None
        v  = sub['velocity'].mean()
        hr = sub['hr'].mean()
        return (v / hr) * 100 if hr > 0 else None

    ef1 = ef(flat.iloc[:mid])
    ef2 = ef(flat.iloc[mid:])
    drift_ef = ((ef2 - ef1) / ef1 * 100) if (ef1 and ef2 and ef1 > 0) else None

    # EF par quartile de distance totale
    total_dist = df['distance'].iloc[-1]
    q_size = total_dist / 4
    ef_q = {}
    for i in range(1, 5):
        q = flat[
            (flat['distance'] >= (i-1)*q_size) &
            (flat['distance'] < i*q_size)
        ]
        ef_q[f'Q{i}'] = ef(q)

    # FC moyenne par quartile sur plat
    def fc_mean(sub):
        v = sub['hr'].dropna()
        return float(v.mean()) if len(v) > 3 else None

    fc_q = {}
    for i in range(1, 5):
        q = flat[
            (flat['distance'] >= (i-1)*q_size) &
            (flat['distance'] < i*q_size)
        ]
        fc_q[f'Q{i}'] = fc_mean(q)

    fc_q1_mean = fc_q.get('Q1')
    fc_q4_mean = fc_q.get('Q4')

    # Régression linéaire FC/temps → pente bpm/heure
    t_arr  = flat['time_s'].to_numpy()
    hr_arr = flat['hr'].to_numpy()
    if len(t_arr) > 10:
        coeffs = np.polyfit(t_arr, hr_arr, 1)
        fc_slope_bph = float(coeffs[0]) * 3600
    else:
        fc_slope_bph = 0.0

    # fc_delta_pct Q1 → Q4
    if fc_q1_mean and fc_q4_mean and fc_q1_mean > 0:
        fc_delta_pct = (fc_q4_mean - fc_q1_mean) / fc_q1_mean * 100
    else:
        fc_delta_pct = None

    # Classification pattern
    # COLLAPSE : FC chute > 10% ET slope < -3 bpm/h
    # DRIFT    : EF dégrade < -2% sans COLLAPSE
    # STABLE   : tout le reste
    if fc_delta_pct is not None and fc_delta_pct < -10 and fc_slope_bph < -3.0:
        pattern      = 'COLLAPSE'
        collapse_pct = fc_delta_pct
        drift_ef     = None  # EF non interprétable en cas de COLLAPSE
    elif drift_ef is not None and drift_ef < -2:
        pattern      = 'DRIFT'
        collapse_pct = None
    else:
        pattern      = 'STABLE'
        collapse_pct = None

    return {
        'ef1':              ef1,
        'ef2':              ef2,
        'drift_pct':        drift_ef,
        'quartiles':        ef_q,
        'pattern':          pattern,
        'collapse_pct':     collapse_pct,
        'fc_slope_bph':     fc_slope_bph,
        'fc_q1_mean':       fc_q1_mean,
        'fc_q4_mean':       fc_q4_mean,
        'insufficient_data': False,
    }


# ══════════════════════════════════════════════════════════════════
# SPLITS PAR KM
# ══════════════════════════════════════════════════════════════════

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

        pace  = dt if dt > 0 else None
        v     = dd / dt if dt > 0 else None
        gap_v = gap_correction(v, seg['grade'].mean()) if v else None

        hr_mean  = seg.loc[seg['hr'] > 50, 'hr'].mean() if seg['hr'].notna().any() else None
        # v3.1 : seuil filtre cadence relevé à 80
        cad_mean = seg.loc[seg['cadence'] > 80, 'cadence'].mean() if seg['cadence'].notna().any() else None

        splits.append({
            'km':       km + 1,
            'pace_s':   pace,
            'pace':     v_to_pace(v) if v else '--:--',
            'gap':      v_to_pace(gap_v) if gap_v else '--:--',
            'd_pos':    int(dz_pos),
            'd_neg':    int(dz_neg),
            'hr':       round(hr_mean) if hr_mean else None,
            'cadence':  round(cad_mean) if cad_mean else None,
            'velocity': v,
        })
    return splits


# ══════════════════════════════════════════════════════════════════
# FC PAR PENTE
# ══════════════════════════════════════════════════════════════════

def hr_by_grade(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[(df['hr'] > 80) & (df['velocity'] > 0.3)].copy()
    bins   = list(range(-20, 25, 5))
    labels = [f"{b}%" for b in bins[:-1]]
    valid['grade_bin'] = pd.cut(valid['grade'], bins=bins, labels=labels, right=False)
    result = valid.groupby('grade_bin', observed=True).agg(
        hr_mean=('hr', 'mean'),
        n=('hr', 'count'),
        pace_mean=('velocity', 'mean'),
    ).reset_index()
    result = result[result['n'] > 30]
    return result


# ══════════════════════════════════════════════════════════════════
# CADENCE
# ══════════════════════════════════════════════════════════════════

def cadence_analysis(df: pd.DataFrame) -> dict:
    # v3.1 : seuil filtre relevé à 80 (post-multiplication ×2)
    valid = df[df['cadence'] > 80]['cadence']
    if len(valid) < 10:
        return {'mean': None, 'max': None, 'dist': {}, 'optimal_pct': None}

    # v3.1 : bins recalculés pour valeurs post-×2 (150–210 spm réaliste trail)
    bins = {'<150': 0, '150-160': 0, '160-170': 0, '170-180': 0,
            '180-190': 0, '190-200': 0, '>200': 0}
    for c in valid:
        if c < 150:    bins['<150'] += 1
        elif c < 160:  bins['150-160'] += 1
        elif c < 170:  bins['160-170'] += 1
        elif c < 180:  bins['170-180'] += 1
        elif c < 190:  bins['180-190'] += 1
        elif c <= 200: bins['190-200'] += 1
        else:          bins['>200'] += 1

    total = len(valid)
    pct = {k: v/total*100 for k, v in bins.items()}
    # Zone optimale trail : 170-190 spm (= 85-95 spm unilatéral Garmin)
    optimal_pct = pct.get('170-180', 0) + pct.get('180-190', 0) + pct.get('190-200', 0)

    return {
        'mean': valid.mean(),
        'max':  valid.max(),
        'dist': pct,
        'optimal_pct': optimal_pct,
    }


# ══════════════════════════════════════════════════════════════════
# RECOMMANDATIONS COACH — v3.4
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

    # Extraction pattern cardiac_drift v3.4
    drift_pattern = drift.get('pattern')
    drift_pct     = drift.get('drift_pct')
    collapse_pct  = drift.get('collapse_pct')
    insufficient  = drift.get('insufficient_data', False)

    cad_mean    = cad_analysis.get('mean')
    optimal_pct = cad_analysis.get('optimal_pct', 0)
    hr_mean     = info.get('hr_mean')

    # ── Intensité globale ────────────────────────────────────────
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

    # ── Découplage cardiaque — 3 patterns ───────────────────────
    if insufficient:
        pass  # pas de recommandation FC sans données plat suffisantes

    elif drift_pattern == 'COLLAPSE':
        fc_q1 = drift.get('fc_q1_mean') or 0
        fc_q4 = drift.get('fc_q4_mean') or 0
        recs.append({
            'level': 'crit',
            'title': 'Effondrement cardiovasculaire détecté',
            'body': (
                f"La FC s'est effondrée de {abs(collapse_pct):.1f}% sur terrain plat "
                f"(FC Q1 : {fc_q1:.0f} → FC Q4 : {fc_q4:.0f} bpm). "
                "Ce pattern 'cardiac collapse' invalide l'EF en fin de course — "
                "l'efficacité apparente est trompeuse. "
                "Causes probables : hypoglycémie sévère, déshydratation, "
                "ou décrochage du système nerveux autonome. "
                "Priorité absolue : revoir la stratégie nutritionnelle (glucides/heure) "
                "et le protocole hydratation. Minimum 10 jours de récupération complète."
            )
        })

    elif drift_pattern == 'DRIFT' and drift_pct is not None:
        if drift_pct < -5:
            recs.append({
                'level': 'warn',
                'title': f'Découplage cardiaque : {drift_pct:.1f}%',
                'body': f"L'EF (Efficiency Factor) se dégrade de {abs(drift_pct):.1f}% entre la 1ère et 2ème moitié "
                        "sur terrain plat. Signal d'une fatigue musculaire ou glycémique. "
                        "Axe de travail : sorties longues >3h en Z2 strict + stratégie nutritionnelle "
                        "(1 gel ou 30-40g glucides/30min après 1h d'effort)."
            })
        else:
            recs.append({
                'level': 'warn',
                'title': f'Découplage cardiaque modéré : {drift_pct:.1f}%',
                'body': f"Légère dégradation de l'EF de {abs(drift_pct):.1f}%. "
                        "Axe de travail : sorties longues >3h en Z2 strict + stratégie nutritionnelle "
                        "(1 gel ou 30-40g glucides/30min après 1h d'effort)."
            })

    elif drift_pattern == 'STABLE' and drift_pct is not None:
        recs.append({
            'level': 'info',
            'title': f'Très bon découplage cardiaque : {drift_pct:.1f}%',
            'body': "L'efficacité de course est quasi-constante sur l'ensemble de l'effort. "
                    "Ton endurance aérobie est solide. Pour continuer à progresser : "
                    "introduis 1 séance/semaine de travail spécifique au seuil (2×20min à FC seuil)."
        })

    # ── Fatigue GAP ──────────────────────────────────────────────
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

    # ── Cadence ──────────────────────────────────────────────────
    if cad_mean:
        # v3.1 : seuils recalculés en SPM total (×2)
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

    # ── Endurance spécifique fin de course ───────────────────────
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
        'body': _strength_advice(profile, drift_pct, cad_mean, dp),
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
