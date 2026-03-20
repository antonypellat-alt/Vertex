"""
╔══════════════════════════════════════════════════════════════════╗
║         VERTEX — engine.py                                       ║
║         GAP · Fatigue · FC · Cadence · Recommandations · v3.5   ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math

import numpy as np
import pandas as pd


def _isnan(v) -> bool:
    """math.isnan() sécurisé — retourne True si v est None ou NaN."""
    if v is None:
        return True
    try:
        return math.isnan(v)
    except (TypeError, ValueError):
        return True


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
        'decay_pct': (1 - ratio)*100 if not _isnan(ratio) else float('nan'),
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


def classify_profile(decay_ratio: float) -> str:
    if _isnan(decay_ratio):
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
# CARDIAC DRIFT — v3.5 (CDC Elena v1.2)
# ══════════════════════════════════════════════════════════════════

def get_collapse_thresholds(duration_s: float, dp_per_km: float) -> tuple:
    """
    Retourne (slope_threshold, delta_threshold) selon le profil de course.
    CDC Elena v1.2 — calibration multi-format Sprint 4.

    Ultra / très montagneux (>4h ou >50 m/km D+) : -2.0 bph / -6%
    Trail moyen (>2h ou >30 m/km D+)             : -2.5 bph / -8%
    Trail court                                   : -3.0 bph / -10%

    Backlog : palier ultra-long >8h (attente GPX Emmanuel EcoTrail)
    """
    is_ultra = duration_s > 14400 or dp_per_km > 50
    is_mid   = duration_s > 7200  or dp_per_km > 30
    if is_ultra:
        return (-2.0, -6.0)
    elif is_mid:
        return (-2.5, -8.0)
    else:
        return (-3.0, -10.0)


def cardiac_drift(df: pd.DataFrame,
                  duration_s: float = None,
                  dp_per_km: float = 0.0) -> dict:
    """
    v3.5 — Détection 3 patterns : STABLE / DRIFT / COLLAPSE
    COLLAPSE A : FC chute > delta_thr OU slope < slope_thr (seuils contextuels)
    COLLAPSE B : FC chute > 20% quelle que soit la pente (segments plats uniquement)
    Sprint 2 item ⑤ : seuil minimum 10 min terrain plat
    CDC Elena v1.2 — seuils adaptatifs multi-format (Sprint 4)
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
    # KNOWN LIMITATION — BUG-3 Sprint 4A :
    # ef1/ef2 calculés sur segments PLATS (grade < 3%) uniquement.
    # Sur parcours quasi-plat (<20 m/km D+), la correction GAP Minetti → 1.0
    # et l'EF devient V/FC sans normalisation terrain, amplifiant les variations
    # de vitesse pure. La dérive drift_ef peut diverger de l'EF globale UI.
    # Limite du modèle Minetti 2002 sur gradient < 2-3%, pas un bug de calcul.
    # Paliatif : disclaimer conditionnel SCI-1. Solution terme : SCI-3 (di Prampero).
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

    # EF par quartile de temps total — F6 : aligné sur ef1/ef2 (demi-temps)
    # Cohérence : découplage cardiaque = phénomène temporel, pas spatial
    total_time = flat['time_s'].max() - flat['time_s'].min()
    t_start    = flat['time_s'].min()
    q_size_t   = total_time / 4
    ef_q = {}
    for i in range(1, 5):
        q = flat[
            (flat['time_s'] >= t_start + (i-1)*q_size_t) &
            (flat['time_s'] <  t_start + i*q_size_t)
        ]
        ef_q[f'Q{i}'] = ef(q)

    # FC moyenne par quartile temps (cohérence F6)
    def fc_mean(sub):
        v = sub['hr'].dropna()
        return float(v.mean()) if len(v) > 3 else None

    fc_q = {}
    for i in range(1, 5):
        q = flat[
            (flat['time_s'] >= t_start + (i-1)*q_size_t) &
            (flat['time_s'] <  t_start + i*q_size_t)
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

    # Classification pattern — CDC Elena v1.2 : seuils adaptatifs multi-format
    # COLLAPSE A : FC chute > delta_thr OU slope < slope_thr (contextuels)
    # COLLAPSE B : FC chute > 20% quelle que soit la pente (segments plats uniquement)
    # DRIFT      : EF dégrade < -2% sans COLLAPSE
    # STABLE     : tout le reste
    _dur_s     = duration_s if duration_s is not None else float(df['time_s'].max())
    slope_thr, delta_thr = get_collapse_thresholds(_dur_s, dp_per_km)
    collapse_a = (fc_delta_pct is not None and (
        fc_delta_pct < delta_thr or                          # chute delta suffisante seule
        (fc_slope_bph < slope_thr and fc_delta_pct < delta_thr / 2)  # slope + chute minimale
    ))
    collapse_b = (fc_delta_pct is not None and fc_delta_pct < -20)

    if collapse_a or collapse_b:
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
# DÉTECTION MARCHE ACTIVE — Sprint 2 item ④
# ══════════════════════════════════════════════════════════════════

def detect_walk_segments(df: pd.DataFrame,
                         grade_threshold: float = 15.0,
                         velocity_threshold: float = 1.5) -> pd.DataFrame:
    """
    Ajoute la colonne 'is_walk' au DataFrame.
    Critères : pente > grade_threshold% ET vitesse < velocity_threshold m/s.
    Ne s'applique que sur les sections en montée (grade positif).
    Retourne le DataFrame avec la colonne is_walk ajoutée.
    """
    df = df.copy()
    df['is_walk'] = (
        (df['grade'] > grade_threshold) &
        (df['velocity'] < velocity_threshold) &
        (df['velocity'] > 0.1)   # filtre les arrêts complets (ravitos, etc.)
    )
    return df


def walk_stats(df: pd.DataFrame, grade_threshold: float = 15.0) -> dict:
    """
    Calcule les statistiques de marche active sur sections raides.
    Retourne un dict avec walk_ratio, walk_time_min, run_time_min,
    walk_distance_m, n_walk_segments.
    Retourne None si pas de section >grade_threshold dans la course.
    """
    if 'is_walk' not in df.columns:
        df = detect_walk_segments(df, grade_threshold)

    # Sections raides uniquement (grade > seuil)
    steep = df[df['grade'] > grade_threshold].copy()
    if len(steep) < 5:
        return {
            'walk_ratio':       None,
            'walk_time_min':    None,
            'run_time_min':     None,
            'walk_distance_m':  None,
            'n_walk_segments':  0,
            'has_steep':        False,
        }

    steep_walk = steep[steep['is_walk']]
    steep_run  = steep[~steep['is_walk']]

    walk_time = steep_walk['dt'].sum() if 'dt' in steep_walk.columns else 0
    run_time  = steep_run['dt'].sum()  if 'dt' in steep_run.columns  else 0
    total_steep_time = walk_time + run_time

    walk_ratio = walk_time / total_steep_time if total_steep_time > 0 else 0

    # Compter les segments de marche consécutifs — sur steep uniquement (F7)
    if 'is_walk' in steep.columns and len(steep) > 1:
        walk_series = steep['is_walk'].astype(int)
        n_segments  = int(((walk_series.diff() == 1).sum()))
    else:
        n_segments = 0

    walk_distance = float(steep_walk['dd'].sum()) if 'dd' in steep_walk.columns else 0

    return {
        'walk_ratio':       round(walk_ratio * 100, 1),   # en %
        'walk_time_min':    round(walk_time / 60, 1),
        'run_time_min':     round(run_time / 60, 1),
        'walk_distance_m':  round(walk_distance),
        'n_walk_segments':  n_segments,
        'has_steep':        True,
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

        # Sprint 2 ④ : flag marche active dans ce km
        has_walk = bool(seg['is_walk'].any()) if 'is_walk' in seg.columns else False

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
            'has_walk': has_walk,
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
# RECOMMANDATIONS COACH — v2.0 (Sprint 4 · R1+R2)
# ══════════════════════════════════════════════════════════════════

def _get_race_profile(distance_km: float, duration_s: float) -> str:
    """
    Détecte le profil de course sur critère combiné ET (plus robuste que OU).
    ULTRA  : ≥ 50 km ET ≥ 4h
    LONG   : ≥ 20 km ET ≥ 1h30
    COURT  : tout le reste
    """
    if distance_km >= 50 and duration_s >= 14400:
        return 'ULTRA'
    if distance_km >= 20 and duration_s >= 5400:
        return 'LONG'
    return 'COURT'


def generate_coach_recommendations(
    profile: str,
    fi: dict,
    drift: dict,
    cad_analysis: dict,
    info: dict,
    fcmax: int,
    cad_cv: float = None,   # Option B — paramètre optionnel, rétrocompatible
) -> list:
    """
    Recommandations coach v2.0 — Sprint 4.
    Règles clés :
    - La reco complète le VERDICT, ne le répète pas. Toujours un verbe d'action.
    - Adaptées au profil de course : COURT / LONG / ULTRA
    - Ordre retour : CRIT > WARN > INFO (tri garanti ici)
    - Contrat UI (app.py) : afficher recs[:3] par défaut, expander pour recs[3:]
      → les 3 premières sont donc toujours les plus critiques
    - R-H (point fort) : absent si aucun critère positif identifié
    - category : usage interne uniquement (tri / filtre futur, jamais affiché)
    """
    raw = []   # accumulation avant tri
    dr = fi.get('decay_ratio', float('nan'))
    dp = fi.get('decay_pct', float('nan'))

    drift_pattern = drift.get('pattern')
    drift_pct     = drift.get('drift_pct')
    collapse_pct  = drift.get('collapse_pct')
    insufficient  = drift.get('insufficient_data', False)

    cad_mean    = cad_analysis.get('mean')
    optimal_pct = cad_analysis.get('optimal_pct', 0)
    hr_mean     = info.get('hr_mean')

    distance_km  = info.get('distance_km', 1)
    elev_gain    = info.get('elevation_gain', 0)
    duration_s   = info.get('total_time_s', 0)
    dp_ratio     = elev_gain / distance_km if distance_km > 0 else 0
    dp_tolerance = max(0.0, (dp_ratio - 30) / 10 * 3)

    race_profile = _get_race_profile(distance_km, duration_s)

    gap_crit_threshold = 15 + dp_tolerance
    gap_warn_threshold = 7  + dp_tolerance
    gap_good_threshold = 4  + dp_tolerance

    if dp_ratio >= 50:
        ctx_deniv = f"sur un profil très engagé ({dp_ratio:.0f} m/km D+)"
    elif dp_ratio >= 30:
        ctx_deniv = f"sur un profil montagneux ({dp_ratio:.0f} m/km D+)"
    else:
        ctx_deniv = f"sur ce profil ({dp_ratio:.0f} m/km D+)"

    # ── R-A / R-B : COLLAPSE ────────────────────────────────────
    # Discriminant : collapse_pct (sévérité FC) — pas decay_ratio.
    # CDC Elena v1.2 : seuil -20% pour COLLAPSE B, seuil -35% pour signal sévère.
    # R-A (cp ≤ 35%) : signal cardiaque isolé — surveiller / nutrition
    # R-B (cp > 35%) : signal sévère — récupération / consultation selon profil
    if not insufficient and drift_pattern == 'COLLAPSE':
        cp = abs(collapse_pct) if collapse_pct is not None else 0

        if cp <= 35:
            # R-A : collapse modéré — action nutritionnelle / surveillance
            _body_collapse = {
                'COURT': (
                    f"FC en chute de {cp:.0f}% sur terrain plat. "
                    "Surveiller à la prochaine sortie intense — consulter si le signal se répète."
                ),
                'LONG': (
                    f"Chute FC de {cp:.0f}% détectée en fin de course. "
                    "Tester 40-60g glucides/h dès le départ à l'entraînement."
                ),
                'ULTRA': (
                    f"Chute FC {cp:.0f}% en Q4 — signal nutritionnel classique sur cette distance. "
                    "Surveiller à la prochaine sortie ≥3h avec ravitaillement simulé."
                ),
            }
            raw.append({
                'level': 'crit',
                'category': 'CARDIAQUE',
                'title': 'Surveiller la réponse cardiaque en fin de course',
                'body': _body_collapse[race_profile],
            })
        else:
            # R-B : collapse sévère (> 35%) — récupération prioritaire
            _body_double = {
                'COURT': (
                    f"FC effondrée de {cp:.0f}% — signal sévère. "
                    "Consulter un médecin du sport avant la prochaine compétition. "
                    "Récupération complète 10-14 jours minimum."
                ),
                'LONG': (
                    f"Effondrement FC {cp:.0f}% — double signal cardio + fatigue. "
                    "Semaine de récupération active, reprise progressive 3 semaines en Z1/Z2."
                ),
                'ULTRA': (
                    f"Chute FC {cp:.0f}% — épuisement complet sur ce format. "
                    "Récupération 2-3 semaines. Revoir stratégie allure + nutrition pour la prochaine."
                ),
            }
            raw.append({
                'level': 'crit',
                'category': 'CARDIAQUE',
                'title': "Récupérer avant toute reprise d'intensité",
                'body': _body_double[race_profile],
            })

    # ── R-C : DRIFT ─────────────────────────────────────────────
    elif not insufficient and drift_pattern == 'DRIFT' and drift_pct is not None:
        _body_drift = {
            'COURT': (
                f"EF en baisse de {abs(drift_pct):.1f}% sur la 2ème moitié. "
                "Renforcer l'endurance aérobie : 2 sorties/semaine en Z2 strict (60-70% FCmax), 45-60 min."
            ),
            'LONG': (
                f"Dérive EF de {abs(drift_pct):.1f}% après la mi-course. "
                "Allonger les sorties longues : 1×/semaine ≥2h en Z2 pour repousser le seuil de dérive."
            ),
            'ULTRA': (
                f"Dérive EF {abs(drift_pct):.1f}% — normal sur cette durée, mais optimisable. "
                "Priorité : sorties 3-4h en Z2 avec ravitaillement simulant les conditions course."
            ),
        }
        raw.append({
            'level': 'warn',
            'category': 'ENDURANCE',
            'title': f'Renforcer l\'endurance aérobie : dérive {drift_pct:.1f}%',
            'body': _body_drift[race_profile],
        })

    # ── R-D : GAP critique ───────────────────────────────────────
    if not _isnan(dp) and dp > gap_crit_threshold:
        deniv_note = f" (tolérance {dp_tolerance:.0f}% appliquée {ctx_deniv})" if dp_tolerance > 0 else ""
        _body_gap_crit = {
            'COURT': (
                f"Perte GAP de {dp:.1f}%{deniv_note}. "
                "Départ trop rapide ou manque de fond. Travailler le seuil : 2×15min à FC seuil, 1×/semaine."
            ),
            'LONG': (
                f"Dégradation GAP {dp:.1f}%{deniv_note} — pacing défaillant. "
                "S'entraîner au negative split : départ délibérément 10% plus lent que l'allure cible."
            ),
            'ULTRA': (
                f"Dégradation GAP {dp:.1f}%{deniv_note} — souvent nutritionnel autant que physique. "
                "Revoir la stratégie d'allure Q1→Q2 et le ravitaillement Q3→Q4."
            ),
        }
        raw.append({
            'level': 'crit',
            'category': 'GAP',
            'title': f'Revoir la stratégie de course : -{dp:.1f}% GAP',
            'body': _body_gap_crit[race_profile],
        })

    # ── R-E : GAP modéré ────────────────────────────────────────
    elif not _isnan(dp) and dp > gap_warn_threshold:
        deniv_note = f" ({ctx_deniv})" if dp_tolerance > 0 else ""
        _body_gap_warn = {
            'COURT': (
                f"Fin de course difficile : -{dp:.1f}% GAP{deniv_note}. "
                "Ajouter du foncier : +20% volume hebdo en Z2 sur 4 semaines."
            ),
            'LONG': (
                f"Dernier quart en difficulté : -{dp:.1f}% GAP{deniv_note}. "
                "Negative split 1×/semaine : courir la dernière heure de sortie longue à allure course."
            ),
            'ULTRA': (
                f"Dégradation modérée -{dp:.1f}%{deniv_note} — bonne gestion globale. "
                "Affiner la nutrition de fin de course : +1 gel/30min après km 40."
            ),
        }
        raw.append({
            'level': 'warn',
            'category': 'GAP',
            'title': f'Travailler la résistance en fin de course : -{dp:.1f}% GAP',
            'body': _body_gap_warn[race_profile],
        })

    # ── R-F : Cadence basse ──────────────────────────────────────
    if cad_mean and cad_mean < 168:
        _body_cad_low = {
            'COURT': (
                f"Cadence à {cad_mean:.0f} spm — sous le seuil optimal (170-190 spm). "
                "Drills cadence 2×/semaine : 4×2min à 180 spm cible, récup 1min."
            ),
            'LONG': (
                f"Cadence {cad_mean:.0f} spm — fatigue musculaire probable en fin de course. "
                "Renforcement spécifique : foulées rapides en côte, 10×20s, 2×/semaine."
            ),
            'ULTRA': (
                f"Cadence {cad_mean:.0f} spm — dégradation attendue sur la distance. "
                "Travailler la cadence sur les 10 premiers km quand le contrôle est encore possible."
            ),
        }
        raw.append({
            'level': 'warn',
            'category': 'CADENCE',
            'title': f'Augmenter la cadence de foulée : {cad_mean:.0f} spm',
            'body': _body_cad_low[race_profile],
        })

    # ── R-G : Cadence irrégulière (cad_cv) ──────────────────────
    # Si cad_cv passé explicitement (Option B) : on l'utilise directement.
    # Sinon : proxy via optimal_pct — si < 50% du temps en zone optimale
    # ET cadence correcte en moyenne → signe de variabilité élevée.
    # Note : calcul sur dist bins serait incorrect (dispersion des bins ≠ CV cadence).
    _cad_cv = cad_cv
    _show_rg = False
    if _cad_cv is not None:
        _show_rg = _cad_cv > 0.08
    elif cad_mean and cad_mean >= 168 and optimal_pct is not None and optimal_pct < 50:
        # Cadence moyenne OK mais peu de temps en zone → variabilité élevée
        _show_rg = True

    if _show_rg:
        raw.append({
            'level': 'info',
            'category': 'CADENCE',
            'title': 'Stabiliser la régularité de foulée',
            'body': (
                "Irrégularité de cadence détectée — signal de fatigue neuromusculaire ou terrain technique. "
                "À surveiller sur la prochaine sortie longue."
            ),
        })

    # ── R-H : Point fort — affiché uniquement si critère positif ─
    strength = _strength_point(profile, drift_pattern, drift_pct, cad_mean, dp,
                               gap_good_threshold, dr, race_profile)
    if strength:
        raw.append({
            'level': 'info',
            'category': 'POINT_FORT',
            'title': strength['title'],
            'body':  strength['body'],
        })

    # ── Tri final : CRIT > WARN > INFO ───────────────────────────
    _order = {'crit': 0, 'warn': 1, 'info': 2}
    recs = sorted(raw, key=lambda r: _order.get(r['level'], 9))

    # ── Fallback Kai : fichier sans FC → message explicite ───────
    # Déclenché si peu de recos ET pas de FC — évite un expander quasi-vide
    if len(recs) < 2 and not info.get('has_hr', True):
        recs.append({
            'level': 'info',
            'category': 'DATA',
            'title': 'Analyse partielle — données FC manquantes',
            'body': "Importe un fichier avec fréquence cardiaque pour une analyse cardiaque complète.",
        })

    return recs[:6]


def _strength_point(profile, drift_pattern, drift_pct, cad_mean, dp,
                    gap_good_threshold, dr, race_profile):
    """
    Retourne un dict {title, body} uniquement si un critère positif est identifié.
    Retourne None si aucun signal positif — R-H n'est alors pas ajoutée.
    Priorité : STABLE > bon GAP > bonne cadence > profil ENDURANCE
    """
    if drift_pattern == 'STABLE' and drift_pct is not None and drift_pct > -3:
        _body = {
            'COURT': "Efficacité cardiaque stable sur tout l'effort — base aérobie solide. Monter en intensité : 1 séance VMA/semaine.",
            'LONG':  "Endurance cardiovasculaire bien développée. Capitaliser : ajouter +10% de volume D+ par cycle de 3 semaines.",
            'ULTRA': "Gestion cardiaque solide sur la durée. Exploiter cette base : sorties longues avec dénivelé progressif.",
        }
        return {'title': 'Maintenir cette efficacité cardiaque', 'body': _body[race_profile]}

    if not _isnan(dp) and not _isnan(dr) and dp < gap_good_threshold:
        _body = {
            'COURT': "Gestion d'allure optimale du début à la fin. Franchir un palier : 2×/semaine fractionné court (6-8×400m à 95% VMA).",
            'LONG':  f"Pacing excellent ({f'-{dp:.1f}% GAP' if not _isnan(dp) and dp > 0 else 'allure stable'}). Exploiter cette régularité : cibler une distance supérieure.",
            'ULTRA': "Stratégie de course maîtrisée sur la distance. Progresser : augmenter le dénivelé cumulé de la prochaine sortie longue.",
        }
        return {'title': "Capitaliser sur cette gestion d'allure", 'body': _body[race_profile]}

    if cad_mean and cad_mean >= 180:
        return {
            'title': 'Maintenir cette cadence de foulée',
            'body': f"Cadence à {cad_mean:.0f} spm — économie de course solide. Exploiter : chaussures plus légères et travail de côtes courtes.",
        }

    if profile == 'PROFIL ENDURANCE':
        _body = {
            'COURT': "Profil endurance identifié — moteur aérobie ton atout. Ajouter du volume D+ : +10% par cycle de 3 semaines.",
            'LONG':  "Profil endurance solide. Transformer en puissance : côtes longues 6-10min à VMA montée, 1×/semaine.",
            'ULTRA': "Profil endurance adapté au format. Consolider : sorties 4h+ avec ravitaillement simulant les conditions course.",
        }
        return {'title': 'Exploiter ce profil endurance', 'body': _body[race_profile]}

    return None  # aucun signal positif → R-H absente


# ══════════════════════════════════════════════════════════════════
# SCORE GLOBAL — Sprint 2 item ⑧
# ══════════════════════════════════════════════════════════════════

def compute_performance_score(fi: dict, drift: dict, dp_per_km: float = 0.0) -> dict:
    """
    Score global de performance VERTEX — 0 à 100.

    Pondération Elena :
      GAP Q4/Q1          50%  — endurance moteur
      Dérive EF          35%  — efficacité cardiaque
      Variance Q1→Q4     15%  — régularité de l'effort

    Règles absolues :
      - Si EF insufficient ou pattern COLLAPSE → poids EF redistribués au GAP
      - Score partiel signalé explicitement dans le return
      - Chaque composante exposée séparément pour l'affichage
      - Si dp_per_km > 40 m/km → variance neutralisée (CV mécanique terrain)

    Returns dict :
      score          int   0-100
      score_gap      int   0-100  (composante GAP)
      score_ef       int | None   (composante EF — None si non disponible)
      score_var      int   0-100  (composante variance)
      partial        bool  True si score partiel
      partial_reason str | None
      weights        dict  poids réels utilisés
      var_neutralized bool  True si variance neutralisée (D+ > 40 m/km)
    """
    # ── Composante 1 : GAP Q4/Q1 ────────────────────────────────
    # decay_ratio : 1.0 = parfait, 0.0 = effondrement total
    # On normalise [0.7, 1.0] → [0, 100] (en dessous de 0.70 c'est catastrophique)
    decay_ratio = fi.get('decay_ratio', float('nan'))
    if _isnan(decay_ratio):
        score_gap = 0
    else:
        score_gap = int(round(max(0, min(100, (decay_ratio - 0.70) / 0.30 * 100))))

    # ── Composante 2 : Variance inter-quartiles Q1→Q4 ───────────
    # Mesure la régularité : faible variance = bon score
    # On calcule l'écart-type des 4 quartiles GAP, normalisé
    # P3 : si D+ > 40 m/km → CV mécanique terrain → neutralisé à 50
    var_neutralized = dp_per_km > 40.0
    quartiles = fi.get('quartiles', {})
    q_vals = [v for v in quartiles.values() if v is not None and not _isnan(v)]
    if var_neutralized:
        score_var = 50  # neutre — CV GAP non interprétable sur parcours très montagneux
    elif len(q_vals) >= 2:
        q_arr   = np.array(q_vals)
        q_mean  = float(np.mean(q_arr))
        q_std   = float(np.std(q_arr))
        # CV (coefficient de variation) : 0% = parfait
        # Seuil 15% : variance normale sur trail technique (F8 — seuil 10% trop pénalisant)
        # Au-delà de 15% CV = gestion effort réellement problématique
        cv = (q_std / q_mean * 100) if q_mean > 0 else 15
        score_var = int(round(max(0, min(100, (1 - cv / 15) * 100))))
    else:
        score_var = 50  # neutre si données insuffisantes

    # ── Composante 3 : Dérive EF ─────────────────────────────────
    # drift_pct : 0% = parfait, -20% = très mauvais
    # COLLAPSE ou insufficient → composante non disponible
    pattern      = drift.get('pattern')
    insufficient = drift.get('insufficient_data', False)
    drift_pct    = drift.get('drift_pct')

    ef_unavailable = insufficient or pattern == 'COLLAPSE'

    if ef_unavailable or drift_pct is None:
        score_ef = None
        partial  = True
        if pattern == 'COLLAPSE':
            partial_reason = "Score partiel — effondrement CV détecté (EF non interprétable)"
        else:
            partial_reason = "Score partiel — terrain insuffisamment plat pour calculer l'EF"
    else:
        # drift_pct ∈ [-20, 0] → score ∈ [0, 100]
        # Au-delà de -20% on plafonne à 0
        score_ef       = int(round(max(0, min(100, (1 + drift_pct / 20) * 100))))
        partial        = False
        partial_reason = None

    # ── Pondération réelle ───────────────────────────────────────
    if ef_unavailable:
        # Redistribution des 35% EF → GAP (total GAP = 85%)
        w_gap = 0.85
        w_ef  = 0.00
        w_var = 0.15
    else:
        w_gap = 0.50
        w_ef  = 0.35
        w_var = 0.15

    # ── Score final ──────────────────────────────────────────────
    ef_contrib = score_ef if score_ef is not None else 0
    score_raw  = w_gap * score_gap + w_ef * ef_contrib + w_var * score_var
    score      = int(round(min(100, max(0, score_raw))))

    return {
        'score':          score,
        'score_gap':      score_gap,
        'score_ef':       score_ef,
        'score_var':      score_var,
        'partial':        partial,
        'partial_reason': partial_reason,
        'var_neutralized': var_neutralized,
        'weights': {
            'gap': w_gap,
            'ef':  w_ef,
            'var': w_var,
        },
    }


# ══════════════════════════════════════════════════════════════════
# VERDICT MATRICE — P2 Elena v2
# ══════════════════════════════════════════════════════════════════

def compute_verdict(fi: dict, drift: dict, perf: dict) -> dict:
    """
    Matrice verdict V1–V7 Elena v2.

    Priorité d'évaluation (ordre strict) :
      1. Données insuffisantes → ℹ ANALYSE INSUFFISANTE
      2. COLLAPSE + decay > 0.90 + score > 75 → V7 Signal cardiaque anormal
      3. COLLAPSE + decay 0.85–0.90              → V3 Dégradation progressive
         (collapse nutritionnel/thermique partiel — allure partiellement tenue)
      4. COLLAPSE (seul, decay < 0.85)           → V6 Signal cardiaque anormal
      5. decay < 0.80 → V5 Effondrement de l'allure
      6. DRIFT + decay < 0.90 + score < 50 → V4 Fatigue combinée
      7. decay 0.80–0.90 → V3 Dégradation progressive
      8. score 50–75 → V2 Performance correcte
      9. score > 75 → V1 Performance solide

    Returns dict :
      code        str   'V1'…'V7' | 'INSUFFICIENT'
      label       str   Texte affiché (majuscules, Barlow Condensed)
      sub         str   Ligne contextuelle (DM Mono)
      color       str   Hex couleur principale
      icon        str   ✓ / ~ / ✕ / ⚠ / ℹ
    """
    decay_ratio  = fi.get('decay_ratio', float('nan'))
    pattern      = drift.get('pattern')
    insufficient = drift.get('insufficient_data', False)
    score        = perf.get('score', 0)
    drift_pct    = drift.get('drift_pct')
    collapse_pct = drift.get('collapse_pct')
    decay_pct    = fi.get('decay_pct', float('nan'))

    # ── Données insuffisantes ────────────────────────────────────
    if _isnan(decay_ratio):
        return {
            'code':  'INSUFFICIENT',
            'label': 'ANALYSE INSUFFISANTE',
            'sub':   "GAP non calculable — fichier trop court ou vitesse nulle.",
            'color': '#2A4050',
            'icon':  'ℹ',
        }

    # ── V7 : COLLAPSE + allure tenue + score élevé ───────────────
    if pattern == 'COLLAPSE' and decay_ratio > 0.90 and score > 75:
        cp = abs(collapse_pct) if collapse_pct is not None else 0
        return {
            'code':  'V7',
            'label': 'SIGNAL CARDIAQUE ANORMAL',
            'sub':   f"FC effondrée de {cp:.1f}% malgré une allure maintenue — dissociation CV/effort à surveiller.",
            'color': '#C84850',
            'icon':  '⚠',
        }

    # ── V3-COLLAPSE : COLLAPSE + decay 0.85–0.90 ─────────────────
    # Collapse nutritionnel ou thermique partiel : allure partiellement tenue
    # (decay > 0.85 = perte GAP < 15%) → dégradation progressive, pas anomalie franche
    # BUG-1 Sprint 4A — CDC Elena v1.2 validé réunion contrôle
    if pattern == 'COLLAPSE' and not _isnan(decay_ratio) and decay_ratio >= 0.85:
        dp = decay_pct if not _isnan(decay_pct) else 0
        cp = abs(collapse_pct) if collapse_pct is not None else 0
        return {
            'code':  'V3',
            'label': 'DÉGRADATION PROGRESSIVE',
            'sub':   f"Chute FC {cp:.1f}% sur plat · écart GAP Q4/Q1 : {dp:.1f}% — fatigue cardiaque partielle.",
            'color': '#C8A84B',
            'icon':  '~',
        }

    # ── V6 : COLLAPSE franc (decay < 0.85) ───────────────────────
    if pattern == 'COLLAPSE':
        cp = abs(collapse_pct) if collapse_pct is not None else 0
        return {
            'code':  'V6',
            'label': 'SIGNAL CARDIAQUE ANORMAL',
            'sub':   f"FC effondrée de {cp:.1f}% sur terrain plat — effort non interprétable par l'EF.",
            'color': '#C84850',
            'icon':  '⚠',
        }

    # ── V5 : Effondrement allure ─────────────────────────────────
    if not _isnan(decay_ratio) and decay_ratio < 0.80:
        dp = decay_pct if not _isnan(decay_pct) else 0
        return {
            'code':  'V5',
            'label': 'EFFONDREMENT DE L\'ALLURE',
            'sub':   f"Écart GAP Q4/Q1 : {dp:.1f}% — dégradation d'allure sur la durée.",
            'color': '#C84850',
            'icon':  '✕',
        }

    # ── V4 : Fatigue combinée ─────────────────────────────────────
    if pattern == 'DRIFT' and not _isnan(decay_ratio) and decay_ratio < 0.90 and score < 50:
        dp  = decay_pct if not _isnan(decay_pct) else 0
        dft = abs(drift_pct) if drift_pct is not None else 0
        return {
            'code':  'V4',
            'label': 'FATIGUE COMBINÉE',
            'sub':   f"Dérive EF {dft:.1f}% + écart GAP {dp:.1f}% — double signal de surmontée.",
            'color': '#C84850',
            'icon':  '✕',
        }

    # ── V3 : Dégradation progressive ─────────────────────────────
    if not _isnan(decay_ratio) and 0.80 <= decay_ratio < 0.90:
        dp = decay_pct if not _isnan(decay_pct) else 0
        return {
            'code':  'V3',
            'label': 'DÉGRADATION PROGRESSIVE',
            'sub':   f"Écart GAP Q4/Q1 : {dp:.1f}% — progression possible sur l'endurance spécifique.",
            'color': '#C8A84B',
            'icon':  '~',
        }

    # ── V2 : Performance correcte ────────────────────────────────
    if score < 75:
        ef_ctx = f"Dérive EF : {abs(drift_pct):.1f}%" if drift_pct is not None else "EF non disponible"
        return {
            'code':  'V2',
            'label': 'PERFORMANCE CORRECTE',
            'sub':   f"Score {score}/100 · {ef_ctx} — marge de progression identifiée.",
            'color': '#C8A84B',
            'icon':  '~',
        }

    # ── V1 : Performance solide ───────────────────────────────────
    dp = decay_pct if not _isnan(decay_pct) else 0
    return {
        'code':  'V1',
        'label': 'PERFORMANCE SOLIDE',
        'sub':   f"Score {score}/100 · Écart GAP Q4/Q1 : {dp:.1f}% — allure et efficacité cardiaque maîtrisées.",
        'color': '#41C8E8',
        'icon':  '✓',
    }
