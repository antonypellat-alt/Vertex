"""B7 validation — SCR-EF1 : score_ef non-None sur profils montagneux."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpx_parser import parse_gpx as _parse_gpx, extract_race_info
try:
    from fit_parser import parse_fit as _parse_fit
except ImportError:
    _parse_fit = None

from engine import (
    fatigue_index, detect_elevation_profile, apply_decay_correction,
    prepare_analysis, compute_performance_score,
)

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset GPX")

DATASETS = [
    ("Jeremy 26km.gpx",           "Jeremy",  "STABLE", "GAP_FALLBACK"),
    ("Julien Eynavay Trail.fit",   "Julien",  "STABLE", "GAP_FALLBACK"),
]

results = []

for filename, athlete, expected_pattern, expected_source in DATASETS:
    print(f"\n{'='*60}")
    path = os.path.join(DATASETS_DIR, filename)
    print(f"DATASET : {filename}  ({athlete})")
    print(f"Path    : {path}")

    if not os.path.exists(path):
        print(f"  ❌ FICHIER INTROUVABLE — skip")
        results.append((filename, False, "fichier introuvable"))
        continue

    ext = filename.rsplit(".", 1)[-1].lower()
    try:
        with open(path, "rb") as fh:
            file_bytes = fh.read()
        if ext == "gpx":
            df = _parse_gpx(file_bytes)
        elif ext == "fit":
            if _parse_fit is None:
                print("  ❌ fit_parser non disponible — skip")
                results.append((filename, False, "fit_parser absent"))
                continue
            df = _parse_fit(file_bytes)
        else:
            print(f"  ❌ format inconnu : {ext}")
            results.append((filename, False, f"format inconnu {ext}"))
            continue
    except Exception as e:
        print(f"  ❌ ERREUR parsing : {e}")
        results.append((filename, False, f"erreur parsing: {e}"))
        continue

    info         = extract_race_info(df, filename)
    fi           = fatigue_index(df)
    elev_profile = detect_elevation_profile(df)
    fi           = apply_decay_correction(fi, elev_profile, df)

    analysis = prepare_analysis(fi, elev_profile, df, info)
    drift    = analysis['drift']
    fi_score = analysis['fi_score']
    dp       = analysis['dp_per_km']

    perf = compute_performance_score(fi_score, drift, dp_per_km=dp)

    pattern   = drift.get('pattern')
    ef_source = drift.get('ef_source', 'FLAT')
    ef_slope  = drift.get('ef_slope_pph')
    drift_pct = drift.get('drift_pct')
    score_ef  = perf.get('score_ef')
    partial   = perf.get('partial')
    score     = perf.get('score')
    insuf     = drift.get('insufficient_data')

    print(f"  has_hr       : {info.get('has_hr')}")
    print(f"  dp_per_km    : {dp:.1f}")
    print(f"  distance_km  : {info.get('distance_km', '?'):.1f}")
    print(f"  duration_s   : {info.get('total_time_s', '?')}")
    print(f"  insufficient : {insuf}")
    print(f"  pattern      : {pattern}  (attendu: {expected_pattern})")
    print(f"  ef_source    : {ef_source}  (attendu: {expected_source})")
    print(f"  ef_slope_pph : {ef_slope}")
    print(f"  drift_pct    : {drift_pct}")
    print(f"  score_ef     : {score_ef}  (attendu: non-None)")
    print(f"  partial      : {partial}")
    print(f"  score_final  : {score}")

    ok = score_ef is not None
    status = "✅ PASS" if ok else "❌ FAIL — score_ef est None"
    print(f"  B7 {status}")
    results.append((filename, ok, status))

    if not ok:
        print(f"\n  === drift complet ===")
        for k, v in drift.items():
            print(f"    {k}: {v}")

print(f"\n{'='*60}")
print("RÉSUMÉ B7 :")
all_pass = True
for fname, ok, msg in results:
    mark = "✅" if ok else "❌"
    print(f"  {mark} {fname}")
    if not ok:
        all_pass = False

if all_pass and results:
    print("\n  → SCR-EF1 peut être clos ✅")
else:
    print("\n  → SCR-EF1 bloqué ❌ — voir diagnostic ci-dessus")
