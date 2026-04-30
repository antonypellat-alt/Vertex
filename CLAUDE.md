\# Projet VERTEX — Contexte



Application Python/Streamlit d'analyse de performance trail running.

Entrée : fichiers GPX/TCX/FIT. Sortie : score physiologique + rapport PDF.

Positionnement : "Strava montre ce que tu as fait. VERTEX explique ce que ça t'a coûté."

Cible : élites trail mondiaux. Déployé sur Streamlit Community Cloud.



\---



\## Stack \& Architecture



\*\*Python\*\* : 3.11–3.14 (requis par scipy 1.17.1)

\*\*Dépendances critiques\*\* :

\- `streamlit` (lancement : `python -m streamlit run app.py`)

\- `pandas==2.2.3` — NE PAS upgrader (Copy-on-Write pandas 3.0 casse le code)

\- `scipy==1.17.1`

\- `plotly`

\- `numpy`

\- `fpdf2` — PDF dark theme via override `header()`

\- `lxml` — parsing GPX



\*\*Architecture modulaire\*\* :

```

gpx\_parser.py   → parsing GPX, Haversine, extract\_race\_info

tcx\_parser.py   → parsing TCX

fit\_parser.py   → parsing FIT (Garmin natif)

engine.py       → GAP (Minetti 2002), cardiac\_drift, score, VERDICT

charts.py       → graphiques Plotly + générateur PDF (VertexPDF)

app.py          → UI Streamlit uniquement (ce fichier)

test\_engine.py  → suite de tests (sections A→S)

```



\*\*Déploiement\*\* : Streamlit Community Cloud · repo `vertex` · branche `main`

\*\*Feedback\*\* : Tally.so (https://tally.so/r/zxeJPM) → Google Sheets



\---



\## Commandes essentielles

```bash

\# Lancer l'application

python -m streamlit run app.py



\# Lancer les tests

python -m pytest test\_engine.py -v



\# Déployer : git push sur main → auto-déploiement Streamlit Cloud

```



\---



\## Conventions de code



\- \*\*Corrections chirurgicales uniquement\*\* — jamais réécrire un fichier entier sans accord explicite

\- Fonctions vectorisées numpy prioritaires sur `.apply()` pandas

\- `@st.cache\_data` sur toutes les fonctions de parsing

\- PDF : fonts Helvetica + Courier natifs FPDF2 uniquement (pas DejaVu)

\- Imports modules VERTEX : toujours en haut de `app.py`, jamais inline

\- Cadence GPX Garmin : valeur brute unilatérale → multiplier ×2 si < 110 spm



\---



\## Fonctionnalités actives



\- \*\*GAP Engine\*\* : modèle Minetti 2002, version scalaire + vectorisée

\- \*\*cardiac\_drift()\*\* CDC v1.4 : COLLAPSE A/B · NEGATIVE\_SPLIT · DRIFT-CARDIO · DRIFT-NEURO · DRIFT · STABLE

\- \*\*Score performance\*\* : zones dynamiques (FLAT/ASCENDING/DESCENDING) + pondération adaptative

\- \*\*VERDICT\*\* : matrice V1→V7 + V5-C + INSUFFICIENT + V1-NS

\- \*\*PDF\*\* : dark theme, hero zone, recommandations, disclaimer médical

\- \*\*Tally form\*\* : placé HORS des blocs `if st.button` (persist on rerender)

\- \*\*Couche traduction athlète\*\* : zéro terme interne visible (pas de EF, decay\_ratio, COLLAPSE A/B)



\---



\## Sprint actif



\*\*Sprint 5\*\* — EN COURS.

Tâches validées :

\- Kai : ligne action immédiate sous VERDICT ("et donc tu fais quoi")

\- VERDICT BUG-2 : COLLAPSE + decay < 0.80 → V5-C "EFFONDREMENT TOTAL"

\- Option B : V2 + DRIFT-CARDIO → sub enrichi fc\_slope chiffré

\- Sara : fenêtre Instagram à activer (signal Dylan/Kiliane)



\*\*Bloquants externes\*\* :

\- GPX Justine · GPX terrain roulant (Alexandre priorité) · FCmax Kiliane/Alexandre

\- GPX "essoufflement" ≥25km ×2 (Adrien/Thibault/Dylan)

\- ELENA② GPX Emmanuel (6–8 sem)



\---



\## Bugs connus \& décisions figées



\*\*BUG-2\*\* : COLLAPSE + decay < 0.80 → doit déclencher V5-C. Corrigé dans matrice, à intégrer Sprint 5.

\*\*Faux positif V7\*\* : angle mort structurel départ montée (dataset Coralie Toureille). Comportement connu, non corrigé.

\*\*GAP dégradé < 2–3% gradient\*\* : limitation connue modèle Minetti. Disclaimer en place.

\*\*EF biaisée faible D+\*\* : SCI-1 disclaimer actif, SCI-3 étude en cours.



\*\*Décisions figées — ne jamais remettre en question\*\* :

\- `pandas==2.2.3` — version verrouillée

\- Seuil COLLAPSE A : slope < -3.0 bpm/h + chute > 10% (CDC Elena v1.4)

\- Seuil COLLAPSE B : chute > 20% sur segments plats

\- Ultra : ≥50km \*\*ET\*\* ≥4h (critère ET strict — validé Elena CDC)

\- Tally form hors bloc `if st.button`



\---



\## Règles absolues



\- Ne \*\*jamais\*\* exposer les termes internes côté athlète : `EF`, `decay\_ratio`, `COLLAPSE A/B`, `dissociation CV`

\- Ne \*\*jamais\*\* toucher au design system sans validation : `#080E14` · `#41C8E8` · `#C84850` · `#C8A84B` · Barlow Condensed + DM Mono

\- Ne \*\*jamais\*\* upgrader `pandas` au-delà de 2.2.3

\- Ne \*\*jamais\*\* proposer de réécriture complète d'un fichier sans demande explicite

\- Toujours inclure disclaimers médicaux dans UI et PDF

\- Analytics Streamlit désactivé via `.streamlit/config.toml` — ne pas réactiver



\---



\## Fichiers clés



| Fichier | Rôle | Sensibilité |

|---|---|---|

| `engine.py` | Logique métier complète | 🔴 critique |

| `app.py` | UI Streamlit | 🔴 critique |

| `charts.py` | Graphiques + PDF | 🟠 élevée |

| `gpx\_parser.py` | Parsing GPX | 🟠 élevée |

| `test\_engine.py` | Suite tests A→S | 🟡 importante |

| `.streamlit/config.toml` | Analytics off | 🟡 importante |



\*\*Règle absolue tests\*\* : aucun push sur `main` sans 100% tests verts.

Sections actives : A→S + C10/C11/C12 + K. État actuel : 143/144 — 1 échec connu M3 (PDF env).

---

\## Specs actives — format P1 obligatoire

\### Règle
Tout item backlog doit avoir une spec écrite ici avant d'être codé.
Item sans spec = item inexistant.

\### Format
[ID] — Titre court
- Problème : symptôme observé ou besoin
- Fichiers concernés : engine.py / app.py / charts.py
- Données requises : GPX / dataset / référence
- Critère de clôture : ce qui prouve que c'est résolu
- Bloquant : oui / non

\### Items actifs
— aucun item ouvert au 30/04/2026 · Sprint 9 clos —

