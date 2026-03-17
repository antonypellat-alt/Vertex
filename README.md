> ⚠️ **License**: BSL 1.1 — Usage commercial interdit sans autorisation.
> © 2025 VERTEX Performance Intelligence
> 
# Vertex
# ▲ VERTEX — Performance Intelligence

> **Transforme un fichier GPX en analyse physiologique complète.**  
> Local-first · No account · No cloud · Données 100% sur ta machine.

---

## Ce que fait VERTEX

VERTEX analyse tes sorties trail avec les mêmes outils qu'un laboratoire de physiologie du sport :

- **GAP Engine** — Allure ajustée au dénivelé via le modèle Minetti (2002)
- **Profil fatigue** — Analyse Q1→Q4 : détecte le décrochage de vitesse en fin de course
- **Zones FC** — Distribution Z1→Z5 calibrée sur ta FCmax réelle ou tes zones personnalisées
- **Découplage cardiaque** — Efficiency Factor + dérive sur terrain plat
- **Cadence** — Distribution SPM, zone optimale, évolution sur la course
- **Splits par km** — Allure / GAP / FC / Cadence / D+ par kilomètre
- **Recommandations coach** — 6 conseils personnalisés basés sur tes données
- **Export PDF** — Rapport complet prêt à partager avec ton coach

---

## Installation

```bash
# 1. Cloner le repo
git clone https://github.com/ton-username/vertex.git
cd vertex

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer l'app
python -m streamlit run app.py
```

---

## Utilisation

1. **Renseigne ton profil** — FCmax et mode zones (auto ou manuel) sur la page d'accueil
2. **Importe ton fichier GPX** — Export depuis Garmin Connect (fichier original avec FC et cadence)
3. **Analyse** — Tous les calculs s'exécutent localement, aucune donnée ne quitte ta machine
4. **Génère le PDF** — Rapport complet téléchargeable

> **Garmin Connect** → Activité → `···` → *Exporter l'original*  
> **Strava** → Ne pas utiliser l'export Strava (supprime la FC). Passer par Garmin Connect directement.

---

## Stack technique

| Composant | Librairie |
|-----------|-----------|
| Interface | Streamlit |
| Calculs | Pandas |
| Graphiques | Plotly |
| Export PDF | FPDF2 |
| Modèle GAP | Minetti et al. (2002) — *J. Experimental Biology* |

---

## Références scientifiques

- **Minetti et al. (2002)** — Energy cost of walking and running at extreme uphill and downhill slopes. *Journal of Experimental Biology*
- **Friel, J. (2009)** — The Triathlete's Training Bible — Efficiency Factor & zones FC
- **Daniels, J. (2005)** — Daniels' Running Formula — Cadence optimale

---

## Roadmap

- [ ] Vectorisation GAP (numpy)
- [ ] Cache Streamlit (`@st.cache_data`)
- [ ] Filtre Savitzky-Golay sur l'altitude
- [ ] Historique local (SQLite)
- [ ] Profil athlète persistant
- [ ] Version Coach Edition (PDF brandé)

---

## Licence

Projet en développement — usage personnel et coaching.

---

*VERTEX v3.2 — Performance Intelligence*
