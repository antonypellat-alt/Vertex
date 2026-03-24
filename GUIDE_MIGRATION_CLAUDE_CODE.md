# Guide Migration Claude Code — VERTEX (Windows)
*Marcus · Head of Engineering · Document opérationnel*

---

## ÉTAPE 1 — Prérequis Windows

Claude Code sur Windows nécessite **Git Bash** (pas juste PowerShell).

**1.1 — Installer Git for Windows**
Télécharger et installer depuis : https://git-scm.com/download/win
→ Cocher "Git Bash Here" pendant l'installation.

**1.2 — Vérifier Git Bash**
Ouvrir Git Bash (clic droit sur le bureau → "Git Bash Here") et taper :
```bash
git --version
```
Résultat attendu : `git version 2.x.x`

**1.3 — Installer Node.js (méthode npm uniquement)**
Si tu passes par npm (voir étape 2), Node.js 18+ est requis.
Télécharger la version LTS depuis : https://nodejs.org
Vérification dans Git Bash :
```bash
node --version   # doit afficher v18.x.x ou supérieur
npm --version    # doit afficher 10.x.x ou supérieur
```
> ⚠️ Si tu utilises l'installateur natif PowerShell (recommandé), Node.js n'est PAS nécessaire.

---

## ÉTAPE 2 — Installer Claude Code

**Option A — Installateur natif (recommandé, sans Node.js)**
Dans PowerShell (pas besoin d'admin) :
```powershell
winget install Anthropic.ClaudeCode
```
Ou depuis PowerShell :
```powershell
irm https://claude.ai/install.ps1 | iex
```
Avantage : auto-update en arrière-plan, aucune dépendance Node.js.

**Option B — npm (si tu veux rester dans l'écosystème npm)**
Dans Git Bash :
```bash
npm install -g @anthropic-ai/claude-code
```
> ⚠️ Ne jamais utiliser `sudo npm install`. Sur Windows Git Bash, pas de sudo de toute façon.

**Vérification installation :**
```bash
claude --version
```
Résultat attendu : numéro de version (ex: `2.x.x`)

---

## ÉTAPE 3 — Authentification

Depuis Git Bash ou PowerShell, dans n'importe quel dossier :
```bash
claude
```
Claude Code demande de t'authentifier. Il ouvre ton navigateur → connecte-toi avec le compte Anthropic lié à ton abonnement Pro/Max.

> ⚠️ Compte Claude.ai gratuit = authentification refusée. Il faut un abonnement **Pro ou Max**.

---

## ÉTAPE 4 — Placer le CLAUDE.md dans le projet VERTEX

1. Copier le fichier `CLAUDE.md` livré par Marcus à la **racine du repo VERTEX** :
```
C:\...\vertex\CLAUDE.md
```
2. Le committer immédiatement :
```bash
git add CLAUDE.md
git commit -m "feat: add CLAUDE.md context file"
git push origin main
```
Claude Code le charge automatiquement à chaque session dans ce dossier.

---

## ÉTAPE 5 — Créer le CLAUDE.md global (préférences transverses)

Ce fichier s'applique à **tous** tes projets, pas seulement VERTEX.
Créer le fichier ici :
```
C:\Users\<TonNom>\.claude\CLAUDE.md
```
Contenu recommandé pour Antony :
```markdown
# Préférences globales

- Réponses en français
- Corrections chirurgicales — jamais de réécriture complète sans demande explicite
- Explication en une phrase avant toute correction
- Développeur no-code : logique oui, syntaxe détaillée non
- Windows + Git Bash
```

---

## ÉTAPE 6 — Lancer la première session VERTEX

Dans Git Bash, naviguer vers la racine du projet :
```bash
cd /c/Users/<TonNom>/<chemin-vers-vertex>
claude
```
Claude Code scanne le projet et charge le CLAUDE.md. Premier scan ~10-20 secondes.

---

## ÉTAPE 7 — Vérifier que le contexte est chargé

Ce que Marcus veut voir au démarrage d'une session :
Taper dans Claude Code :
```
Quels sont les fichiers clés du projet et quel est le sprint actif ?
```
Réponse attendue : liste de `engine.py`, `app.py`, `charts.py`, `gpx_parser.py`, mention du Sprint 5.
Si la réponse est générique → le CLAUDE.md n'est pas chargé. Vérifier son emplacement.

---

## ÉTAPE 8 — Les 5 commandes slash à maîtriser

| Commande | Usage |
|---|---|
| `/compact` | Résume la session en cours pour libérer le contexte (à faire toutes les ~2h) |
| `/clear` | Repart de zéro — nouvelle session propre |
| `/memory` | Affiche ce que Claude Code a mémorisé globalement |
| `/cost` | Affiche le coût de la session en cours |
| `/doctor` | Vérifie que l'installation est saine |

---

## ÉTAPE 9 — Forcer une mise à jour du CLAUDE.md en session

Si tu modifies le CLAUDE.md pendant une session (nouveau sprint, bug résolu), forcer le rechargement avec le raccourci `#` :
```
# Le sprint 5 est maintenant soldé. Mettre à jour le contexte.
```
Le `#` en début de message signale à Claude Code de re-lire les fichiers de configuration.

---

## ÉTAPE 10 — Protocole de fin de session

Avant de quitter chaque session Claude Code :

1. **Vérifier les tests** :
```bash
python -m pytest test_engine.py -v
```
Aucun push si un test est rouge (sauf échecs environnement plotly/fpdf2 connus).

2. **Committer le travail** :
```bash
git add -A
git commit -m "feat: <description concise>"
git push origin main
```

3. **Mettre à jour le CLAUDE.md** si le sprint a avancé :
- Marquer les tâches terminées dans `## Sprint actif`
- Mettre à jour les bugs résolus dans `## Bugs connus`

4. **Utiliser `/compact`** pour archiver le contexte de la session avant `/clear`.

---

## CHECKLIST VALIDATION MIGRATION — 10 POINTS

- [ ] **1.** Git Bash installé et fonctionnel (`git --version` répond)
- [ ] **2.** Claude Code installé (`claude --version` répond)
- [ ] **3.** Authentification réussie (compte Pro ou Max)
- [ ] **4.** `CLAUDE.md` à la racine du repo VERTEX et commité sur `main`
- [ ] **5.** `~/.claude/CLAUDE.md` global créé avec préférences Antony
- [ ] **6.** Première session lancée depuis le dossier VERTEX (`cd` correct)
- [ ] **7.** Vérification contexte chargé (réponse correcte sur fichiers clés + sprint actif)
- [ ] **8.** Commandes `/compact`, `/clear`, `/cost`, `/doctor` testées
- [ ] **9.** `python -m pytest test_engine.py -v` lancé depuis Claude Code → 129/134 verts
- [ ] **10.** Premier commit post-migration poussé sur `main` sans régression

---

## VERDICT D'ARCHITECTE

**Ce qui est solide :**
- Architecture modulaire propre — séparation UI / moteur / charts / parsers irréprochable
- Suite de tests bien structurée — le garde-fou existe, il fonctionne
- Décisions techniques figées documentées — aucun risque de regression involontaire

**Ce qui est fragile :**
- `pandas==2.2.3` verrouillé à vie — toute dépendance tierce qui pull pandas 3.x casse silencieusement
- 5 tests échec environnement (plotly/fpdf2) — acceptables en dev, **bloquants** si CI/CD un jour
- CLAUDE.md devra être mis à jour manuellement à chaque sprint soldé — risque de dérive contexte

**3 risques à surveiller Sprint 5+ :**
1. **Dérive du CLAUDE.md** : si non mis à jour à chaque sprint, Claude Code travaille sur contexte obsolète → corrections sur mauvaise base
2. **Dépendances transverses** : `requirements.txt` non versionné dans la mémoire — vérifier qu'il est à jour dans le repo avant tout onboarding nouveau contributeur
3. **Faux positifs VERDICT** (angle mort V7) : bug connu non corrigé — si un beta testeur tombe dessus sans contexte, signal de confiance négatif

---
*Marcus — VERTEX Engineering · Document à placer à la racine du repo*
