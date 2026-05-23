# 🚀 Alien Rescue — Game Analytics

**A behavioral data analysis project from a game studio perspective.**

Analyzing 85,194 player log events from 159 users to uncover what drives success, why players fail, and how the game can be improved.

---

## 📁 Project Structure

```
alien-rescue-analytics/
├── data/
│   ├── Log_Raw.csv              # 85,194 raw in-game events
│   ├── Consoles.csv             # Tool open/close events
│   ├── Gates.csv                # Zone transition events
│   └── Duration_Charateristics.csv  # Player profiles & scores
├── notebooks/
│   └── analysis.py             # Full analysis pipeline
├── outputs/
│   └── figures/                # 14 generated charts
├── presentation.pdf            # Executive summary
└── README.md
```

---

## 🎯 Business Questions Answered

| # | Question | Method |
|---|---|---|
| 1 | How are players performing overall? | Distribution analysis |
| 2 | Which tools are used most — and do they help? | Adoption & correlation |
| 3 | Are there distinct player types? | K-Means clustering |
| 4 | Can we detect struggling players early? | Early warning system |
| 5 | Which tools need redesign? | Tool positioning matrix |
| 6 | How do players navigate between tools? | Transition matrix |

---

## 🔍 Dataset

**Source:** Liu, S. & Liu, M. (2019). *Data on player activity and characteristics in a Serious Game Environment.* Data in Brief. DOI: 10.1016/j.dib.2019.104965

**Game:** Alien Rescue — a Problem-Based Learning serious game where players must find suitable planets for 6 displaced alien species using scientific tools.

**Players:** 159 undergraduate students, ~1 hour gameplay session each.

| File | Rows | Description |
|---|---|---|
| Log_Raw | 85,194 | Every click, note, navigation action |
| Consoles | 5,916 | Tool open/close events |
| Gates | 4,460 | Zone door crossings |
| Duration_Characteristics | 159 | Per-player summary + psych scores |

---

## 📊 Key Findings

### Performance
- **33% of players scored 0** — indicating a serious onboarding/discoverability problem
- Average score: 3.0 / 7 — room for improvement across the board

### Player Segments (K-Means, k=3)

| Segment | Count | Avg Score | Signature Behavior |
|---|---|---|---|
| 🟢 Achievers | 36 | **3.97** | 2× more notes, 16% time in Notebook |
| 🔵 Explorers | 80 | 2.86 | Most passive, lowest actions |
| 🔴 Lost Players | 43 | 2.60 | Most actions (748!) but scattered |

### Early Warning
- First 20 minutes **cannot predict** final score alone (R² = -0.07)
- However, a 4-factor **risk score** separates Low Risk (avg 3.41) from High Risk (avg 1.88)
- Implication: the critical decision point happens *mid-game*, not at the start

### Tool Analysis

| Category | Tools | Recommendation |
|---|---|---|
| ⭐ Core | Mission Control, Probe Design, Notebook | Protect & surface early |
| 💎 Hidden Gem | Solar DB, Periodic Table, Concepts DB | Add to onboarding tutorial |
| ⚠️ Busy | Comm Center | 100% adoption but negative correlation — simplify content |
| 🔴 Dead Weight | Missions DB | Negative correlation — redesign or restructure |

---

## 🛠️ Tech Stack

- **Python 3.12**
- **pandas** — data wrangling
- **numpy** — numerical operations
- **matplotlib + seaborn** — visualization
- **scikit-learn** — K-Means, PCA, Random Forest, StandardScaler

---


## 📈 Visualizations Produced

| # | File | What it shows |
|---|---|---|
| 01 | performance_overview | Score distribution + success segments |
| 02 | tool_usage | Avg time + distribution per tool |
| 03 | notebook_vs_performance | Note-taking impact on score |
| 04 | activity_vs_performance | Actions & metacognition vs score |
| 05 | correlation_heatmap | All feature correlations |
| 06 | optimal_k | Elbow + Silhouette for clustering |
| 07 | segments_overview | PCA scatter + score by segment |
| 08 | segment_radar | Behavioral fingerprint per segment |
| 09 | segment_tool_heatmap | Tool usage patterns by segment |
| 10 | early_warning | First-20-min signals & activity over time |
| 11 | risk_scores | Risk group distribution & performance |
| 12 | tool_positioning_matrix | Adoption vs value 2×2 matrix |
| 13 | tool_transitions | Navigation flow between tools |
| 14 | segment_adoption | Tool adoption rate per segment |

---

*Dataset originally collected at University of Texas at Austin, Alien Rescue Research Team.*
