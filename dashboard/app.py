"""
dashboard/app.py
----------------
F1 Race Intelligence System - Web Dashboard (Flask)
F1 Pit Wall / Race Strategy temalı profesyonel dashboard.
Streamlit yerine Flask + vanilla JS kullanılmıştır (bağımlılık gerektirmez).
"""

import sys
import json
import time
from pathlib import Path

# Proje kökünü path'e ekle
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, render_template_string, jsonify, request
from config import (
    ALL_2025_RACES, DEFAULT_RACES, SEASON,
    PROCESSED_DATA_DIR, DASHBOARD_CONFIG
)
from src.database import get_all_processed_races, load_laps_from_db
from src.data_loader import load_or_fetch_race_data, is_race_processed
from src.analysis import run_all_analyses, compare_drivers
from src.models import run_all_models
from src.visualization import generate_all_charts
from src.report_generator import generate_race_report
from src.insight_engine import generate_race_insights
from src.cache_manager import load_analysis_cache, save_analysis_cache
from src.pdf_report import generate_pdf_report
from src.logger import get_logger

logger = get_logger("Dashboard")
app = Flask(__name__)

# ─────────────────────────────────────────────────────────────
# HTML ŞABLONU
# ─────────────────────────────────────────────────────────────


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>F1 Race Intelligence System</title>
<style>
/* ═══════════════════════════════════════════════════════════
   F1 RACE INTELLIGENCE — PREMIUM DARK THEME
   Formula 1 Pit Wall / Telemetry Center Aesthetic
═══════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
  /* Core palette */
  --bg:        #080808;
  --surface:   #0e0e0e;
  --card:      #121212;
  --card-hover:#161616;
  --border:    #1e1e1e;
  --border-bright: #2a2a2a;

  /* Brand */
  --primary:   #E8002D;
  --primary-dim: rgba(232,0,45,0.15);
  --primary-glow: rgba(232,0,45,0.35);
  --accent:    #FF3355;

  /* Text */
  --text:      #F0F0F0;
  --text-dim:  #888888;
  --text-muted:#444444;

  /* Semantic */
  --success:   #00D45A;
  --success-dim: rgba(0,212,90,0.15);
  --warning:   #FFB400;
  --warning-dim: rgba(255,180,0,0.15);
  --info:      #3B9EFF;
  --info-dim:  rgba(59,158,255,0.15);
  --purple:    #B47FFF;   /* fastest lap — F1 purple */

  /* Tire compounds */
  --soft:   #E8002D;
  --medium: #D4C000;
  --hard:   #C8C8C8;
  --inter:  #39B54A;
  --wet:    #0067FF;

  /* Sector colors */
  --sector-fastest: #B47FFF;
  --sector-good:    #00D45A;
  --sector-slow:    #E8002D;

  /* Typography */
  --font-ui:   'Inter', 'Segoe UI', system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Courier New', monospace;

  /* Spacing */
  --radius:    8px;
  --radius-sm: 4px;
  --radius-lg: 12px;

  /* Timing */
  --transition: 0.18s ease;
  --transition-slow: 0.35s ease;
}

/* ── RESET ───────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: var(--font-ui);
  font-size: 13px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
  overflow-x: hidden;
}

/* ── SCROLLBAR ───────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* ══════════════════════════════════════════════════
   HEADER — F1 BROADCAST BAR
══════════════════════════════════════════════════ */
.header {
  position: sticky; top: 0; z-index: 200;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  box-shadow: 0 1px 0 var(--border), 0 4px 24px rgba(0,0,0,0.6);
  padding: 0 24px;
  height: 56px;
  display: flex; align-items: center; justify-content: space-between;
}

/* Red stripe accent on header top */
.header::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--primary) 0%, #FF6B6B 50%, var(--primary) 100%);
  background-size: 200% 100%;
  animation: shimmer 3s linear infinite;
}
@keyframes shimmer {
  0%   { background-position: 200% center; }
  100% { background-position: -200% center; }
}

.header-left { display: flex; align-items: center; gap: 16px; }

.header-logo {
  display: flex; align-items: center; gap: 10px;
}
.logo-flag {
  width: 28px; height: 20px;
  background: var(--primary);
  display: grid; grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr;
  gap: 1px; border-radius: 2px; flex-shrink: 0;
}
.logo-flag span { background: var(--bg); }
.logo-flag span:nth-child(2),
.logo-flag span:nth-child(3) { background: var(--primary); }

.logo {
  font-family: var(--font-ui);
  font-size: 15px; font-weight: 900;
  letter-spacing: 0.08em;
  color: var(--text);
  text-transform: uppercase;
}
.logo em { color: var(--primary); font-style: normal; }

.subtitle {
  font-family: var(--font-mono);
  font-size: 9px; font-weight: 400;
  color: var(--text-dim); letter-spacing: 0.2em;
  text-transform: uppercase; margin-top: 1px;
}

/* Header status bar */
.status-bar { display: flex; align-items: center; gap: 10px; }

.status-pill {
  display: flex; align-items: center; gap: 6px;
  background: var(--card); border: 1px solid var(--border);
  padding: 5px 12px; border-radius: 20px;
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); letter-spacing: 0.05em;
  transition: var(--transition);
}
.status-pill.live {
  border-color: var(--primary-glow);
  color: var(--primary);
  box-shadow: 0 0 12px var(--primary-glow);
}
.status-pill.live::before {
  content: '';
  width: 6px; height: 6px;
  background: var(--primary);
  border-radius: 50%;
  animation: pulse-dot 1.5s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(0.7); }
}

/* Data/sim badge */
.data-badge {
  font-family: var(--font-mono); font-size: 9px;
  padding: 3px 10px; border-radius: 20px;
  font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
}
.data-badge.real {
  background: var(--success-dim); color: var(--success);
  border: 1px solid var(--success); box-shadow: 0 0 8px rgba(0,212,90,0.2);
}
.data-badge.simulated {
  background: var(--warning-dim); color: var(--warning);
  border: 1px solid rgba(255,180,0,0.3);
}

/* ══════════════════════════════════════════════════
   LAYOUT
══════════════════════════════════════════════════ */
.layout { display: flex; min-height: calc(100vh - 56px); }

/* ══════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════ */
.sidebar {
  width: 220px; flex-shrink: 0;
  background: var(--surface);
  border-right: 1px solid var(--border);
  padding: 20px 14px;
  position: sticky; top: 56px;
  height: calc(100vh - 56px);
  overflow-y: auto;
  display: flex; flex-direction: column; gap: 0;
}

.sidebar-section { margin-bottom: 24px; }

.sidebar-label {
  font-family: var(--font-mono);
  font-size: 9px; font-weight: 600;
  color: var(--text-muted); letter-spacing: 0.25em;
  text-transform: uppercase;
  margin-bottom: 10px;
  display: flex; align-items: center; gap: 6px;
}
.sidebar-label::after {
  content: '';
  flex: 1; height: 1px;
  background: var(--border);
}

/* Select */
select {
  width: 100%; padding: 8px 10px;
  background: var(--card); border: 1px solid var(--border);
  color: var(--text); font-family: var(--font-mono); font-size: 11px;
  border-radius: var(--radius-sm); cursor: pointer;
  transition: var(--transition); outline: none;
  appearance: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23666'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: right 10px center;
  padding-right: 28px;
}
select:hover { border-color: var(--border-bright); }
select:focus { border-color: var(--primary); box-shadow: 0 0 0 2px var(--primary-dim); }
select + select { margin-top: 6px; }

/* Buttons */
.btn {
  width: 100%; padding: 8px 12px;
  background: var(--card); border: 1px solid var(--border);
  color: var(--text); font-family: var(--font-mono);
  font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
  border-radius: var(--radius-sm); cursor: pointer;
  transition: var(--transition); text-transform: uppercase;
  display: flex; align-items: center; justify-content: center; gap: 6px;
}
.btn:hover {
  background: var(--card-hover); border-color: var(--border-bright);
  transform: translateY(-1px);
}
.btn:active { transform: translateY(0); }
.btn + .btn { margin-top: 6px; }

.btn-primary {
  background: var(--primary); border-color: var(--primary);
  color: #fff; box-shadow: 0 2px 12px var(--primary-dim);
}
.btn-primary:hover {
  background: var(--accent); border-color: var(--accent);
  box-shadow: 0 4px 20px var(--primary-glow);
  transform: translateY(-1px);
}

/* Processed list */
.processed-item {
  display: flex; align-items: center; gap: 6px;
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); padding: 3px 0;
  border-bottom: 1px solid var(--border);
}
.processed-item::before {
  content: ''; width: 5px; height: 5px;
  background: var(--success); border-radius: 50%; flex-shrink: 0;
}

/* System info */
.sys-info {
  font-family: var(--font-mono); font-size: 9px;
  color: var(--text-muted); line-height: 2;
}
.sys-info span { color: var(--text-dim); }

/* ══════════════════════════════════════════════════
   MAIN CONTENT
══════════════════════════════════════════════════ */
.main { flex: 1; padding: 20px 24px; overflow-x: hidden; min-width: 0; }

/* ══════════════════════════════════════════════════
   TAB BAR — F1 TIMING TOWER STYLE
══════════════════════════════════════════════════ */
.tab-bar {
  display: flex; gap: 2px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px; flex-wrap: wrap;
  position: relative;
}

.tab {
  position: relative; overflow: hidden;
  padding: 10px 16px;
  background: transparent; border: none;
  color: var(--text-dim); font-family: var(--font-mono);
  font-size: 10px; font-weight: 600; letter-spacing: 0.1em;
  cursor: pointer; text-transform: uppercase;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color var(--transition);
  white-space: nowrap;
}
.tab::before {
  content: '';
  position: absolute; bottom: 0; left: 50%; right: 50%;
  height: 2px; background: var(--primary);
  transition: left var(--transition), right var(--transition);
}
.tab:hover { color: var(--text); }
.tab:hover::before { left: 0; right: 0; }

.tab.active { color: var(--primary); }
.tab.active::before { left: 0; right: 0; }

/* Tab ripple on click */
.tab::after {
  content: '';
  position: absolute; inset: 0;
  background: var(--primary-dim);
  opacity: 0;
  transition: opacity 0.1s;
}
.tab:active::after { opacity: 1; }

.tab-content { display: none; animation: fadeIn 0.2s ease; }
.tab-content.active { display: block; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

/* ══════════════════════════════════════════════════
   KPI METRIC CARDS
══════════════════════════════════════════════════ */
.metrics-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(155px, 1fr));
  gap: 12px; margin-bottom: 20px;
}

.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 16px;
  transition: border-color var(--transition), box-shadow var(--transition);
  position: relative; overflow: hidden;
}
.card:hover {
  border-color: var(--border-bright);
}

/* Metric card variant */
.metric-card {
  padding: 14px 16px; margin-bottom: 0;
  cursor: default;
}
.metric-card:hover {
  border-color: var(--border-bright);
  box-shadow: 0 4px 20px rgba(0,0,0,0.4);
  transform: translateY(-1px);
}
.metric-card.accent {
  border-color: rgba(232,0,45,0.3);
  box-shadow: 0 0 0 1px rgba(232,0,45,0.1);
}
.metric-card.accent:hover {
  border-color: var(--primary);
  box-shadow: 0 0 20px var(--primary-glow), 0 4px 20px rgba(0,0,0,0.4);
}
/* Subtle top border gradient on accent cards */
.metric-card.accent::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--primary), transparent);
}

.card-icon {
  font-size: 16px; margin-bottom: 6px; display: block;
  opacity: 0.7;
}
.card-label {
  font-family: var(--font-mono); font-size: 9px; font-weight: 600;
  color: var(--text-muted); letter-spacing: 0.2em; text-transform: uppercase;
  margin-bottom: 6px;
}
.card-value {
  font-family: var(--font-mono); font-size: 22px; font-weight: 700;
  color: var(--text); letter-spacing: -0.02em; line-height: 1.1;
}
.metric-card.accent .card-value { color: var(--primary); }
.card-sub {
  font-family: var(--font-mono); font-size: 10px; color: var(--text-dim);
  margin-top: 4px; font-weight: 400;
}

/* Section card */
.card-title {
  font-family: var(--font-mono); font-size: 9px; font-weight: 700;
  color: var(--text-dim); letter-spacing: 0.2em; text-transform: uppercase;
  margin-bottom: 14px; padding-bottom: 10px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 8px;
}
.card-title::before {
  content: '';
  display: inline-block; width: 3px; height: 12px;
  background: var(--primary); border-radius: 2px; flex-shrink: 0;
}

/* ══════════════════════════════════════════════════
   CHART CONTAINERS
══════════════════════════════════════════════════ */
.chart-container {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 16px;
  transition: border-color var(--transition);
}
.chart-container:hover { border-color: var(--border-bright); }

.chart-title {
  font-family: var(--font-mono); font-size: 9px; font-weight: 700;
  color: var(--text-dim); letter-spacing: 0.2em; text-transform: uppercase;
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px;
}
.chart-title::before {
  content: '';
  display: inline-block; width: 3px; height: 12px;
  background: var(--primary); border-radius: 2px; flex-shrink: 0;
}

.chart-container img {
  width: 100%; border-radius: var(--radius-sm);
  display: block;
}

.charts-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
}

.chart-empty {
  display: flex; align-items: center; justify-content: center;
  min-height: 80px; color: var(--text-muted);
  font-family: var(--font-mono); font-size: 10px; letter-spacing: 0.1em;
}

/* ══════════════════════════════════════════════════
   TABLE
══════════════════════════════════════════════════ */
.table-wrap { overflow-x: auto; border-radius: var(--radius-sm); }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
thead { position: sticky; top: 0; z-index: 1; }
th {
  background: #0a0a0a;
  color: var(--text-muted); font-family: var(--font-mono);
  font-size: 9px; font-weight: 700; letter-spacing: 0.2em;
  text-transform: uppercase; padding: 10px 12px;
  text-align: left; border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
td {
  padding: 9px 12px;
  border-bottom: 1px solid var(--border);
  color: var(--text-dim); font-family: var(--font-mono);
  font-size: 11px; transition: background var(--transition);
  white-space: nowrap;
}
tr:last-child td { border-bottom: none; }
tr:hover td { background: rgba(255,255,255,0.02); color: var(--text); }

/* Rank highlights */
.rank-1 td { color: var(--primary); }
.rank-1 td:first-child { font-weight: 700; }
.rank-2 td { color: var(--text); }
.rank-3 td { color: var(--text-dim); }

/* Badges */
.badge {
  display: inline-flex; align-items: center;
  padding: 2px 9px; border-radius: 20px; font-size: 9px;
  font-family: var(--font-mono); font-weight: 700;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-soft     { background: rgba(232,0,45,0.15);  color: var(--soft);   border: 1px solid rgba(232,0,45,0.3); }
.badge-medium   { background: rgba(212,192,0,0.15); color: var(--medium); border: 1px solid rgba(212,192,0,0.3); }
.badge-hard     { background: rgba(200,200,200,0.1);color: var(--hard);   border: 1px solid rgba(200,200,200,0.2); }
.badge-intermediate { background: rgba(57,181,74,0.15); color: var(--inter); border: 1px solid rgba(57,181,74,0.3); }
.badge-wet      { background: rgba(0,103,255,0.15); color: var(--wet);    border: 1px solid rgba(0,103,255,0.3); }

/* ══════════════════════════════════════════════════
   COMMENT / ALERT BOX
══════════════════════════════════════════════════ */
.comment {
  background: rgba(232,0,45,0.06);
  border: 1px solid rgba(232,0,45,0.2);
  border-left: 3px solid var(--primary);
  padding: 10px 14px; margin: 12px 0;
  font-size: 11px; color: var(--text-dim);
  border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
  line-height: 1.6;
}

.alert {
  padding: 10px 14px; border-radius: var(--radius-sm);
  margin-bottom: 16px; font-size: 11px;
  font-family: var(--font-mono); line-height: 1.6;
  display: flex; align-items: flex-start; gap: 8px;
}
.alert-info    { background: var(--info-dim);    border: 1px solid rgba(59,158,255,0.3);  color: #90CAF9; }
.alert-warning { background: var(--warning-dim); border: 1px solid rgba(255,180,0,0.3);   color: #FFE082; }
.alert-success { background: var(--success-dim); border: 1px solid rgba(0,212,90,0.3);    color: #69F0AE; }

/* ══════════════════════════════════════════════════
   LOADING
══════════════════════════════════════════════════ */
.loading {
  display: flex; align-items: center; justify-content: center;
  min-height: 300px; flex-direction: column; gap: 20px;
}
.spinner {
  width: 36px; height: 36px;
  border: 2px solid var(--border);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-text {
  font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); letter-spacing: 0.3em; text-transform: uppercase;
}
/* Loading dots */
.loading-text::after {
  content: '';
  animation: dots 1.5s steps(4, end) infinite;
}
@keyframes dots {
  0%   { content: ''; }
  25%  { content: '.'; }
  50%  { content: '..'; }
  75%  { content: '...'; }
  100% { content: ''; }
}

/* ══════════════════════════════════════════════════
   K-MEANS CLUSTER CARDS
══════════════════════════════════════════════════ */
.cluster-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}
.cluster-card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px;
  transition: var(--transition);
}
.cluster-card:hover { border-color: var(--border-bright); transform: translateY(-1px); }
.cluster-stable   { border-top: 2px solid var(--success); }
.cluster-agresif  { border-top: 2px solid var(--warning); }
.cluster-duzensiz { border-top: 2px solid var(--text-muted); }
.cluster-name {
  font-family: var(--font-mono); font-size: 10px; font-weight: 700;
  letter-spacing: 0.1em; margin-bottom: 10px; text-transform: uppercase;
}
.cluster-stable   .cluster-name { color: var(--success); }
.cluster-agresif  .cluster-name { color: var(--warning); }
.cluster-duzensiz .cluster-name { color: var(--text-dim); }
.cluster-member {
  font-family: var(--font-mono); font-size: 10px; color: var(--text-dim);
  padding: 3px 0; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 5px;
}
.cluster-member:last-child { border-bottom: none; }
.cluster-member::before { content: '·'; color: var(--text-muted); }

/* ══════════════════════════════════════════════════
   FEATURE IMPORTANCE BARS
══════════════════════════════════════════════════ */
.feat-row {
  display: flex; align-items: center; gap: 10px;
  padding: 6px 0; border-bottom: 1px solid var(--border);
}
.feat-row:last-child { border-bottom: none; }
.feat-name {
  width: 200px; font-family: var(--font-mono); font-size: 10px;
  color: var(--text-dim); flex-shrink: 0; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}
.feat-bar-bg {
  flex: 1; height: 6px; background: var(--border); border-radius: 3px;
  overflow: hidden;
}
.feat-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--primary), #FF6B6B);
  border-radius: 3px;
  transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.feat-val {
  width: 46px; text-align: right;
  font-family: var(--font-mono); font-size: 10px; color: var(--text);
  flex-shrink: 0;
}

/* ══════════════════════════════════════════════════
   MODEL COMPARISON STATS
══════════════════════════════════════════════════ */
.model-stats {
  display: flex; gap: 24px; margin-bottom: 16px; flex-wrap: wrap;
}
.model-stat { text-align: center; }
.model-stat-label {
  font-family: var(--font-mono); font-size: 9px; color: var(--text-muted);
  letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 4px;
}
.model-stat-value {
  font-family: var(--font-mono); font-size: 28px; font-weight: 700;
  line-height: 1;
}
.model-stat-value.dt  { color: var(--primary); }
.model-stat-value.knn { color: var(--info); }
.model-stat-value.win { color: var(--success); font-size: 16px; }
.model-stat-comment {
  font-family: var(--font-mono); font-size: 10px; color: var(--text-dim);
  margin-top: 6px; line-height: 1.5;
}

/* ══════════════════════════════════════════════════
   REPORT BOX
══════════════════════════════════════════════════ */
.report-box {
  background: #060606;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px; font-family: var(--font-mono);
  font-size: 11px; line-height: 1.75;
  white-space: pre-wrap; color: var(--text-dim);
  max-height: 600px; overflow-y: auto;
}
.report-actions {
  display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap;
}
.report-actions .btn { width: auto; padding: 8px 20px; }

/* ══════════════════════════════════════════════════
   PRESENTATION MODE
══════════════════════════════════════════════════ */
.pres-header {
  text-align: center; padding: 24px 0 20px;
  border-bottom: 1px solid var(--border); margin-bottom: 28px;
}
.pres-race-title {
  font-family: var(--font-ui); font-size: 36px; font-weight: 900;
  color: var(--text); letter-spacing: -0.02em; line-height: 1;
}
.pres-race-title em { color: var(--primary); font-style: normal; }
.pres-season {
  font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);
  letter-spacing: 0.3em; text-transform: uppercase; margin-top: 6px;
}

.pres-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 16px; margin-bottom: 28px;
}
.pres-insight {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 20px 16px;
  text-align: center;
  position: relative; overflow: hidden;
  transition: var(--transition);
}
.pres-insight:hover {
  border-color: rgba(232,0,45,0.4);
  box-shadow: 0 0 24px rgba(232,0,45,0.12);
  transform: translateY(-2px);
}
.pres-insight::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: var(--primary);
}
.pres-number {
  font-family: var(--font-mono); font-size: 28px; font-weight: 900;
  color: var(--primary); line-height: 1; margin: 6px 0;
  text-shadow: 0 0 20px var(--primary-glow);
}
.pres-label {
  font-family: var(--font-mono); font-size: 9px; color: var(--text-muted);
  letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 6px;
}
.pres-text {
  font-size: 11px; color: var(--text-dim); line-height: 1.5; margin-top: 6px;
}

/* ══════════════════════════════════════════════════
   SECTOR COLORS (F1 timing style)
══════════════════════════════════════════════════ */
.sector-fastest { color: var(--sector-fastest); font-weight: 700; }
.sector-good    { color: var(--sector-good); }
.sector-slow    { color: var(--sector-slow); }

/* Correlation colors */
.corr-positive { color: var(--success); font-weight: 600; }
.corr-negative { color: var(--accent); font-weight: 600; }
.corr-neutral  { color: var(--text-dim); }

/* ══════════════════════════════════════════════════
   RESPONSIVE
══════════════════════════════════════════════════ */
@media (max-width: 1024px) {
  .charts-grid { grid-template-columns: 1fr; }
  .pres-grid   { grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); }
}
@media (max-width: 768px) {
  .sidebar { display: none; }
  .main    { padding: 16px; }
  .header  { padding: 0 16px; }
  .tab     { padding: 8px 10px; font-size: 9px; }
  .metrics-row { grid-template-columns: repeat(2, 1fr); }
}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════
     HEADER
══════════════════════════════════════════════ -->
<div class="header">
  <div class="header-left">
    <div class="header-logo">
      <div class="logo-flag">
        <span></span><span></span><span></span><span></span>
      </div>
      <div>
        <div class="logo">F1 <em>Race Intelligence</em></div>
        <div class="subtitle">Strategy Analysis System &mdash; {{ season }} Season</div>
      </div>
    </div>
  </div>

  <div class="status-bar">
    <div class="status-pill live" id="race-indicator">⚡ LOADING</div>
    <div class="status-pill" id="data-pill">📊 — tur</div>
    <div class="data-badge simulated" id="data-source-badge">● SIMULATED</div>
  </div>
</div>

<div class="layout">

<!-- ══════════════════════════════════════════════
     SIDEBAR
══════════════════════════════════════════════ -->
<div class="sidebar">

  <div class="sidebar-section">
    <div class="sidebar-label">Grand Prix</div>
    <select id="race-select" onchange="loadRace()">
      {% for race in all_races %}
      <option value="{{ race }}" {% if race == default_race %}selected{% endif %}>
        {{ race }}{% if race in processed_races %} ✓{% endif %}
      </option>
      {% endfor %}
    </select>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">Driver Compare</div>
    <select id="driver1-select">
      <option value="">— Driver 1 —</option>
    </select>
    <select id="driver2-select">
      <option value="">— Driver 2 —</option>
    </select>
    <button class="btn btn-primary" style="margin-top:8px" onclick="compareDrivers()">
      ⚔ KARŞILAŞTIR
    </button>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">Tools</div>
    <button class="btn" onclick="generateReport()">📄 Rapor Üret</button>
    <button class="btn" onclick="downloadPDF()">📊 PDF İndir</button>
    <button class="btn" onclick="exportCSV()">💾 CSV Export</button>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">Ready Races</div>
    <div id="processed-list">
      <div class="processed-item" style="opacity:.4">Yükleniyor...</div>
    </div>
  </div>

  <div class="sidebar-section" style="margin-top:auto;">
    <div class="sidebar-label">System</div>
    <div class="sys-info">
      Season <span>{{ season }}</span><br>
      Races <span>{{ all_races|length }} GP</span><br>
      Engine <span>FastF1 + SQLite</span><br>
      Cache <span>✓ Active</span><br>
      Lazy Load <span>✓ Active</span>
    </div>
  </div>

</div>

<!-- ══════════════════════════════════════════════
     MAIN CONTENT
══════════════════════════════════════════════ -->
<div class="main">

  <!-- TAB BAR -->
  <div class="tab-bar">
    <button class="tab active" onclick="showTab('overview')">🏠 Overview</button>
    <button class="tab" onclick="showTab('driver')">👤 Driver Perf.</button>
    <button class="tab" onclick="showTab('tire')">🔴 Tire Strategy</button>
    <button class="tab" onclick="showTab('pit')">🔧 Pit Stop</button>
    <button class="tab" onclick="showTab('weather')">🌡️ Weather</button>
    <button class="tab" onclick="showTab('ml')">🤖 ML Models</button>
    <button class="tab" onclick="showTab('report')">📄 Report</button>
    <button class="tab" onclick="showTab('presentation')">🎯 Sunum</button>
  </div>

  <!-- GLOBAL LOADING -->
  <div id="global-loading" style="display:none;">
    <div class="loading">
      <div class="spinner"></div>
      <div class="loading-text" id="loading-msg">Veri Yükleniyor</div>
    </div>
  </div>

  <!-- ═══ OVERVIEW ═══ -->
  <div id="tab-overview" class="tab-content active">
    <div id="overview-alert"></div>

    <div class="metrics-row">
      <div class="card metric-card accent">
        <span class="card-icon">🏎️</span>
        <div class="card-label">Grand Prix</div>
        <div class="card-value" id="ov-race">—</div>
        <div class="card-sub">{{ season }} Sezonu</div>
      </div>
      <div class="card metric-card">
        <span class="card-icon">👤</span>
        <div class="card-label">Pilots</div>
        <div class="card-value" id="ov-drivers">—</div>
        <div class="card-sub">toplam pilot</div>
      </div>
      <div class="card metric-card">
        <span class="card-icon">🔄</span>
        <div class="card-label">Laps</div>
        <div class="card-value" id="ov-laps">—</div>
        <div class="card-sub">toplam tur</div>
      </div>
      <div class="card metric-card accent">
        <span class="card-icon">⚡</span>
        <div class="card-label">Fastest Lap</div>
        <div class="card-value" id="ov-fastest-time">—</div>
        <div class="card-sub" id="ov-fastest-driver">—</div>
      </div>
      <div class="card metric-card">
        <span class="card-icon">🎯</span>
        <div class="card-label">Most Consistent</div>
        <div class="card-value" id="ov-stable">—</div>
        <div class="card-sub">consistency</div>
      </div>
      <div class="card metric-card">
        <span class="card-icon">🌡️</span>
        <div class="card-label">Temperature</div>
        <div class="card-value" id="ov-temp">—</div>
        <div class="card-sub">air / track (°C)</div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="chart-container">
        <div class="chart-title">Ortalama Tur Süresi</div>
        <div id="chart-driver-avg"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
      <div class="chart-container">
        <div class="chart-title">Consistency Score</div>
        <div id="chart-consistency"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
    </div>
  </div>

  <!-- ═══ DRIVER PERFORMANCE ═══ -->
  <div id="tab-driver" class="tab-content">
    <div class="card">
      <div class="card-title">Pilot İstikrar Sıralaması</div>
      <div class="table-wrap">
        <table id="stability-table">
          <thead><tr>
            <th>Sıra</th><th>Pilot</th><th>Takım</th>
            <th>Ort. Tur (s)</th><th>En Hızlı (s)</th>
            <th>Std Sapma</th><th>Consistency</th>
          </tr></thead>
          <tbody id="stability-tbody"></tbody>
        </table>
      </div>
    </div>
    <div id="driver-comment" class="comment" style="display:none;"></div>
    <div class="chart-container">
      <div class="chart-title">Pilot Karşılaştırma</div>
      <div id="chart-driver-compare"><div class="chart-empty">Sidebar'dan iki pilot seçin →</div></div>
      <div id="compare-comment" class="comment" style="display:none;margin-top:10px;"></div>
    </div>
  </div>

  <!-- ═══ TIRE STRATEGY ═══ -->
  <div id="tab-tire" class="tab-content">
    <div class="card">
      <div class="card-title">Lastik Tipi Karşılaştırması</div>
      <div class="table-wrap">
        <table id="compound-table">
          <thead><tr>
            <th>Lastik</th><th>Ort. Tur (s)</th><th>En Hızlı (s)</th>
            <th>Std Sapma</th><th>Tur Sayısı</th>
          </tr></thead>
          <tbody id="compound-tbody"></tbody>
        </table>
      </div>
    </div>
    <div id="tire-comment" class="comment" style="display:none;"></div>

    <div class="charts-grid">
      <div class="chart-container">
        <div class="chart-title">Compound Ort. Tur Süresi</div>
        <div id="chart-compound"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
      <div class="chart-container">
        <div class="chart-title">Lastik Bozulma Eğrisi</div>
        <div id="chart-degradation"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
    </div>

    <div class="chart-container">
      <div class="chart-title">Lastik Yaşı vs Tur Zamanı</div>
      <div id="chart-tyre-scatter"><div class="chart-empty">Yükleniyor...</div></div>
    </div>

    <div class="card">
      <div class="card-title">Lastik Bozulma Analizi</div>
      <div class="table-wrap">
        <table id="degradation-table">
          <thead><tr>
            <th>Lastik</th><th>Bozulma (s/tur)</th><th>R²</th><th>Yorum</th>
          </tr></thead>
          <tbody id="degradation-tbody"></tbody>
        </table>
      </div>
    </div>
    <div id="deg-comment" class="comment" style="display:none;"></div>
  </div>

  <!-- ═══ PIT STOP ═══ -->
  <div id="tab-pit" class="tab-content">
    <div class="metrics-row" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">
      <div class="card metric-card accent">
        <span class="card-icon">🔧</span>
        <div class="card-label">Ort. Pit Stop Faydası</div>
        <div class="card-value" id="pit-avg-impact">—</div>
        <div class="card-sub">pozitif = pit sonrası hızlandı</div>
      </div>
    </div>
    <div class="chart-container">
      <div class="chart-title">Pit Stop Etkisi (Öncesi / Sonrası)</div>
      <div id="chart-pit"><div class="chart-empty">Yükleniyor...</div></div>
    </div>
    <div class="card">
      <div class="card-title">Pilot Bazlı Pit Stop Analizi</div>
      <div class="table-wrap">
        <table id="pit-table">
          <thead><tr>
            <th>Pilot</th><th>Pit Sayısı</th><th>Ort. Etki (s)</th><th>Değerlendirme</th>
          </tr></thead>
          <tbody id="pit-tbody"></tbody>
        </table>
      </div>
    </div>
    <div id="pit-comment" class="comment" style="display:none;"></div>
  </div>

  <!-- ═══ WEATHER ═══ -->
  <div id="tab-weather" class="tab-content">
    <div class="charts-grid">
      <div class="chart-container">
        <div class="chart-title">Sıcaklık vs Tur Zamanı</div>
        <div id="chart-temp"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
      <div class="chart-container">
        <div class="chart-title">Korelasyon Matrisi</div>
        <div id="chart-corr"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Pearson Korelasyon Analizi</div>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th>Değişken</th><th>Pearson r</th><th>p-Değeri</th>
            <th>Anlamlı?</th><th>Yorum</th>
          </tr></thead>
          <tbody id="weather-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- ═══ ML MODELS ═══ -->
  <div id="tab-ml" class="tab-content">
    <div class="alert alert-info">
      🤖 Makine öğrenmesi modelleri pilot performans verisi üzerinde çalıştırılmıştır.
    </div>

    <div class="card">
      <div class="card-title">K-Means Clustering — Performans Kümeleri</div>
      <div id="kmeans-clusters"></div>
    </div>

    <div class="charts-grid">
      <div class="chart-container">
        <div class="chart-title">Cluster Görselleştirmesi (PCA)</div>
        <div id="chart-kmeans"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
      <div class="chart-container">
        <div class="chart-title">Elbow Method (Optimal k)</div>
        <div id="chart-elbow"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
    </div>

    <div class="charts-grid">
      <div class="card">
        <div class="card-title">Model Karşılaştırması</div>
        <div id="model-comparison-stats"></div>
        <div id="chart-model-compare"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
      <div class="card">
        <div class="card-title">Feature Importance (Decision Tree)</div>
        <div id="feature-importance-bars"></div>
        <div id="chart-feature-imp"><div class="chart-empty">Yükleniyor...</div></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">🔑 En Önemli Performans Faktörü</div>
      <div id="most-important-feature"
           style="font-family:var(--font-mono);font-size:20px;color:var(--primary);
                  padding:12px 0;font-weight:700;letter-spacing:0.05em;">
        Analiz yapılıyor...
      </div>
      <div id="ml-comment" class="comment"></div>
    </div>
  </div>

  <!-- ═══ REPORT ═══ -->
  <div id="tab-report" class="tab-content">
    <div class="report-actions">
      <button class="btn btn-primary" onclick="generateReport()">📄 Rapor Üret</button>
      <button class="btn" onclick="downloadPDF()">📊 PDF İndir</button>
      <button class="btn" onclick="exportCSV()">💾 CSV Export</button>
    </div>
    <div id="report-loading" style="display:none;">
      <div class="loading">
        <div class="spinner"></div>
        <div class="loading-text">Rapor Üretiliyor</div>
      </div>
    </div>
    <div id="report-content" class="report-box">
      Rapor üretmek için "RAPOR ÜRET" butonuna tıklayın.
    </div>
  </div>

  <!-- ═══ PRESENTATION MODE ═══ -->
  <div id="tab-presentation" class="tab-content">
    <div class="pres-header">
      <div class="pres-race-title" id="pres-race-name">
        <em>—</em> Grand Prix
      </div>
      <div class="pres-season">Formula 1 · {{ season }} Season · Race Intelligence Analysis</div>
    </div>
    <div class="pres-grid" id="pres-insights"></div>
    <div class="chart-container" style="margin-top:8px;">
      <div class="chart-title">Key Performance Chart</div>
      <div id="pres-chart"><div class="chart-empty">Yükleniyor...</div></div>
    </div>
  </div>

</div><!-- /main -->
</div><!-- /layout -->

<script>
// ─────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────
let currentRace = "{{ default_race }}";
let currentData = null;

// ─────────────────────────────────────────────────────────────
// TAB YÖNETİMİ
// ─────────────────────────────────────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}

// ─────────────────────────────────────────────────────────────
// YARıŞ YÜKLE
// ─────────────────────────────────────────────────────────────
function loadRace() {
  currentRace = document.getElementById('race-select').value;
  setLoading(true, currentRace + ' GP verisi yükleniyor...');

  fetch('/api/race/' + encodeURIComponent(currentRace))
    .then(r => r.json())
    .then(data => {
      currentData = data;
      setLoading(false);
      renderAll(data);
    })
    .catch(err => {
      setLoading(false);
      showAlert('Veri yüklenemedi: ' + err, 'warning');
    });
}

function setLoading(show, msg) {
  document.getElementById('global-loading').style.display = show ? 'flex' : 'none';
  document.querySelectorAll('.tab-content').forEach(el => {
    el.style.opacity = show ? '0.3' : '1';
    el.style.pointerEvents = show ? 'none' : 'auto';
  });
  if (msg) document.getElementById('loading-msg').textContent = msg.toUpperCase();
  document.getElementById('race-indicator').textContent = show ? '⏳ ' + currentRace : '⚡ ' + currentRace + ' GP';
}

// ─────────────────────────────────────────────────────────────
// TÜM VERİYİ RENDER ET
// ─────────────────────────────────────────────────────────────
function renderAll(data) {
  if (data.error) {
    showAlert('⚠️ ' + data.error, 'warning');
    return;
  }

  renderOverview(data);
  renderDrivers(data);
  renderTires(data);
  renderPit(data);
  renderWeather(data);
  renderML(data);
  renderInsights(data);
  renderPresentation(data);
  updateDriverSelects(data);
  updateProcessedList(data.processed_races || []);
  if (data.is_simulated) {
    showAlert('⚠️ SİMÜLE VERİ: Gerçek FastF1 için internet + pip install fastf1 gereklidir.', 'warning');
  }

  document.getElementById('data-pill').textContent =
    '📊 ' + (data.overview?.total_laps || '–') + ' tur';
}

// ─────────────────────────────────────────────────────────────
// OVERVIEW
// ─────────────────────────────────────────────────────────────
function renderOverview(data) {
  const ov = data.overview || {};
  setText('ov-race', ov.race_name || currentRace);
  setText('ov-drivers', ov.total_drivers ?? '—');
  setText('ov-laps', ov.total_laps ?? '—');

  const fl = ov.fastest_lap;
  setText('ov-fastest-time', fl ? formatTime(fl) : '—');
  setText('ov-fastest-driver', ov.fastest_driver || '—');
  setText('ov-stable', ov.most_stable_driver || '—');
  setText('ov-temp',
    (ov.avg_air_temp?.toFixed(1) ?? '–') + ' / ' +
    (ov.avg_track_temp?.toFixed(1) ?? '–'));

  if (data.first_load) {
    showAlert('⏳ Bu yarış ilk kez yükleniyor — veriler işlendi.', 'info', 'overview-alert');
  }

  // Data source badge
  const badge = document.getElementById('data-source-badge');
  if (badge) {
    if (data.is_simulated) {
      badge.textContent = '● SIMULATED';
      badge.className = 'data-badge simulated';
    } else {
      badge.textContent = '● REAL FastF1';
      badge.className = 'data-badge real';
    }
  }

  setChart('chart-driver-avg', data.charts?.driver_avg_laptime);
  setChart('chart-consistency', data.charts?.consistency_score);
}

// ─────────────────────────────────────────────────────────────
// DRIVER PERFORMANCE
// ─────────────────────────────────────────────────────────────
function renderDrivers(data) {
  const stab = data.analysis?.driver_stability || {};
  const rows = stab.data || [];
  const tbody = document.getElementById('stability-tbody');
  tbody.innerHTML = '';
  rows.forEach((r, i) => {
    const cls = i === 0 ? 'rank-1' : (i === 1 ? 'rank-2' : (i === 2 ? 'rank-3' : ''));
    tbody.innerHTML += `<tr class="${cls}">
      <td>${r.stability_rank ?? i+1}</td>
      <td>${r.Driver}</td>
      <td>${r.Team || '—'}</td>
      <td>${(r.avg_lap_time||0).toFixed(3)}</td>
      <td>${(r.best_lap_time||0).toFixed(3)}</td>
      <td>${(r.lap_time_std||0).toFixed(3)}</td>
      <td>${(r.consistency_score||0).toFixed(4)}</td>
    </tr>`;
  });
  if (stab.comment) {
    showEl('driver-comment', stab.comment);
  }
}

// ─────────────────────────────────────────────────────────────
// TIRES
// ─────────────────────────────────────────────────────────────
function renderTires(data) {
  const tire = data.analysis?.tire_strategy || {};
  const tbody = document.getElementById('compound-tbody');
  tbody.innerHTML = '';
  (tire.compound_stats || []).forEach(r => {
    const badge = `<span class="badge badge-${r.Compound?.toLowerCase()}">${r.Compound}</span>`;
    tbody.innerHTML += `<tr>
      <td>${badge}</td>
      <td>${(r.avg_lap_time||0).toFixed(3)}</td>
      <td>${(r.best_lap_time||0).toFixed(3)}</td>
      <td>${(r.std||0).toFixed(3)}</td>
      <td>${parseInt(r.count||0)}</td>
    </tr>`;
  });
  if (tire.comment) showEl('tire-comment', tire.comment);

  // Degradation table
  const deg = data.analysis?.tire_degradation || {};
  const degTbody = document.getElementById('degradation-tbody');
  degTbody.innerHTML = '';
  Object.entries(deg.by_compound || {}).forEach(([compound, info]) => {
    const rate = info.degradation_rate;
    degTbody.innerHTML += `<tr>
      <td><span class="badge badge-${compound.toLowerCase()}">${compound}</span></td>
      <td>${rate != null ? rate.toFixed(4) : 'N/A'}</td>
      <td>${info.r_squared != null ? info.r_squared.toFixed(3) : 'N/A'}</td>
      <td style="font-size:10px;color:var(--text-dim);">${info.comment || ''}</td>
    </tr>`;
  });
  if (deg.comment) showEl('deg-comment', deg.comment);

  setChart('chart-compound', data.charts?.compound_avg_laptime);
  setChart('chart-degradation', data.charts?.tire_degradation);
  setChart('chart-tyre-scatter', data.charts?.tyre_life_scatter);
}

// ─────────────────────────────────────────────────────────────
// PIT STOP
// ─────────────────────────────────────────────────────────────
function renderPit(data) {
  const pit = data.analysis?.pit_stop || {};
  const impact = pit.race_avg_impact;
  setText('pit-avg-impact',
    impact != null ? (impact >= 0 ? '+' : '') + impact.toFixed(3) + 's' : '—');

  const tbody = document.getElementById('pit-tbody');
  tbody.innerHTML = '';
  (pit.driver_analysis || []).forEach(d => {
    const impact_s = d.avg_impact >= 0 ? '+' + d.avg_impact.toFixed(3) : d.avg_impact.toFixed(3);
    const color = d.avg_impact > 0 ? 'var(--success)' : 'var(--accent)';
    tbody.innerHTML += `<tr>
      <td>${d.Driver}</td>
      <td>${d.pit_count}</td>
      <td style="color:${color};font-weight:600;">${impact_s}s</td>
      <td style="font-size:10px;color:var(--text-dim);">
        ${d.avg_impact > 0 ? '✅ Taze lastik avantajı' : '⚠️ Sınırlı etki'}
      </td>
    </tr>`;
  });
  if (pit.comment) showEl('pit-comment', pit.comment);
  setChart('chart-pit', data.charts?.pit_impact);
}

// ─────────────────────────────────────────────────────────────
// WEATHER
// ─────────────────────────────────────────────────────────────
function renderWeather(data) {
  const weather = data.analysis?.weather_impact || {};
  const tbody = document.getElementById('weather-tbody');
  tbody.innerHTML = '';
  Object.entries(weather).forEach(([col, info]) => {
    if (!info.pearson_r) return;
    const r = info.pearson_r;
    const cls = r > 0.3 ? 'corr-positive' : (r < -0.3 ? 'corr-negative' : 'corr-neutral');
    const sig = info.significant ? '<span style="color:var(--success)">✅ Anlamlı</span>'
                                 : '<span style="color:var(--text-muted)">— Anlamsız</span>';
    tbody.innerHTML += `<tr>
      <td>${info.label || col}</td>
      <td class="${cls}">${r.toFixed(4)}</td>
      <td>${info.p_value?.toFixed(4)}</td>
      <td>${sig}</td>
      <td style="font-size:10px;color:var(--text-dim);">${info.interpretation || ''}</td>
    </tr>`;
  });
  setChart('chart-temp', data.charts?.temp_vs_laptime);
  setChart('chart-corr', data.charts?.correlation_heatmap);
}

// ─────────────────────────────────────────────────────────────
// ML MODELS
// ─────────────────────────────────────────────────────────────
function renderML(data) {
  const ml = data.ml || {};

  // K-Means clusters
  const kmeans = ml.kmeans || {};
  const clusterDiv = document.getElementById('kmeans-clusters');
  clusterDiv.innerHTML = '<div class="cluster-grid">';
  Object.entries(kmeans.cluster_summary || {}).forEach(([name, members]) => {
    const cls = name.includes('Stabil') ? 'cluster-stable' :
                name.includes('Agresif') ? 'cluster-agresif' : 'cluster-duzensiz';
    clusterDiv.innerHTML += `
      <div class="cluster-card ${cls}">
        <div class="cluster-name">${name}</div>
        ${members.map(m => `<div class="cluster-member">${m}</div>`).join('')}
      </div>`;
  });
  clusterDiv.innerHTML += '</div>';
  if (kmeans.comment) {
    clusterDiv.innerHTML += `<div class="comment" style="margin-top:12px">${kmeans.comment}</div>`;
  }

  // Charts
  setChart('chart-kmeans', data.charts?.kmeans_clusters);
  setChart('chart-elbow', data.charts?.elbow_curve);
  setChart('chart-feature-imp', data.charts?.feature_importance);
  setChart('chart-model-compare', data.charts?.model_comparison);

  // Model comparison stats
  const comp = ml.comparison || {};
  const compDiv = document.getElementById('model-comparison-stats');
  compDiv.innerHTML = `
    <div class="model-stats">
      <div class="model-stat">
        <div class="model-stat-label">Decision Tree</div>
        <div class="model-stat-value dt">%${((comp.decision_tree_accuracy||0)*100).toFixed(1)}</div>
      </div>
      <div class="model-stat">
        <div class="model-stat-label">kNN (k=${ml.knn?.k||'?'})</div>
        <div class="model-stat-value knn">%${((comp.knn_accuracy||0)*100).toFixed(1)}</div>
      </div>
      <div class="model-stat">
        <div class="model-stat-label">Kazanan</div>
        <div class="model-stat-value win">${comp.winner || '—'}</div>
      </div>
    </div>
    ${comp.comment ? `<div class="model-stat-comment">${comp.comment}</div>` : ''}
  `;

  // Feature importance bars
  const dt = ml.decision_tree || {};
  const featsDiv = document.getElementById('feature-importance-bars');
  featsDiv.innerHTML = '';
  const maxImp = Math.max(...(dt.feature_importance || []).map(f => f[1]), 0.001);
  (dt.feature_importance || []).forEach(([feat, imp]) => {
    const pct = (imp / maxImp * 100).toFixed(0);
    featsDiv.innerHTML += `
      <div class="feat-row">
        <div class="feat-name">${feat}</div>
        <div class="feat-bar-bg">
          <div class="feat-bar-fill" style="width:${pct}%"></div>
        </div>
        <div class="feat-val">${imp.toFixed(3)}</div>
      </div>`;
  });

  // Most important feature
  setText('most-important-feature',
    dt.most_important_feature ? '🔑 ' + dt.most_important_feature : '—');
  if (dt.comment) showEl('ml-comment', dt.comment);
}

// ─────────────────────────────────────────────────────────────
// INSIGHTS
// ─────────────────────────────────────────────────────────────
function renderInsights(data) {
  const ins = data.insights || {};
  const top5 = ins.top5_summary || [];
  const tc = ins.track_character || {};

  const raceInd = document.getElementById('race-indicator');
  if (raceInd && tc.headline) {
    raceInd.title = tc.headline;
  }
}

// ─────────────────────────────────────────────────────────────
// PRESENTATION
// ─────────────────────────────────────────────────────────────
function renderPresentation(data) {
  const ov = data.overview || {};
  const ml = data.ml || {};
  const tire = data.analysis?.tire_strategy || {};
  const stab = data.analysis?.driver_stability || {};
  const dt = ml.decision_tree || {};

  // Race title
  const titleEl = document.getElementById('pres-race-name');
  if (titleEl) titleEl.innerHTML = `<em>${currentRace}</em> Grand Prix`;

  const insights = [
    {
      num: ov.fastest_driver || '—',
      label: 'En Hızlı Pilot',
      text: `En hızlı tur: ${ov.fastest_lap ? formatTime(ov.fastest_lap) : '—'}`
    },
    {
      num: ov.most_stable_driver || stab.most_stable || '—',
      label: 'En İstikrarlı Pilot',
      text: 'En yüksek consistency score'
    },
    {
      num: tire.best_compound || '—',
      label: 'En Avantajlı Lastik',
      text: tire.comment || 'Lastik stratejisi analizi'
    },
    {
      num: dt.most_important_feature || '—',
      label: 'En Kritik Faktör',
      text: 'Decision Tree feature importance'
    },
    {
      num: '%' + (((ml.comparison?.decision_tree_accuracy||0)*100).toFixed(0)),
      label: 'Model Doğruluğu',
      text: 'Kazanan: ' + (ml.comparison?.winner || '—')
    },
  ];

  const grid = document.getElementById('pres-insights');
  grid.innerHTML = insights.map(ins => `
    <div class="pres-insight">
      <div class="pres-label">${ins.label}</div>
      <div class="pres-number">${ins.num}</div>
      <div class="pres-text">${ins.text}</div>
    </div>
  `).join('');

  setChart('pres-chart', data.charts?.driver_avg_laptime);
}

// ─────────────────────────────────────────────────────────────
// RAPOR / PDF / CSV
// ─────────────────────────────────────────────────────────────
function downloadPDF() {
  window.open('/api/pdf/' + encodeURIComponent(currentRace), '_blank');
}

function generateReport() {
  showEl('report-loading', '', true);
  document.getElementById('report-content').style.display = 'none';

  fetch('/api/report/' + encodeURIComponent(currentRace))
    .then(r => r.json())
    .then(data => {
      document.getElementById('report-loading').style.display = 'none';
      const box = document.getElementById('report-content');
      box.style.display = 'block';
      box.textContent = data.content || data.error || 'Rapor üretilemedi.';
    });
}

// ─────────────────────────────────────────────────────────────
// PILOT KARŞILAŞTIR
// ─────────────────────────────────────────────────────────────
function compareDrivers() {
  const d1 = document.getElementById('driver1-select').value;
  const d2 = document.getElementById('driver2-select').value;
  if (!d1 || !d2 || d1 === d2) {
    alert('Lütfen iki farklı pilot seçin.');
    return;
  }
  fetch(`/api/compare/${encodeURIComponent(currentRace)}/${encodeURIComponent(d1)}/${encodeURIComponent(d2)}`)
    .then(r => r.json())
    .then(data => {
      if (data.chart) setChart('chart-driver-compare', data.chart);
      if (data.comment) showEl('compare-comment', data.comment);
      document.querySelector('.tab:nth-child(2)').click();
    });
}

// ─────────────────────────────────────────────────────────────
// CSV EXPORT
// ─────────────────────────────────────────────────────────────
function exportCSV() {
  window.open('/api/export/' + encodeURIComponent(currentRace), '_blank');
}

// ─────────────────────────────────────────────────────────────
// YARDIMCILAR
// ─────────────────────────────────────────────────────────────
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function setChart(id, b64) {
  const el = document.getElementById(id);
  if (!el) return;
  if (b64 && b64.startsWith('data:image')) {
    el.innerHTML = `<img src="${b64}" style="width:100%;border-radius:4px;">`;
  } else {
    el.innerHTML = '<div class="chart-empty">Grafik verisi yok</div>';
  }
}

function showEl(id, text, asLoading) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.display = 'block';
  if (!asLoading) el.textContent = text;
}

function showAlert(msg, type, targetId) {
  const id = targetId || 'overview-alert';
  const el = document.getElementById(id);
  if (el) {
    el.innerHTML = `<div class="alert alert-${type}">${msg}</div>`;
    setTimeout(() => { el.innerHTML = ''; }, 6000);
  }
}

function formatTime(seconds) {
  if (!seconds) return '—';
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(3).padStart(6, '0');
  return m + ':' + s;
}

function updateDriverSelects(data) {
  const drivers = (data.analysis?.driver_stability?.data || []).map(d => d.Driver);
  ['driver1-select', 'driver2-select'].forEach((id, i) => {
    const sel = document.getElementById(id);
    sel.innerHTML = `<option value="">— Driver ${i+1} —</option>`;
    drivers.forEach(d => {
      sel.innerHTML += `<option value="${d}">${d}</option>`;
    });
  });
}

function updateProcessedList(races) {
  const el = document.getElementById('processed-list');
  if (!el) return;
  el.innerHTML = races.map(r => `<div class="processed-item">${r}</div>`).join('')
               || '<div class="processed-item" style="opacity:.4">Henüz yok</div>';
}

// ─────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  loadRace();
});
</script>
</body>
</html>
"""




# ─────────────────────────────────────────────────────────────
# FLASK ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    processed_races = get_all_processed_races()
    return render_template_string(
        HTML_TEMPLATE,
        all_races=ALL_2025_RACES,
        default_race=DASHBOARD_CONFIG["default_race"],
        processed_races=processed_races,
        season=SEASON,
    )


@app.route("/api/race/<race_name>")
def api_race(race_name):
    """Yarış için tüm analiz verilerini döndür. Cache destekli."""
    try:
        logger.info(f"API isteği: {race_name}")
        first_load = not is_race_processed(race_name)

        # Veri yükle (lazy loading)
        df = load_or_fetch_race_data(race_name, use_sample_on_failure=True)
        if df is None or df.empty:
            return jsonify({"error": f"{race_name} verisi yüklenemedi."})

        is_sim = "IsSimulated" in df.columns and bool(df["IsSimulated"].any())

        # Analizler
        analysis = run_all_analyses(df, race_name)
        ml       = run_all_models(df, race_name)

        # Insight engine
        try:
            insights = generate_race_insights(race_name, df, analysis, ml)
        except Exception as e:
            logger.warning(f"Insight hatası: {e}")
            insights = {}

        # Grafik üretimi
        charts = generate_all_charts(df, analysis, ml, race_name)

        # Cache kaydet
        try:
            save_analysis_cache(race_name, analysis, ml, charts)
        except Exception as e:
            logger.warning(f"Cache kayıt hatası: {e}")

        # JSON'a dönüştür
        result = {
            "race_name":       race_name,
            "first_load":      first_load,
            "is_simulated":    is_sim,
            "overview":        analysis.get("race_overview", {}),
            "analysis":        _safe_serialize(analysis),
            "ml":              _safe_serialize(ml),
            "insights":        _safe_serialize(insights),
            "charts":          charts,
            "processed_races": get_all_processed_races(),
        }

        return jsonify(result)

    except Exception as e:
        logger.error(f"API hatası ({race_name}): {e}")
        return jsonify({"error": str(e)})


@app.route("/api/report/<race_name>")
def api_report(race_name):
    """Yarış raporu üret (MD + PDF) ve içeriğini döndür."""
    try:
        df = load_or_fetch_race_data(race_name, use_sample_on_failure=True)
        if df is None:
            return jsonify({"error": "Veri yok."})

        analysis = run_all_analyses(df, race_name)
        ml       = run_all_models(df, race_name)
        insights_data = {}
        try:
            insights_data = generate_race_insights(race_name, df, analysis, ml)
        except Exception:
            pass
        charts   = generate_all_charts(df, analysis, ml, race_name)

        # MD + TXT
        paths    = generate_race_report(race_name, analysis, ml)
        md_content = paths["markdown"].read_text(encoding="utf-8")

        # PDF
        pdf_path = None
        try:
            pdf_path = generate_pdf_report(race_name, analysis, ml, insights_data, charts)
        except Exception as e:
            logger.warning(f"PDF rapor hatası: {e}")

        return jsonify({
            "content":  md_content,
            "txt_path": str(paths["txt"]),
            "pdf_path": str(pdf_path) if pdf_path else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/compare/<race_name>/<driver1>/<driver2>")
def api_compare(race_name, driver1, driver2):
    """İki pilot karşılaştırması."""
    try:
        df = load_or_fetch_race_data(race_name, use_sample_on_failure=True)
        if df is None:
            return jsonify({"error": "Veri yok."})

        from src.analysis import compare_drivers
        from src.visualization import plot_driver_comparison
        comp = compare_drivers(df, driver1, driver2, race_name)
        chart = plot_driver_comparison(comp, race_name)
        return jsonify({
            "comparison": _safe_serialize(comp),
            "chart":      chart,
            "comment":    comp.get("comment", ""),
        })
    except Exception as e:
        return jsonify({"error": str(e)})




@app.route("/api/pdf/<race_name>")
def api_pdf(race_name):
    """PDF raporu indir."""
    from flask import send_file, Response
    from pathlib import Path as P
    from config import GENERATED_REPORTS_DIR
    import glob
    slug = race_name.lower().replace(" ", "_")
    pdfs = sorted(glob.glob(str(GENERATED_REPORTS_DIR / f"{slug}_report_*.pdf")))
    if pdfs:
        return send_file(pdfs[-1], mimetype="application/pdf",
                         as_attachment=True,
                         download_name=f"{slug}_f1_report.pdf")
    return Response("PDF bulunamadı.", status=404)

@app.route("/api/export/<race_name>")
def api_export(race_name):
    """İşlenmiş CSV'yi indir."""
    from flask import send_file, Response
    from src.utils import get_processed_csv_path
    path = get_processed_csv_path(race_name, PROCESSED_DATA_DIR)
    if path.exists():
        return send_file(str(path), mimetype="text/csv",
                         as_attachment=True,
                         download_name=f"{race_name.lower().replace(' ','_')}_processed.csv")
    return Response("Dosya bulunamadı.", status=404)


# ─────────────────────────────────────────────────────────────
# JSON SERİALİZATİON
# ─────────────────────────────────────────────────────────────

def _safe_serialize(obj):
    """Numpy/pandas tiplerini JSON'a güvenli serialize et."""
    import numpy as np
    import math

    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_serialize(i) for i in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, "to_dict"):   # DataFrame
        return _safe_serialize(obj.to_dict("records"))
    elif hasattr(obj, "item"):      # numpy scalar
        return obj.item()
    elif isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    print(f"""
╔══════════════════════════════════════════════╗
║  🏎️  F1 Race Intelligence Dashboard          ║
║  http://localhost:{port}                       ║
╚══════════════════════════════════════════════╝
    """)
    app.run(host="0.0.0.0", port=port, debug=False)
