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
  :root {
    --bg: #0D0D0D; --surface: #141414; --card: #1A1A1A;
    --border: #2A2A2A; --primary: #E8002D; --accent: #FF4D6D;
    --text: #FFFFFF; --subtext: #888; --success: #00C851;
    --warning: #FFB300; --info: #2196F3;
    --soft: #E8002D; --medium: #d4c200; --hard: #ccc;
    --font: 'Courier New', 'Lucida Console', monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: var(--font);
         font-size: 13px; line-height: 1.5; }

  /* HEADER */
  .header { background: #0a0a0a; border-bottom: 2px solid var(--primary);
             padding: 12px 24px; display: flex; align-items: center;
             justify-content: space-between; position: sticky; top: 0; z-index: 100; }
  .header-left { display: flex; align-items: center; gap: 12px; }
  .logo { font-size: 22px; font-weight: 900; color: var(--primary);
          letter-spacing: 2px; text-shadow: 0 0 20px rgba(232,0,45,0.5); }
  .subtitle { color: var(--subtext); font-size: 11px; letter-spacing: 3px; }
  .status-bar { display: flex; gap: 16px; }
  .status-pill { background: var(--card); border: 1px solid var(--border);
                 padding: 4px 12px; border-radius: 20px; font-size: 11px;
                 color: var(--subtext); }
  .status-pill.live { border-color: var(--primary); color: var(--primary); }

  /* LAYOUT */
  .layout { display: flex; min-height: calc(100vh - 60px); }

  /* SIDEBAR */
  .sidebar { width: 240px; background: var(--surface); border-right: 1px solid var(--border);
             padding: 16px; flex-shrink: 0; position: sticky; top: 60px;
             height: calc(100vh - 60px); overflow-y: auto; }
  .sidebar-section { margin-bottom: 20px; }
  .sidebar-label { font-size: 10px; color: var(--subtext); letter-spacing: 2px;
                   text-transform: uppercase; margin-bottom: 8px; }
  select, .btn { width: 100%; padding: 8px 10px; background: var(--card);
                 border: 1px solid var(--border); color: var(--text);
                 font-family: var(--font); font-size: 12px; border-radius: 4px;
                 cursor: pointer; transition: all 0.2s; }
  select:focus, .btn:hover { border-color: var(--primary); outline: none; }
  .btn-primary { background: var(--primary); border-color: var(--primary);
                 color: white; font-weight: bold; letter-spacing: 1px; }
  .btn-primary:hover { background: var(--accent); }
  .btn + .btn { margin-top: 6px; }

  /* TABS */
  .main { flex: 1; padding: 20px; overflow-x: hidden; }
  .tab-bar { display: flex; gap: 4px; border-bottom: 1px solid var(--border);
             margin-bottom: 20px; flex-wrap: wrap; }
  .tab { padding: 8px 16px; background: none; border: none; color: var(--subtext);
         font-family: var(--font); font-size: 12px; cursor: pointer;
         border-bottom: 2px solid transparent; margin-bottom: -1px;
         transition: all 0.2s; letter-spacing: 1px; }
  .tab.active { color: var(--primary); border-bottom-color: var(--primary); }
  .tab:hover { color: var(--text); }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  /* CARDS */
  .card { background: var(--card); border: 1px solid var(--border);
          border-radius: 6px; padding: 16px; margin-bottom: 16px; }
  .card-title { font-size: 11px; color: var(--subtext); letter-spacing: 2px;
                text-transform: uppercase; margin-bottom: 12px;
                padding-bottom: 8px; border-bottom: 1px solid var(--border); }
  .card-value { font-size: 28px; font-weight: 900; color: var(--text); }
  .card-sub { font-size: 11px; color: var(--subtext); margin-top: 2px; }
  .card-accent { border-left: 3px solid var(--primary); }

  /* METRICS ROW */
  .metrics-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
                 gap: 12px; margin-bottom: 20px; }

  /* CHART */
  .chart-container { background: var(--card); border: 1px solid var(--border);
                     border-radius: 6px; padding: 16px; margin-bottom: 16px; }
  .chart-title { font-size: 11px; color: var(--subtext); letter-spacing: 2px;
                 margin-bottom: 12px; }
  .chart-container img { width: 100%; border-radius: 4px; }
  .charts-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

  /* TABLE */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { background: #111; color: var(--subtext); font-size: 10px;
       letter-spacing: 1px; text-transform: uppercase; padding: 8px 10px;
       text-align: left; border-bottom: 1px solid var(--border); }
  td { padding: 8px 10px; border-bottom: 1px solid #1f1f1f; }
  tr:hover td { background: rgba(232,0,45,0.05); }
  .rank-1 { color: var(--primary); font-weight: bold; }
  .rank-2 { color: #aaa; }
  .badge { padding: 2px 8px; border-radius: 10px; font-size: 10px; }
  .badge-soft { background: rgba(232,0,45,0.2); color: var(--soft); }
  .badge-medium { background: rgba(212,194,0,0.2); color: var(--medium); }
  .badge-hard { background: rgba(200,200,200,0.15); color: var(--hard); }

  /* COMMENT BOX */
  .comment { background: rgba(232,0,45,0.08); border-left: 3px solid var(--primary);
             padding: 10px 14px; margin: 12px 0; font-size: 12px;
             color: #ddd; border-radius: 0 4px 4px 0; }

  /* CLUSTER BOX */
  .cluster-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                  gap: 12px; }
  .cluster-card { background: var(--card); border: 1px solid var(--border);
                  border-radius: 6px; padding: 14px; }
  .cluster-name { font-size: 12px; font-weight: bold; margin-bottom: 8px; }
  .cluster-stable { border-top: 3px solid var(--success); }
  .cluster-agresif { border-top: 3px solid var(--warning); }
  .cluster-duzensiz { border-top: 3px solid var(--subtext); }
  .cluster-member { font-size: 11px; color: var(--subtext);
                    padding: 2px 0; border-bottom: 1px solid #222; }

  /* LOADING */
  .loading { display: flex; align-items: center; justify-content: center;
             min-height: 300px; flex-direction: column; gap: 16px; }
  .spinner { width: 40px; height: 40px; border: 3px solid var(--border);
             border-top-color: var(--primary); border-radius: 50%;
             animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-text { color: var(--subtext); font-size: 12px; letter-spacing: 2px; }

  /* ALERT */
  .alert { padding: 12px 16px; border-radius: 4px; margin-bottom: 16px; font-size: 12px; }
  .alert-info { background: rgba(33,150,243,0.1); border: 1px solid rgba(33,150,243,0.3);
                color: #90CAF9; }
  .alert-warning { background: rgba(255,179,0,0.1); border: 1px solid rgba(255,179,0,0.3);
                   color: #FFE082; }
  .alert-success { background: rgba(0,200,81,0.1); border: 1px solid rgba(0,200,81,0.3);
                   color: #69F0AE; }

  /* REPORT */
  .report-box { background: #0a0a0a; border: 1px solid var(--border); border-radius: 6px;
                padding: 20px; font-size: 11px; line-height: 1.7; white-space: pre-wrap;
                max-height: 600px; overflow-y: auto; color: #ccc; }
  .report-box h1, .report-box h2, .report-box h3 { color: var(--primary); }

  /* FEATURE BAR */
  .feat-row { display: flex; align-items: center; gap: 8px;
              padding: 4px 0; border-bottom: 1px solid #1f1f1f; }
  .feat-name { width: 200px; font-size: 11px; color: var(--subtext); flex-shrink: 0; }
  .feat-bar-bg { flex: 1; height: 8px; background: var(--border); border-radius: 4px; }
  .feat-bar-fill { height: 100%; background: var(--primary); border-radius: 4px;
                   transition: width 0.5s ease; }
  .feat-val { width: 50px; text-align: right; font-size: 11px; color: var(--text); }

  /* PRESENTATION */
  .pres-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .pres-insight { background: var(--card); border: 1px solid var(--primary);
                  border-radius: 6px; padding: 16px; }
  .pres-number { font-size: 36px; color: var(--primary); font-weight: 900; }
  .pres-label { font-size: 11px; color: var(--subtext); letter-spacing: 1px; }
  .pres-text { font-size: 13px; color: var(--text); margin-top: 6px; }

  /* CORR TABLE */
  .corr-positive { color: #69F0AE; }
  .corr-negative { color: var(--accent); }
  .corr-neutral  { color: var(--subtext); }

  /* RESPONSIVE */
  @media (max-width: 900px) {
    .charts-grid { grid-template-columns: 1fr; }
    .pres-grid { grid-template-columns: 1fr; }
    .sidebar { display: none; }
  }
</style>
</head>
<body>

<!-- HEADER -->
<div class="header">
  <div class="header-left">
    <div>
      <div class="logo">F1 RACE INTELLIGENCE</div>
      <div class="subtitle">STRATEGY ANALYSIS SYSTEM — {{ season }} SEASON</div>
    </div>
  </div>
  <div class="status-bar">
    <div class="status-pill live" id="race-indicator">⚡ LOADING...</div>
    <div class="status-pill" id="data-pill">📊 --</div>
  </div>
</div>

<div class="layout">

<!-- SIDEBAR -->
<div class="sidebar">
  <div class="sidebar-section">
    <div class="sidebar-label">🏁 Yarış Seçimi</div>
    <select id="race-select" onchange="loadRace()">
      {% for race in all_races %}
      <option value="{{ race }}" {% if race == default_race %}selected{% endif %}>
        {{ race }}{% if race in processed_races %} ✓{% endif %}
      </option>
      {% endfor %}
    </select>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">👤 Pilot Karşılaştırma</div>
    <select id="driver1-select">
      <option value="">Pilot 1 Seçin</option>
    </select>
    <select id="driver2-select" style="margin-top:6px">
      <option value="">Pilot 2 Seçin</option>
    </select>
    <button class="btn btn-primary" style="margin-top:8px" onclick="compareDrivers()">KARŞILAŞTIR</button>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">🔧 Araçlar</div>
    <button class="btn" onclick="generateReport()">📄 Rapor Üret</button>
    <button class="btn" onclick="exportCSV()">💾 CSV İndir</button>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">📋 Hazır Yarışlar</div>
    <div id="processed-list" style="font-size:11px;color:var(--subtext);line-height:2;">
      Yükleniyor...
    </div>
  </div>

  <div class="sidebar-section">
    <div class="sidebar-label">ℹ️ Sistem</div>
    <div style="font-size:10px;color:var(--subtext);line-height:1.8;">
      Sezon: {{ season }}<br>
      Toplam: {{ all_races|length }} GP<br>
      FastF1 + SQLite<br>
      Lazy Loading ✓
    </div>
  </div>
</div>

<!-- MAIN -->
<div class="main">

  <!-- TAB BAR -->
  <div class="tab-bar">
    <button class="tab active" onclick="showTab('overview')">🏠 OVERVIEW</button>
    <button class="tab" onclick="showTab('driver')">👤 DRIVER PERF.</button>
    <button class="tab" onclick="showTab('tire')">🔴 TIRE STRATEGY</button>
    <button class="tab" onclick="showTab('pit')">🔧 PIT STOP</button>
    <button class="tab" onclick="showTab('weather')">🌡️ WEATHER</button>
    <button class="tab" onclick="showTab('ml')">🤖 ML MODELS</button>
    <button class="tab" onclick="showTab('report')">📄 REPORT</button>
    <button class="tab" onclick="showTab('presentation')">🎯 SUNUM</button>
  </div>

  <!-- LOADING STATE -->
  <div id="global-loading" style="display:none;">
    <div class="loading">
      <div class="spinner"></div>
      <div class="loading-text" id="loading-msg">VERİ YÜKLENİYOR...</div>
    </div>
  </div>

  <!-- ═══ OVERVIEW TAB ═══ -->
  <div id="tab-overview" class="tab-content active">
    <div id="overview-alert"></div>
    <div class="metrics-row" id="overview-metrics">
      <div class="card card-accent">
        <div class="card-title">🏎️ YARIŞ</div>
        <div class="card-value" id="ov-race">—</div>
        <div class="card-sub">{{ season }} Sezonu</div>
      </div>
      <div class="card">
        <div class="card-title">👤 PİLOT</div>
        <div class="card-value" id="ov-drivers">—</div>
        <div class="card-sub">toplam pilot</div>
      </div>
      <div class="card">
        <div class="card-title">🔄 TUR</div>
        <div class="card-value" id="ov-laps">—</div>
        <div class="card-sub">toplam tur</div>
      </div>
      <div class="card card-accent">
        <div class="card-title">⚡ EN HIZLI TUR</div>
        <div class="card-value" id="ov-fastest-time">—</div>
        <div class="card-sub" id="ov-fastest-driver">—</div>
      </div>
      <div class="card">
        <div class="card-title">🎯 EN İSTİKRARLI</div>
        <div class="card-value" id="ov-stable">—</div>
        <div class="card-sub">consistency</div>
      </div>
      <div class="card">
        <div class="card-title">🌡️ SICAKLIK</div>
        <div class="card-value" id="ov-temp">—</div>
        <div class="card-sub">hava / pist (°C)</div>
      </div>
    </div>
    <div class="charts-grid">
      <div class="chart-container">
        <div class="chart-title">ORTALAMA TUR SÜRESİ</div>
        <div id="chart-driver-avg"></div>
      </div>
      <div class="chart-container">
        <div class="chart-title">CONSISTENCY SCORE</div>
        <div id="chart-consistency"></div>
      </div>
    </div>
  </div>

  <!-- ═══ DRIVER PERFORMANCE TAB ═══ -->
  <div id="tab-driver" class="tab-content">
    <div class="card">
      <div class="card-title">PİLOT İSTİKRAR SIRALAMASI</div>
      <table id="stability-table">
        <thead><tr>
          <th>SıRA</th><th>PİLOT</th><th>TAKIM</th>
          <th>ORT. TUR (s)</th><th>EN HIZLI (s)</th>
          <th>STD SAPMA</th><th>CONSISTENCY</th>
        </tr></thead>
        <tbody id="stability-tbody"></tbody>
      </table>
    </div>
    <div id="driver-comment" class="comment" style="display:none;"></div>

    <div class="chart-container">
      <div class="chart-title">PILOT KARŞILAŞTIRMA</div>
      <div id="chart-driver-compare"></div>
      <div id="compare-comment" class="comment" style="display:none;margin-top:10px;"></div>
    </div>
  </div>

  <!-- ═══ TIRE STRATEGY TAB ═══ -->
  <div id="tab-tire" class="tab-content">
    <div class="card">
      <div class="card-title">LASTİK TİPİ KARŞILAŞTIRMASI</div>
      <table id="compound-table">
        <thead><tr>
          <th>LASTİK</th><th>ORT. TUR (s)</th><th>EN HIZLI (s)</th>
          <th>STD SAPMA</th><th>TUR SAYISI</th>
        </tr></thead>
        <tbody id="compound-tbody"></tbody>
      </table>
    </div>
    <div id="tire-comment" class="comment" style="display:none;"></div>
    <div class="charts-grid">
      <div class="chart-container">
        <div class="chart-title">COMPOUND ORT. TUR SÜRESİ</div>
        <div id="chart-compound"></div>
      </div>
      <div class="chart-container">
        <div class="chart-title">LASTİK BOZULMA EĞRİSİ</div>
        <div id="chart-degradation"></div>
      </div>
    </div>
    <div class="chart-container">
      <div class="chart-title">LASTİK YAŞI vs TUR ZAMANI</div>
      <div id="chart-tyre-scatter"></div>
    </div>
    <div class="card">
      <div class="card-title">LASTİK BOZULMA ANALİZİ</div>
      <table id="degradation-table">
        <thead><tr>
          <th>LASTİK</th><th>BOZULMA (s/tur)</th><th>R²</th><th>YORUM</th>
        </tr></thead>
        <tbody id="degradation-tbody"></tbody>
      </table>
    </div>
    <div id="deg-comment" class="comment" style="display:none;"></div>
  </div>

  <!-- ═══ PIT STOP TAB ═══ -->
  <div id="tab-pit" class="tab-content">
    <div class="card card-accent">
      <div class="card-title">ORTALAMA PIT STOP ETKİSİ</div>
      <div class="card-value" id="pit-avg-impact">—</div>
      <div class="card-sub">saniye (pozitif = pit sonrası hızlandı)</div>
    </div>
    <div class="chart-container">
      <div class="chart-title">PİT STOP ETKİSİ (ÖNCESİ / SONRASI)</div>
      <div id="chart-pit"></div>
    </div>
    <div class="card">
      <div class="card-title">PİLOT BAZLI PİT STOP ANALİZİ</div>
      <table id="pit-table">
        <thead><tr>
          <th>PİLOT</th><th>PİT SAYISI</th><th>ORTALAMA ETKİ (s)</th><th>YORUM</th>
        </tr></thead>
        <tbody id="pit-tbody"></tbody>
      </table>
    </div>
    <div id="pit-comment" class="comment" style="display:none;"></div>
  </div>

  <!-- ═══ WEATHER TAB ═══ -->
  <div id="tab-weather" class="tab-content">
    <div class="chart-container">
      <div class="chart-title">SICAKLIK vs TUR ZAMANI</div>
      <div id="chart-temp"></div>
    </div>
    <div class="chart-container">
      <div class="chart-title">KORELASYON MATRİSİ</div>
      <div id="chart-corr"></div>
    </div>
    <div class="card">
      <div class="card-title">PEARSON KORELASYON ANALİZİ</div>
      <table>
        <thead><tr>
          <th>DEĞİŞKEN</th><th>PEARSON r</th><th>p-DEĞERİ</th>
          <th>ANLAMLI?</th><th>YORUM</th>
        </tr></thead>
        <tbody id="weather-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- ═══ ML MODELS TAB ═══ -->
  <div id="tab-ml" class="tab-content">
    <div class="alert alert-info">
      🤖 Makine öğrenmesi modelleri pilot performans verisi üzerinde çalıştırılmıştır.
    </div>

    <div class="card">
      <div class="card-title">K-MEANS CLUSTERING — PERFORMANS KÜMELERİ</div>
      <div id="kmeans-clusters"></div>
    </div>
    <div class="chart-container">
      <div class="chart-title">CLUSTER GÖRSELLEŞTİRMESİ (PCA)</div>
      <div id="chart-kmeans"></div>
    </div>
    <div class="chart-container">
      <div class="chart-title">ELBOW METHOD (OPTİMAL K)</div>
      <div id="chart-elbow"></div>
    </div>

    <div class="charts-grid">
      <div class="card">
        <div class="card-title">MODEL KARŞILAŞTIRMASI</div>
        <div id="model-comparison-stats"></div>
        <div id="chart-model-compare"></div>
      </div>
      <div class="card">
        <div class="card-title">FEATURE IMPORTANCE (DECISION TREE)</div>
        <div id="feature-importance-bars"></div>
        <div id="chart-feature-imp"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-title">🔑 EN ÖNEMLİ FAKTÖR</div>
      <div id="most-important-feature"
           style="font-size:20px;color:var(--primary);padding:12px 0;font-weight:900;">
        Yükleniyor...
      </div>
      <div id="ml-comment" class="comment"></div>
    </div>
  </div>

  <!-- ═══ REPORT TAB ═══ -->
  <div id="tab-report" class="tab-content">
    <div style="display:flex;gap:10px;margin-bottom:16px;">
      <button class="btn btn-primary" onclick="generateReport()" style="width:auto;padding:8px 20px;">
        📄 RAPOR ÜRET
      </button>
      <button class="btn" onclick="downloadPDF()" style="width:auto;padding:8px 20px;margin-left:8px;background:#333;border-color:#555;">
        📊 PDF İNDİR
      </button>
    </div>
    <div id="report-loading" style="display:none;">
      <div class="loading"><div class="spinner"></div><div class="loading-text">RAPOR ÜRETILIYOR...</div></div>
    </div>
    <div id="report-content" class="report-box">
      Rapor üretmek için "RAPOR ÜRET" butonuna tıklayın.
    </div>
  </div>

  <!-- ═══ PRESENTATION TAB ═══ -->
  <div id="tab-presentation" class="tab-content">
    <div class="alert alert-success">
      🎯 Sunum Modu — En kritik 5 çıkarım ve model sonuçları
    </div>
    <div class="pres-grid" id="pres-insights"></div>
    <div class="chart-container" style="margin-top:20px;">
      <div class="chart-title">EN ÖNEMLİ YARIŞ GRAFİĞİ</div>
      <div id="pres-chart"></div>
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
    showAlert('⏳ Bu yarış ilk kez yükleniyor — veriler FastF1\'den çekildi ve işlendi.', 'info', 'overview-alert');
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
    const cls = i === 0 ? 'rank-1' : (i === 1 ? 'rank-2' : '');
    tbody.innerHTML += `<tr class="${cls}">
      <td>${r.stability_rank ?? i+1}</td>
      <td>${r.Driver}</td>
      <td style="color:var(--subtext)">${r.Team || '—'}</td>
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
      <td style="color:var(--subtext);font-size:11px;">${info.comment || ''}</td>
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
      <td style="color:${color};font-weight:bold;">${impact_s}s</td>
      <td style="color:var(--subtext);font-size:11px;">
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
                                 : '<span style="color:var(--subtext)">❌ Anlamsız</span>';
    tbody.innerHTML += `<tr>
      <td>${info.label || col}</td>
      <td class="${cls}">${r.toFixed(4)}</td>
      <td>${info.p_value?.toFixed(4)}</td>
      <td>${sig}</td>
      <td style="color:var(--subtext);font-size:11px;">${info.interpretation || ''}</td>
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
    const color = name.includes('Stabil') ? 'var(--success)' :
                  name.includes('Agresif') ? 'var(--warning)' : 'var(--subtext)';
    clusterDiv.innerHTML += `
      <div class="cluster-card ${cls}">
        <div class="cluster-name" style="color:${color}">${name}</div>
        ${members.map(m => `<div class="cluster-member">· ${m}</div>`).join('')}
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
    <div style="display:flex;gap:20px;margin-bottom:12px;">
      <div>
        <div style="font-size:10px;color:var(--subtext)">DECISION TREE</div>
        <div style="font-size:22px;color:var(--primary);font-weight:900;">
          %${((comp.decision_tree_accuracy||0)*100).toFixed(1)}
        </div>
      </div>
      <div>
        <div style="font-size:10px;color:var(--subtext)">kNN (k=${ml.knn?.k||'?'})</div>
        <div style="font-size:22px;color:#3A8AFF;font-weight:900;">
          %${((comp.knn_accuracy||0)*100).toFixed(1)}
        </div>
      </div>
      <div>
        <div style="font-size:10px;color:var(--subtext)">KAZANAN</div>
        <div style="font-size:16px;color:var(--success);font-weight:bold;">
          ${comp.winner || '—'}
        </div>
      </div>
    </div>
    ${comp.comment ? `<div style="font-size:11px;color:var(--subtext)">${comp.comment}</div>` : ''}
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
// PRESENTATION
// ─────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────
// INSIGHTS
// ─────────────────────────────────────────────────────────────
function renderInsights(data) {
  const ins = data.insights || {};
  const top5 = ins.top5_summary || [];
  const tc = ins.track_character || {};

  // AI Insights sekmesi için basit gösterim
  // (sekme yapısı Flask'ta statik — JS ile content inject edilir)
  // Track character badge
  const raceInd = document.getElementById('race-indicator');
  if (raceInd && tc.headline) {
    raceInd.title = tc.headline;
  }
}

function renderPresentation(data) {
  const ov = data.overview || {};
  const ml = data.ml || {};
  const tire = data.analysis?.tire_strategy || {};
  const stab = data.analysis?.driver_stability || {};
  const dt = ml.decision_tree || {};

  const insights = [
    {
      num: ov.fastest_driver || '—',
      label: 'EN HIZLI PİLOT',
      text: `En hızlı tur: ${ov.fastest_lap ? formatTime(ov.fastest_lap) : '—'}`
    },
    {
      num: ov.most_stable_driver || stab.most_stable || '—',
      label: 'EN İSTİKRARLI PİLOT',
      text: 'En yüksek consistency score'
    },
    {
      num: tire.best_compound || '—',
      label: 'EN AVANTAJLI LASTİK',
      text: tire.comment || 'Lastik stratejisi analizi'
    },
    {
      num: dt.most_important_feature || '—',
      label: 'EN KRİTİK FAKTÖR',
      text: 'Decision Tree feature importance'
    },
    {
      num: '%' + (((ml.comparison?.decision_tree_accuracy||0)*100).toFixed(0)),
      label: 'MODEL DOĞRULUĞU',
      text: 'Kazanan: ' + (ml.comparison?.winner || '—')
    },
  ];

  const grid = document.getElementById('pres-insights');
  grid.innerHTML = insights.map(ins => `
    <div class="pres-insight">
      <div class="pres-number">${ins.num}</div>
      <div class="pres-label">${ins.label}</div>
      <div class="pres-text">${ins.text}</div>
    </div>
  `).join('');

  setChart('pres-chart', data.charts?.driver_avg_laptime);
}

// ─────────────────────────────────────────────────────────────
// RAPOR ÜRET
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
// PİLOT KARŞILAŞTIR
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
      // Driver sekmesine geç
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
    el.innerHTML = '<div style="padding:20px;color:var(--subtext);text-align:center;">Grafik yok</div>';
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
    sel.innerHTML = `<option value="">Pilot ${i+1} Seçin</option>`;
    drivers.forEach(d => {
      sel.innerHTML += `<option value="${d}">${d}</option>`;
    });
  });
}

function updateProcessedList(races) {
  const el = document.getElementById('processed-list');
  if (!el) return;
  el.innerHTML = races.map(r => `<div>✓ ${r}</div>`).join('') || '<div>Yok</div>';
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
