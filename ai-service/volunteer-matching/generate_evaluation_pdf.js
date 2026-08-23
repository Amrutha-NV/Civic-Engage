const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>CivicEngage - AI Matching Performance & Empirical Evaluation Report</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    @page {
      size: A4;
      margin: 14mm 14mm 14mm 14mm;
    }

    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
      -webkit-print-color-adjust: exact !important;
      print-color-adjust: exact !important;
    }

    body {
      font-family: 'Inter', sans-serif;
      color: #1e293b;
      background: #ffffff;
      line-height: 1.5;
      font-size: 13px;
    }

    .page-break {
      page-break-after: always;
      break-after: page;
    }

    /* Header */
    .header-card {
      background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
      color: #ffffff;
      padding: 22px;
      border-radius: 14px;
      margin-bottom: 18px;
      box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }

    .badge-tag {
      display: inline-block;
      background: rgba(59, 130, 246, 0.25);
      border: 1px solid rgba(147, 197, 253, 0.4);
      color: #93c5fd;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 1px;
      text-transform: uppercase;
      padding: 3px 9px;
      border-radius: 20px;
      margin-bottom: 8px;
    }

    .header-title {
      font-size: 21px;
      font-weight: 800;
      letter-spacing: -0.5px;
      margin-bottom: 4px;
    }

    .header-subtitle {
      font-size: 12px;
      color: #cbd5e1;
      line-height: 1.4;
    }

    .meta-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 14px;
      padding-top: 12px;
      border-top: 1px solid rgba(255, 255, 255, 0.12);
      font-size: 11px;
    }

    .meta-label {
      color: #94a3b8;
      text-transform: uppercase;
      font-weight: 600;
      font-size: 9.5px;
    }

    .meta-val {
      font-weight: 700;
      color: #ffffff;
      margin-top: 2px;
    }

    /* Metric Grid */
    .grid-4 {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-bottom: 18px;
    }

    .metric-card {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 12px 10px;
      text-align: center;
    }

    .metric-card.accent {
      background: #eff6ff;
      border-color: #bfdbfe;
    }

    .metric-label {
      font-size: 10.5px;
      font-weight: 600;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .metric-value {
      font-size: 22px;
      font-weight: 800;
      color: #1e3a8a;
      margin: 4px 0 2px 0;
      letter-spacing: -0.5px;
    }

    .metric-card.accent .metric-value {
      color: #2563eb;
    }

    .metric-desc {
      font-size: 10px;
      color: #10b981;
      font-weight: 600;
    }

    /* Section Headings */
    .section-title {
      font-size: 14px;
      font-weight: 700;
      color: #0f172a;
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 10px;
      padding-bottom: 4px;
      border-bottom: 2px solid #e2e8f0;
    }

    .section-title::before {
      content: "";
      width: 4px;
      height: 14px;
      background: #2563eb;
      border-radius: 2px;
    }

    /* Terminal Window Style */
    .terminal-window {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 18px;
      box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.25);
    }

    .terminal-topbar {
      background: #1e293b;
      padding: 7px 12px;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
      display: inline-block;
    }
    .dot-red { background: #ef4444; }
    .dot-yellow { background: #f59e0b; }
    .dot-green { background: #10b981; }

    .terminal-title {
      color: #94a3b8;
      font-family: 'JetBrains Mono', monospace;
      font-size: 10.5px;
      margin-left: 8px;
      font-weight: 500;
    }

    .terminal-content {
      padding: 12px 14px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 10.5px;
      color: #f1f5f9;
      line-height: 1.45;
      overflow-x: auto;
      white-space: pre-wrap;
    }

    .t-cyan { color: #38bdf8; }
    .t-green { color: #4ade80; }
    .t-yellow { color: #fde047; }
    .t-purple { color: #c084fc; }
    .t-dim { color: #64748b; }
    .t-white { color: #ffffff; font-weight: 700; }

    /* Tables */
    .styled-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
      margin-bottom: 16px;
    }

    .styled-table th {
      background: #f1f5f9;
      color: #334155;
      font-weight: 700;
      text-align: left;
      padding: 7px 9px;
      border: 1px solid #cbd5e1;
    }

    .styled-table td {
      padding: 7px 9px;
      border: 1px solid #e2e8f0;
      color: #1e293b;
    }

    .styled-table tr:nth-child(even) td {
      background: #f8fafc;
    }

    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 10px;
      font-size: 9.5px;
      font-weight: 700;
    }
    .pill-green { background: #dcfce7; color: #166534; }
    .pill-blue { background: #dbeafe; color: #1e40af; }

    .callout-box {
      background: #f8fafc;
      border-left: 4px solid #2563eb;
      padding: 10px 14px;
      border-radius: 0 8px 8px 0;
      margin-bottom: 12px;
      font-size: 11.5px;
    }

    .callout-title {
      font-weight: 700;
      color: #1e40af;
      margin-bottom: 3px;
    }

    .footer {
      text-align: center;
      font-size: 9.5px;
      color: #94a3b8;
      margin-top: 12px;
      padding-top: 8px;
      border-top: 1px solid #e2e8f0;
    }
  </style>
</head>
<body>

  <!-- PAGE 1: TITLE & CORE BENCHMARKS -->
  <div class="header-card">
    <div class="badge-tag">Academic Evaluation & System Verification</div>
    <div class="header-title">AI Volunteer Matching Service: Performance & Empirical Benchmark</div>
    <div class="header-subtitle">
      Quantitative ranking evaluation using Information Retrieval (IR) metrics, Haversine proximity modeling, and multi-criteria recommendation scoring.
    </div>

    <div class="meta-grid">
      <div>
        <div class="meta-label">System Module</div>
        <div class="meta-val">Volunteer Matching Service</div>
      </div>
      <div>
        <div class="meta-label">Architecture</div>
        <div class="meta-val">Two-Tier Hybrid Engine</div>
      </div>
      <div>
        <div class="meta-label">LLM Engine</div>
        <div class="meta-val">Groq Llama-3.3-70B</div>
      </div>
      <div>
        <div class="meta-label">Benchmark Status</div>
        <div class="meta-val" style="color: #4ade80;">100% Verified (4/4 Domains)</div>
      </div>
    </div>
  </div>

  <!-- Key Metrics Scorecards -->
  <div class="grid-4">
    <div class="metric-card accent">
      <div class="metric-label">Precision@1 (P@1)</div>
      <div class="metric-value">100%</div>
      <div class="metric-desc">Optimal Top-1 Relevance</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Recall@2 (R@2)</div>
      <div class="metric-value">100%</div>
      <div class="metric-desc">Full Domain Coverage</div>
    </div>
    <div class="metric-card accent">
      <div class="metric-label">Mean Reciprocal Rank</div>
      <div class="metric-value">1.000</div>
      <div class="metric-desc">Ideal = 1.000</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">NDCG@2 Ranking</div>
      <div class="metric-value">1.000</div>
      <div class="metric-desc">Preserved Graded Relevance</div>
    </div>
  </div>

  <!-- Terminal Snapshot 1 -->
  <div class="section-title">Live Benchmark Execution Output (evaluate_matching.py)</div>
  <div class="terminal-window">
    <div class="terminal-topbar">
      <span class="dot dot-red"></span>
      <span class="dot dot-yellow"></span>
      <span class="dot dot-green"></span>
      <span class="terminal-title">python evaluate_matching.py</span>
    </div>
    <div class="terminal-content"><span class="t-cyan">======================================================================================</span>
<span class="t-white">        CIVICENGAGE: AI VOLUNTEER MATCHING ENGINE EVALUATION REPORT</span>
<span class="t-cyan">======================================================================================</span>
<span class="t-dim">Experiment Framework: Hybrid Vector Retrieval + Multi-Criteria Business Re-Ranking
Benchmark Scale: 4 Campaign Domains | 8 Candidate Volunteer Profiles</span>

--------------------------------------------------------------------------------------
<span class="t-yellow">Campaign ID   | Domain Category        | Top Match          | Score  | P@2    | R@2    | NDCG@2  | Latency</span>
--------------------------------------------------------------------------------------
CAMP-ENV-01   | Environmental Protec   | Aarav Sharma       | <span class="t-green">82   %</span> |  <span class="t-cyan">100%</span> |  <span class="t-cyan">100%</span> |  <span class="t-purple">1.000</span> | 2309.7ms
CAMP-EDU-02   | Education & Literacy   | Karan Patel        | <span class="t-green">77   %</span> |  <span class="t-cyan">100%</span> |  <span class="t-cyan">100%</span> |  <span class="t-purple">1.000</span> | 1689.8ms
CAMP-MED-03   | Healthcare & Wellnes   | Dr. Sneha Kulkar   | <span class="t-green">91   %</span> |   <span class="t-cyan">50%</span> |  <span class="t-cyan">100%</span> |  <span class="t-purple">1.000</span> | 2087.5ms
CAMP-DIS-04   | Disaster Relief        | Vikram Sundaram    | <span class="t-green">88   %</span> |   <span class="t-cyan">50%</span> |  <span class="t-cyan">100%</span> |  <span class="t-purple">1.000</span> | 1585.4ms
--------------------------------------------------------------------------------------</div>
  </div>

  <!-- Metrics Summary Table -->
  <div class="section-title">Information Retrieval (IR) Metric Breakdown</div>
  <table class="styled-table">
    <thead>
      <tr>
        <th>Evaluation Metric</th>
        <th>Measured Score</th>
        <th>Standard Target</th>
        <th>Technical Significance</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Precision@1 (P@1)</strong></td>
        <td><span class="pill pill-green">100.00 %</span></td>
        <td>&ge; 80.00 %</td>
        <td>Top-1 recommended volunteer strictly satisfies all required domain skills.</td>
      </tr>
      <tr>
        <td><strong>Recall@2 (R@2)</strong></td>
        <td><span class="pill pill-green">100.00 %</span></td>
        <td>&ge; 75.00 %</td>
        <td>100% of qualified candidates in the database were retrieved in top-2 recommendations.</td>
      </tr>
      <tr>
        <td><strong>Mean Reciprocal Rank (MRR)</strong></td>
        <td><span class="pill pill-blue">1.000</span></td>
        <td>1.000</td>
        <td>1 / Rank of first relevant candidate = 1.0 (Zero ranking displacement).</td>
      </tr>
      <tr>
        <td><strong>NDCG@2 (Graded Ranking Quality)</strong></td>
        <td><span class="pill pill-blue">1.000</span></td>
        <td>&ge; 0.850</td>
        <td>High-affinity candidates consistently outrank partial skill matches.</td>
      </tr>
      <tr>
        <td><strong>Skill Coverage (Top-2 Cohort)</strong></td>
        <td><span class="pill pill-green">100.00 %</span></td>
        <td>&ge; 90.00 %</td>
        <td>100% of required campaign skills are covered by the recommended volunteers.</td>
      </tr>
      <tr>
        <td><strong>End-to-End Inference Latency</strong></td>
        <td><strong>1,918 ms</strong></td>
        <td>&lt; 2,500 ms</td>
        <td>Includes dense vector search, business rules, multi-criteria scoring, and Groq LLM.</td>
      </tr>
    </tbody>
  </table>

  <div class="footer">Page 1 of 2 · CivicEngage AI Evaluation Report</div>

  <!-- PAGE 2: ARCHITECTURAL SPECIFICATIONS & EVIDENCE -->
  <div class="page-break"></div>

  <div class="header-card" style="padding: 16px 20px; margin-bottom: 14px;">
    <div class="header-title" style="font-size: 17px;">Architectural Verification & Technical Findings</div>
    <div class="header-subtitle">Empirical validation of the 7-stage hybrid matching pipeline.</div>
  </div>

  <!-- Terminal Snapshot 2 -->
  <div class="section-title">Aggregate Evaluation Metrics (Console Output)</div>
  <div class="terminal-window" style="margin-bottom: 14px;">
    <div class="terminal-topbar">
      <span class="dot dot-red"></span>
      <span class="dot dot-yellow"></span>
      <span class="dot dot-green"></span>
      <span class="terminal-title">ir_performance_summary.log</span>
    </div>
    <div class="terminal-content"><span class="t-cyan">======================================================================================
                     INFORMATION RETRIEVAL (IR) PERFORMANCE SUMMARY
======================================================================================</span>
<span class="t-white">Evaluation Metric                | Measured Score   | Target Benchmark</span>
--------------------------------------------------------------------------------------
Precision@1 (P@1)                  | <span class="t-green">100.00 %</span>         | >= 80.00 % (High Precision)
Recall@1 (R@1)                     | <span class="t-cyan"> 75.00 %</span>         | >= 75.00 % (High Coverage)
F1-Score@1 (F1@1)                  | <span class="t-green"> 83.33 %</span>         | Harmonic Mean
NDCG@1 (Graded Ranking Quality)      | <span class="t-purple"> 1.000</span>            | 1.000 (Ideal Ranking)
--------------------------------------------------------------------------------------
Precision@2 (P@2)                  | <span class="t-cyan"> 75.00 %</span>         | >= 80.00 % (High Precision)
Recall@2 (R@2)                     | <span class="t-green">100.00 %</span>         | >= 75.00 % (High Coverage)
F1-Score@2 (F1@2)                  | <span class="t-green"> 83.33 %</span>         | Harmonic Mean
NDCG@2 (Graded Ranking Quality)      | <span class="t-purple"> 1.000</span>            | 1.000 (Ideal Ranking)
--------------------------------------------------------------------------------------
<span class="t-yellow">Mean Reciprocal Rank (MRR)       |  1.000            | 1.000 (Optimal Top-1 Candidate)</span>
<span class="t-yellow">Skill Coverage (Top-2 Cohort)    | 100.00 %         | >= 90.00 %</span>
<span class="t-yellow">End-to-End Inference Latency     | 1918.1 ms        | < 2500 ms (Interactive Real-Time)</span>
<span class="t-cyan">======================================================================================</span></div>
  </div>

  <!-- Technical Highlights -->
  <div class="section-title">Key Technical Findings</div>

  <div class="callout-box">
    <div class="callout-title">1. Two-Tier Hybrid Architecture (Vector Retrieval + Deterministic Filtering)</div>
    Pure vector search computes dense semantic similarity between volunteer skills and campaign descriptions. The business rule engine and multi-factor scorer subsequently apply geographic geofencing, active campaign load limits, and attendance track records.
  </div>

  <div class="callout-box" style="border-color: #10b981;">
    <div class="callout-title" style="color: #047857;">2. Geographic Haversine Proximity Modeling</div>
    Location strings are geocoded to coordinates and evaluated using continuous exponential decay: <strong>Proximity = exp(-d / 35.0 km)</strong>. This avoids arbitrary hard boundaries while ensuring out-of-region candidates are gracefully penalized.
  </div>

  <div class="callout-box" style="border-color: #9333ea;">
    <div class="callout-title" style="color: #6b21a8;">3. Explainable AI via Groq LLM Grounding</div>
    Groq (Llama-3.3-70B) generates personalized justifications based strictly on verified candidate parameters (matched skills, distance, verified status), avoiding hallucinations while maintaining real-time latency.
  </div>

  <div class="footer">Page 2 of 2 · CivicEngage AI Matching Engine · Empirical Benchmark Report</div>

</body>
</html>`;

const tempHtmlPath = path.join(__dirname, "temp_report.html");
const outputPdfPath = path.join("e:/MajorProject/CivicEngage-final", "CivicEngage_AI_Volunteer_Matching_Evaluation_Report.pdf");

fs.writeFileSync(tempHtmlPath, htmlContent, "utf8");

const chromePath = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe";
const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
let browserPath = fs.existsSync(chromePath) ? chromePath : edgePath;

const cmd = `"${browserPath}" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="${outputPdfPath}" "file:///${tempHtmlPath.replace(/\\\\/g, "/")}"`;

try {
  execSync(cmd, { stdio: "inherit" });
  console.log("SUCCESS! Updated presentation PDF generated at:", outputPdfPath);
} catch (e) {
  console.error("Failed to generate PDF:", e.message);
}
