/**
 * Retrieval Trace Inspector & Head-to-Head Benchmark Studio
 */
const TraceInspector = {
  showTrace(trace) {
    if (!trace) return;

    const lat = trace.latencies_ms || {};
    const denseHits = trace.dense_hits || [];
    const sparseHits = trace.sparse_hits || [];
    const rerankedHits = trace.reranked_hits || trace.fused_hits || [];

    const denseRows = denseHits.slice(0, 5).map(h => `
      <tr>
        <td>#${h.rank}</td>
        <td><strong>${h.title}</strong></td>
        <td><code>${h.score}</code></td>
      </tr>
    `).join("");

    const sparseRows = sparseHits.slice(0, 5).map(h => `
      <tr>
        <td>#${h.rank}</td>
        <td><strong>${h.title}</strong></td>
        <td><code>${h.score}</code></td>
      </tr>
    `).join("");

    const rerankedRows = rerankedHits.slice(0, 5).map(h => `
      <tr>
        <td>#${h.rank}</td>
        <td><strong>${h.title}</strong></td>
        <td><code>${h.rerank_score || h.rrf_score || h.score}</code></td>
      </tr>
    `).join("");

    App.openDrawer("RAG Retrieval Trace", `
      <div style="display:flex; flex-direction:column; gap:16px;">
        <!-- Latency Waterfall -->
        <div style="background:var(--bg-surface-elevated); padding:14px; border-radius:8px; border:1px solid var(--border-subtle);">
          <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:8px;">STAGE LATENCY WATERFALL</div>
          <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; text-align:center;">
            <div style="background:var(--bg-base); padding:8px; border-radius:6px;">
              <div style="font-size:0.75rem; color:var(--text-muted);">Dense</div>
              <div style="font-weight:700; color:var(--accent-secondary);">${lat.dense || 0}ms</div>
            </div>
            <div style="background:var(--bg-base); padding:8px; border-radius:6px;">
              <div style="font-size:0.75rem; color:var(--text-muted);">BM25</div>
              <div style="font-weight:700; color:#38bdf8;">${lat.sparse || 0}ms</div>
            </div>
            <div style="background:var(--bg-base); padding:8px; border-radius:6px;">
              <div style="font-size:0.75rem; color:var(--text-muted);">Rerank</div>
              <div style="font-weight:700; color:#a855f7;">${lat.rerank || 0}ms</div>
            </div>
            <div style="background:var(--bg-base); padding:8px; border-radius:6px;">
              <div style="font-size:0.75rem; color:var(--text-muted);">Total</div>
              <div style="font-weight:700; color:var(--accent-emerald);">${lat.total_retrieval || 0}ms</div>
            </div>
          </div>
        </div>

        <!-- Dense Candidates -->
        <div>
          <div style="font-size:0.8rem; font-weight:600; color:var(--text-secondary); margin-bottom:6px;">Dense Vector Top Candidates</div>
          <table class="benchmark-table" style="font-size:0.8rem;">
            <thead><tr><th>Rank</th><th>Document</th><th>Cosine</th></tr></thead>
            <tbody>${denseRows || '<tr><td colspan="3">No hits</td></tr>'}</tbody>
          </table>
        </div>

        <!-- Sparse BM25 Candidates -->
        <div>
          <div style="font-size:0.8rem; font-weight:600; color:var(--text-secondary); margin-bottom:6px;">Sparse BM25 Lexical Candidates</div>
          <table class="benchmark-table" style="font-size:0.8rem;">
            <thead><tr><th>Rank</th><th>Document</th><th>BM25 Score</th></tr></thead>
            <tbody>${sparseRows || '<tr><td colspan="3">No hits</td></tr>'}</tbody>
          </table>
        </div>

        <!-- Final Cross-Encoder Reranked -->
        <div>
          <div style="font-size:0.8rem; font-weight:600; color:var(--text-secondary); margin-bottom:6px;">Final Cross-Encoder Reranked Passages</div>
          <table class="benchmark-table" style="font-size:0.8rem;">
            <thead><tr><th>Rank</th><th>Document</th><th>Score</th></tr></thead>
            <tbody>${rerankedRows || '<tr><td colspan="3">No hits</td></tr>'}</tbody>
          </table>
        </div>
      </div>
    `);
  },
};

const BenchmarkStudio = {
  init() {
    this.queryInput = document.getElementById("benchmark-query-input");
    this.runBtn = document.getElementById("run-benchmark-btn");
    this.resultsContainer = document.getElementById("benchmark-results-container");
    this.tableBody = document.getElementById("benchmark-table-body");

    if (this.runBtn) {
      this.runBtn.addEventListener("click", () => this.runComparativeBenchmark());
    }
  },

  async runComparativeBenchmark() {
    const query = this.queryInput.value.trim();
    if (!query) return;

    this.runBtn.disabled = true;
    this.runBtn.innerHTML = `Running 4-way benchmark...`;

    try {
      const data = await API.runBenchmark(query);
      const results = data.results || {};

      this.tableBody.innerHTML = "";
      this.resultsContainer.style.display = "block";

      const strategyNames = {
        hybrid_reranked: "Hybrid + Cross-Encoder",
        hybrid: "Hybrid RRF",
        dense_only: "Dense Vector Only",
        bm25_only: "Sparse BM25 Only",
      };

      for (const [key, res] of Object.entries(results)) {
        const row = document.createElement("tr");

        const scoreClass =
          res.composite_score >= 0.75
            ? "score-high"
            : res.composite_score >= 0.55
            ? "score-mid"
            : "score-low";

        row.innerHTML = `
          <td><strong>${strategyNames[key] || key}</strong></td>
          <td><code>${res.latency_ms} ms</code></td>
          <td>${res.top_docs.join(", ") || "None"}</td>
          <td>${(res.context_relevance * 100).toFixed(0)}%</td>
          <td>${(res.faithfulness * 100).toFixed(0)}%</td>
          <td><span class="score-badge ${scoreClass}">${(res.composite_score * 100).toFixed(0)} / 100</span></td>
        `;

        this.tableBody.appendChild(row);
      }
    } catch (err) {
      alert("Benchmark failed: " + err.message);
    } finally {
      this.runBtn.disabled = false;
      this.runBtn.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
        Run Benchmark
      `;
    }
  },
};
