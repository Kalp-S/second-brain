/**
 * Second Brain API Client
 */
const API = {
  baseUrl: "/api/v1",

  async getSystemStatus() {
    const res = await fetch(`${this.baseUrl}/system/status`);
    return await res.json();
  },

  async listDocuments() {
    const res = await fetch(`${this.baseUrl}/documents`);
    return await res.json();
  },

  async getDocument(docId) {
    const res = await fetch(`${this.baseUrl}/documents/${docId}`);
    return await res.json();
  },

  async deleteDocument(docId) {
    const res = await fetch(`${this.baseUrl}/documents/${docId}`, { method: "DELETE" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Delete failed with status ${res.status}`);
    }
    return await res.json();
  },

  async syncVault() {
    const res = await fetch(`${this.baseUrl}/documents/sync-vault`, { method: "POST" });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Sync failed with status ${res.status}`);
    }
    return await res.json();
  },

  async uploadFiles(fileList) {
    const formData = new FormData();
    for (const file of fileList) {
      formData.append("files", file);
    }
    const res = await fetch(`${this.baseUrl}/documents/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }
    return await res.json();
  },

  async listSessions() {
    const res = await fetch(`${this.baseUrl}/rag/sessions`);
    return await res.json();
  },

  async getSessionMessages(sessionId) {
    const res = await fetch(`${this.baseUrl}/rag/sessions/${sessionId}/messages`);
    return await res.json();
  },

  async queryRag(query, strategy = "hybrid_reranked", sessionId = null) {
    const res = await fetch(`${this.baseUrl}/rag/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        strategy,
        session_id: sessionId,
      }),
    });
    return await res.json();
  },

  async getKnowledgeGraph() {
    const res = await fetch(`${this.baseUrl}/graph`);
    return await res.json();
  },

  async runBenchmark(query) {
    const res = await fetch(`${this.baseUrl}/evaluation/benchmark`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    return await res.json();
  },
};
