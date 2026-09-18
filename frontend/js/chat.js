/**
 * Chat & Conversational RAG Copilot Controller
 */
const ChatController = {
  currentSessionId: null,
  activeCitations: {},

  init() {
    this.messagesContainer = document.getElementById("chat-messages-container");
    this.welcomeHero = document.getElementById("chat-welcome-hero");
    this.chatInput = document.getElementById("chat-input");
    this.sendBtn = document.getElementById("send-chat-btn");
    this.strategySelect = document.getElementById("strategy-selector");
    this.sessionList = document.getElementById("chat-session-list");
    this.newChatBtn = document.getElementById("new-chat-btn");

    this.bindEvents();
    this.loadSessions();
  },

  bindEvents() {
    this.sendBtn.addEventListener("click", () => this.handleSendMessage());

    this.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        this.handleSendMessage();
      }
    });

    // Auto-grow textarea
    this.chatInput.addEventListener("input", () => {
      this.chatInput.style.height = "auto";
      this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 140) + "px";
    });

    this.newChatBtn.addEventListener("click", () => {
      this.currentSessionId = null;
      this.clearMessages();
      this.updateActiveSessionUI();
    });

    // Quick prompt buttons
    document.querySelectorAll(".quick-prompt-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const query = btn.getAttribute("data-query");
        if (query) {
          this.chatInput.value = query;
          this.handleSendMessage();
        }
      });
    });
  },

  async loadSessions() {
    try {
      const sessions = await API.listSessions();
      this.sessionList.innerHTML = "";
      if (!sessions || sessions.length === 0) {
        this.sessionList.innerHTML = `<div style="font-size:0.78rem; color:var(--text-muted); padding:8px;">No past sessions</div>`;
        return;
      }
      sessions.forEach((s) => {
        const item = document.createElement("div");
        item.className = `session-item ${s.id === this.currentSessionId ? "active" : ""}`;
        item.textContent = s.title || "Untitled Session";
        item.addEventListener("click", () => this.switchSession(s.id));
        this.sessionList.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load sessions:", err);
    }
  },

  async switchSession(sessionId) {
    this.currentSessionId = sessionId;
    this.updateActiveSessionUI();
    try {
      const messages = await API.getSessionMessages(sessionId);
      this.clearMessages();
      if (messages.length === 0) {
        this.welcomeHero.style.display = "block";
      } else {
        this.welcomeHero.style.display = "none";
        messages.forEach((m) => {
          this.appendMessageRow(m.role, m.content, m.citations, m.trace);
        });
      }
    } catch (err) {
      console.error("Failed to fetch session messages:", err);
    }
  },

  updateActiveSessionUI() {
    const items = this.sessionList.querySelectorAll(".session-item");
    items.forEach((item) => {
      item.classList.remove("active");
    });
  },

  clearMessages() {
    this.messagesContainer.innerHTML = "";
    this.messagesContainer.appendChild(this.welcomeHero);
    this.welcomeHero.style.display = "block";
  },

  async handleSendMessage() {
    const query = this.chatInput.value.trim();
    if (!query) return;

    const strategy = this.strategySelect.value;
    this.chatInput.value = "";
    this.chatInput.style.height = "auto";
    this.welcomeHero.style.display = "none";

    // 1. Render User Message
    this.appendMessageRow("user", query);

    // 2. Prepare Assistant Bubble
    const { bubbleEl, citationsTrayEl, traceBtnEl } = this.createAssistantBubble();
    this.sendBtn.disabled = true;

    try {
      // Use SSE streaming via fetch
      const response = await fetch("/api/v1/rag/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          strategy,
          session_id: this.currentSessionId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let accumulatedText = "";
      let currentTrace = null;
      let currentCitations = [];

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split("\n\n");
        buffer = events.pop() || "";

        for (const evt of events) {
          if (!evt.trim()) continue;
          const lines = evt.split("\n");
          let eventType = "message";
          let eventData = "";

          for (const line of lines) {
            if (line.startsWith("event: ")) {
              eventType = line.replace("event: ", "").trim();
            } else if (line.startsWith("data: ")) {
              eventData = line.replace("data: ", "").trim();
            }
          }

          if (eventType === "trace") {
            try {
              currentTrace = JSON.parse(eventData);
            } catch (e) {}
          } else if (eventType === "token") {
            try {
              const parsed = JSON.parse(eventData);
              accumulatedText += parsed.token;
              bubbleEl.innerHTML = this.renderMarkdownWithCitations(accumulatedText);
              this.scrollToBottom();
            } catch (e) {}
          } else if (eventType === "citations") {
            try {
              currentCitations = JSON.parse(eventData);
              this.renderCitations(citationsTrayEl, currentCitations);
            } catch (e) {}
          } else if (eventType === "done") {
            try {
              const doneData = JSON.parse(eventData);
              if (doneData.session_id) {
                this.currentSessionId = doneData.session_id;
              }
            } catch (e) {}
            if (currentTrace) {
              traceBtnEl.style.display = "flex";
              traceBtnEl.addEventListener("click", () => {
                TraceInspector.showTrace(currentTrace);
              });
            }
            await this.loadSessions();
          }
        }
      }

      // Re-highlight code
      bubbleEl.querySelectorAll("pre code").forEach((el) => {
        hljs.highlightElement(el);
      });
    } catch (err) {
      console.error("Query failed, falling back to non-stream query:", err);
      try {
        const fallbackRes = await API.queryRag(query, strategy, this.currentSessionId);
        this.currentSessionId = fallbackRes.session_id;
        bubbleEl.innerHTML = this.renderMarkdownWithCitations(fallbackRes.answer);
        if (fallbackRes.citations) {
          this.renderCitations(citationsTrayEl, fallbackRes.citations);
        }
        if (fallbackRes.trace) {
          traceBtnEl.style.display = "flex";
          traceBtnEl.addEventListener("click", () => {
            TraceInspector.showTrace(fallbackRes.trace);
          });
        }
        this.loadSessions();
      } catch (e2) {
        bubbleEl.innerHTML = `<span style="color:var(--accent-rose)">Error generating response: ${err.message}</span>`;
      }
    } finally {
      this.sendBtn.disabled = false;
      this.scrollToBottom();
    }
  },

  appendMessageRow(role, content, citations = [], trace = {}) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatar = document.createElement("div");
    avatar.className = `avatar ${role}`;
    avatar.textContent = role === "user" ? "You" : "🧠";

    const body = document.createElement("div");
    body.className = "message-body";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = this.renderMarkdownWithCitations(content);

    body.appendChild(bubble);

    if (role === "assistant") {
      const citationsTray = document.createElement("div");
      citationsTray.className = "citations-tray";
      if (citations && citations.length > 0) {
        this.renderCitations(citationsTray, citations);
      }
      body.appendChild(citationsTray);

      if (trace && Object.keys(trace).length > 0) {
        const traceBtn = document.createElement("button");
        traceBtn.className = "trace-button-inline";
        traceBtn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
          Inspect Retrieval Trace (${trace.latencies_ms?.total_retrieval || 0}ms)
        `;
        traceBtn.addEventListener("click", () => TraceInspector.showTrace(trace));
        body.appendChild(traceBtn);
      }
    }

    if (role === "user") {
      row.appendChild(body);
      row.appendChild(avatar);
    } else {
      row.appendChild(avatar);
      row.appendChild(body);
    }

    this.messagesContainer.appendChild(row);
    this.scrollToBottom();

    // Code highlighting
    bubble.querySelectorAll("pre code").forEach((el) => {
      hljs.highlightElement(el);
    });
  },

  createAssistantBubble() {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const avatar = document.createElement("div");
    avatar.className = "avatar assistant";
    avatar.textContent = "🧠";

    const body = document.createElement("div");
    body.className = "message-body";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = `<span style="color:var(--text-muted)">Searching your Second Brain notes...</span>`;

    const citationsTray = document.createElement("div");
    citationsTray.className = "citations-tray";

    const traceBtn = document.createElement("button");
    traceBtn.className = "trace-button-inline";
    traceBtn.style.display = "none";
    traceBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
      Inspect Retrieval Trace
    `;

    body.appendChild(bubble);
    body.appendChild(citationsTray);
    body.appendChild(traceBtn);

    row.appendChild(avatar);
    row.appendChild(body);
    this.messagesContainer.appendChild(row);
    this.scrollToBottom();

    return { bubbleEl: bubble, citationsTrayEl: citationsTray, traceBtnEl: traceBtn };
  },

  renderMarkdownWithCitations(rawText) {
    if (!rawText) return "";
    let html = marked.parse(rawText);

    // Replace [1], [2], etc. with interactive citation badge buttons
    html = html.replace(/\[(\d+)\]/g, (match, p1) => {
      return `<button class="citation-pill" onclick="ChatController.inspectCitation(${p1})">[${p1}]</button>`;
    });

    return html;
  },

  renderCitations(trayEl, citations) {
    trayEl.innerHTML = "";
    if (!citations || citations.length === 0) return;

    this.activeCitations = {};
    citations.forEach((c) => {
      this.activeCitations[c.citation_id] = c;
      const chip = document.createElement("div");
      chip.className = "source-chip";
      chip.innerHTML = `
        <span class="source-badge">[${c.citation_id}]</span>
        <span>${c.doc_title}</span>
        <span style="color:var(--text-muted); font-size:0.7rem;">(${Math.round((c.score || 0) * 100)}%)</span>
      `;
      chip.addEventListener("click", () => this.inspectCitation(c.citation_id));
      trayEl.appendChild(chip);
    });
  },

  inspectCitation(citationId) {
    const citation = this.activeCitations[citationId];
    if (!citation) return;
    App.openDrawer(`Source [${citationId}]: ${citation.doc_title}`, `
      <div style="display:flex; flex-direction:column; gap:14px;">
        <div style="background:var(--bg-surface-elevated); padding:12px; border-radius:8px; border:1px solid var(--border-subtle);">
          <div style="font-size:0.8rem; color:var(--text-muted);">SECTION BREADCRUMB</div>
          <div style="font-weight:600; font-size:0.95rem; color:var(--accent-secondary);">${citation.header_path || "General"}</div>
          <div style="display:flex; gap:12px; margin-top:8px; font-size:0.78rem; color:var(--text-muted);">
            <div>Score: <strong>${citation.score || "N/A"}</strong></div>
            <div>Doc ID: <code>${citation.document_id?.slice(0, 8)}</code></div>
          </div>
        </div>
        <div>
          <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:6px;">RETRIEVED CONTEXT SNIPPET</div>
          <div style="background:var(--bg-base); padding:14px; border-radius:8px; border:1px solid var(--border-subtle); font-size:0.88rem; line-height:1.6; white-space:pre-wrap;">${citation.snippet}</div>
        </div>
        <button class="btn-secondary" onclick="DocumentsController.viewDocument('${citation.document_id}')" style="justify-content:center;">
          Open Full Document
        </button>
      </div>
    `);
  },

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  },
};
