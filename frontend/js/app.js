/**
 * Second Brain Main App Orchestrator
 */
const App = {
  init() {
    this.bindNavigation();
    this.bindDrawer();
    this.updateStatus();

    // Initialize sub-controllers
    ChatController.init();
    DocumentsController.init();
    GraphController.init();
    BenchmarkStudio.init();
  },

  bindNavigation() {
    const navButtons = document.querySelectorAll(".nav-tab");
    const panes = document.querySelectorAll(".tab-pane");

    navButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");

        navButtons.forEach((b) => b.classList.remove("active"));
        panes.forEach((p) => p.classList.remove("active"));

        btn.classList.add("active");
        const activePane = document.getElementById(targetTab);
        if (activePane) {
          activePane.classList.add("active");

          // Trigger dynamic re-renders if needed
          if (targetTab === "tab-graph") {
            GraphController.renderGraph();
          } else if (targetTab === "tab-vault") {
            DocumentsController.loadDocuments();
          }
        }
      });
    });
  },

  bindDrawer() {
    this.drawer = document.getElementById("slide-drawer");
    this.backdrop = document.getElementById("drawer-backdrop");
    this.drawerTitle = document.getElementById("drawer-title");
    this.drawerBody = document.getElementById("drawer-body");
    this.drawerClose = document.getElementById("drawer-close-btn");

    this.drawerClose.addEventListener("click", () => this.closeDrawer());
    this.backdrop.addEventListener("click", () => this.closeDrawer());

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") this.closeDrawer();
    });
  },

  openDrawer(title, htmlContent) {
    this.drawerTitle.textContent = title;
    this.drawerBody.innerHTML = htmlContent;
    this.backdrop.classList.add("open");
    this.drawer.classList.add("open");
  },

  closeDrawer() {
    this.backdrop.classList.remove("open");
    this.drawer.classList.remove("open");
  },

  async updateStatus() {
    try {
      const status = await API.getSystemStatus();
      const dot = document.getElementById("ollama-dot");
      const text = document.getElementById("ollama-text");
      const docCount = document.getElementById("doc-count-display");

      if (status.ollama_status === "connected") {
        dot.className = "status-dot";
        text.textContent = `Ollama: ${status.active_llm_model}`;
      } else {
        dot.className = "status-dot offline";
        text.textContent = `LLM: ${status.llm_provider} (${status.ollama_status})`;
      }

      if (status.vault_statistics) {
        docCount.textContent = `${status.vault_statistics.documents} Docs (${status.vault_statistics.child_chunks} Chunks)`;
      }
    } catch (err) {
      console.warn("Status check failed:", err);
    }
  },
};

// Start application on DOMContentLoaded
document.addEventListener("DOMContentLoaded", () => {
  App.init();
});
