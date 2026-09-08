/**
 * Knowledge Vault & Document Management Controller
 */
const DocumentsController = {
  init() {
    this.docGrid = document.getElementById("document-grid-container");
    this.dropzone = document.getElementById("upload-dropzone");
    this.fileInput = document.getElementById("file-upload-input");
    this.syncBtn = document.getElementById("sync-vault-btn");

    this.bindEvents();
    this.loadDocuments();
  },

  bindEvents() {
    this.syncBtn.addEventListener("click", async () => {
      this.syncBtn.disabled = true;
      this.syncBtn.innerHTML = `Syncing...`;
      try {
        const res = await API.syncVault();
        alert(`Vault synced! Indexed ${res.synced_count} documents.`);
        await this.loadDocuments();
        App.updateStatus();
      } catch (err) {
        alert("Failed to sync vault: " + err.message);
      } finally {
        this.syncBtn.disabled = false;
        this.syncBtn.innerHTML = `
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>
          Sync Sample Vault
        `;
      }
    });

    this.fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        this.handleUpload(e.target.files);
      }
    });

    // Drag and Drop
    this.dropzone.addEventListener("click", () => this.fileInput.click());

    ["dragenter", "dragover"].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      this.dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        this.dropzone.classList.remove("dragover");
      });
    });

    this.dropzone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files.length > 0) {
        this.handleUpload(dt.files);
      }
    });
  },

  async handleUpload(fileList) {
    try {
      const res = await API.uploadFiles(fileList);
      alert(`Uploaded and indexed ${res.uploaded} file(s).`);
      await this.loadDocuments();
      App.updateStatus();
    } catch (err) {
      alert("Upload failed: " + err.message);
    }
  },

  async loadDocuments() {
    try {
      const docs = await API.listDocuments();
      this.docGrid.innerHTML = "";

      if (!docs || docs.length === 0) {
        this.docGrid.innerHTML = `
          <div style="grid-column: 1 / -1; text-align:center; padding:40px; color:var(--text-muted);">
            No documents in vault yet. Click "Sync Sample Vault" or drop markdown files above.
          </div>
        `;
        return;
      }

      docs.forEach((d) => {
        const card = document.createElement("div");
        card.className = "doc-card";

        const tagsHtml = d.tags.map((t) => `<span class="tag-pill">#${t}</span>`).join(" ");

        card.innerHTML = `
          <div class="doc-card-header">
            <div class="doc-card-title">${d.title}</div>
            <span class="doc-badge">${d.file_type}</span>
          </div>
          <div class="doc-stats">
            <span>💾 ${(d.byte_size / 1024).toFixed(1)} KB</span>
            <span>📑 ${d.num_parents} Parents</span>
            <span>🧩 ${d.num_children} Chunks</span>
          </div>
          <div class="doc-tags">${tagsHtml || '<span style="color:var(--text-muted); font-size:0.75rem;">No tags</span>'}</div>
          <div class="doc-card-footer">
            <button class="btn-card-action" onclick="DocumentsController.viewDocument('${d.id}')">View Content</button>
            <button class="btn-card-action btn-card-delete" onclick="DocumentsController.deleteDoc('${d.id}', '${d.title.replace(/'/g, "\\'")}')">Delete</button>
          </div>
        `;

        this.docGrid.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load documents:", err);
    }
  },

  async viewDocument(docId) {
    try {
      const doc = await API.getDocument(docId);
      const parsedContent = marked.parse(doc.raw_content || "*(No raw content)*");

      App.openDrawer(doc.title, `
        <div style="display:flex; flex-direction:column; gap:16px;">
          <div style="background:var(--bg-surface-elevated); padding:12px; border-radius:8px; border:1px solid var(--border-subtle);">
            <div style="font-size:0.8rem; color:var(--text-muted);">METADATA</div>
            <div style="font-size:0.85rem; margin-top:4px;">File: <code>${doc.filename}</code> (${(doc.byte_size / 1024).toFixed(1)} KB)</div>
            <div style="font-size:0.85rem; margin-top:4px;">Parent Sections: ${doc.parents?.length || 0}</div>
          </div>
          <div class="message-bubble" style="background:var(--bg-base); border:1px solid var(--border-subtle); padding:16px;">
            ${parsedContent}
          </div>
        </div>
      `);

      // Highlight code blocks
      document.querySelectorAll("#drawer-body pre code").forEach((el) => {
        hljs.highlightElement(el);
      });
    } catch (err) {
      alert("Failed to fetch document: " + err.message);
    }
  },

  async deleteDoc(docId, title) {
    if (!confirm(`Are you sure you want to delete "${title}" and all its vector/BM25 chunks?`)) {
      return;
    }
    try {
      await API.deleteDocument(docId);
      await this.loadDocuments();
      App.updateStatus();
    } catch (err) {
      alert("Failed to delete document: " + err.message);
    }
  },
};
