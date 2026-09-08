/**
 * Interactive Knowledge Graph Controller using Vis.js
 */
const GraphController = {
  network: null,
  container: null,

  init() {
    this.container = document.getElementById("network-graph");
    this.fitBtn = document.getElementById("graph-fit-btn");

    if (this.fitBtn) {
      this.fitBtn.addEventListener("click", () => {
        if (this.network) this.network.fit({ animation: true });
      });
    }
  },

  async renderGraph() {
    if (!this.container) return;

    try {
      const data = await API.getKnowledgeGraph();
      if (!data.nodes || data.nodes.length === 0) {
        this.container.innerHTML = `
          <div style="display:flex; height:100%; align-items:center; justify-content:center; color:var(--text-muted);">
            No nodes in knowledge graph. Index documents in the vault to build the graph.
          </div>
        `;
        return;
      }

      const visNodes = new vis.DataSet(
        data.nodes.map((n) => {
          const isDoc = n.group === "document";
          return {
            id: n.id,
            label: n.label,
            title: n.title,
            shape: isDoc ? "dot" : "box",
            size: isDoc ? (n.value || 20) : 14,
            color: {
              background: isDoc ? "#6366f1" : "#06b6d4",
              border: isDoc ? "#818cf8" : "#22d3ee",
              highlight: {
                background: "#f43f5e",
                border: "#fda4af",
              },
            },
            font: {
              color: "#f8fafc",
              face: "Inter",
              size: isDoc ? 13 : 11,
            },
            docId: n.doc_id,
          };
        })
      );

      const visEdges = new vis.DataSet(
        data.edges.map((e) => ({
          id: e.id,
          from: e.from,
          to: e.to,
          color: {
            color: e.label === "references" ? "rgba(168, 85, 247, 0.6)" : "rgba(255, 255, 255, 0.15)",
            highlight: "#06b6d4",
          },
          width: e.label === "references" ? 2 : 1,
          arrows: e.label === "references" ? "to" : "",
        }))
      );

      const options = {
        nodes: {
          borderWidth: 2,
          shadow: true,
        },
        edges: {
          smooth: {
            type: "continuous",
          },
        },
        physics: {
          stabilization: { iterations: 150 },
          barnesHut: {
            gravitationalConstant: -3000,
            springConstant: 0.04,
            springLength: 120,
          },
        },
        interaction: {
          hover: true,
          tooltipDelay: 100,
          zoomView: true,
        },
      };

      this.network = new vis.Network(this.container, { nodes: visNodes, edges: visEdges }, options);

      this.network.on("click", (params) => {
        if (params.nodes.length > 0) {
          const clickedNodeId = params.nodes[0];
          const nodeData = visNodes.get(clickedNodeId);
          if (nodeData && nodeData.docId) {
            DocumentsController.viewDocument(nodeData.docId);
          }
        }
      });
    } catch (err) {
      console.error("Failed to render knowledge graph:", err);
    }
  },
};
