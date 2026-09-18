from backend.app.services.ingestion.chunker import HierarchicalChunker


def test_hierarchical_chunker_basic():
    chunker = HierarchicalChunker(parent_max_chars=500, child_max_chars=120, child_overlap=20)

    sample_text = """# Distributed Consensus

Raft is a consensus protocol that manages a replicated log across nodes.
It decomposes the problem into leader election, log replication, and safety.

## Leader Election

When a follower election timeout expires, it increments its term and becomes a candidate.
Candidates broadcast RequestVote RPCs to all peers in the cluster.
"""

    parents = chunker.chunk_document("Raft Protocol", sample_text)

    assert len(parents) >= 2
    assert parents[0].header_path == "Distributed Consensus"
    assert parents[1].header_path == "Leader Election"

    # Check that children have contextual enrichment header
    first_parent = parents[0]
    assert len(first_parent.children) >= 1
    child = first_parent.children[0]
    assert "[Document: Raft Protocol" in child.contextual_content
    assert "Distributed Consensus" in child.contextual_content


def test_chunker_empty_input():
    chunker = HierarchicalChunker()
    parents = chunker.chunk_document("Empty", "")
    assert parents == []
