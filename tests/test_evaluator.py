from backend.app.services.llm.evaluator import RAGTriadEvaluator

def test_rag_triad_evaluator_high_score():
    query = "How does Raft leader election prevent split-brain?"
    context = "In the Raft protocol, leader election is designed to prevent split-brain by requiring candidates to win votes from a strict majority (N/2 + 1) of cluster servers using randomized election timeouts."
    answer = "Raft prevents split-brain by requiring a strict majority of server votes (N/2 + 1) during leader election and using randomized election timeouts [1]."

    scores = RAGTriadEvaluator.evaluate(query, context, answer)
    assert scores["context_relevance"] > 0.6
    assert scores["faithfulness"] > 0.7
    assert scores["answer_relevance"] > 0.6
    assert scores["composite_score"] > 0.6

def test_rag_triad_evaluator_hallucination_penalty():
    query = "How does Raft elect leaders?"
    context = "Raft uses randomized election timeouts and votes."
    hallucinated_answer = "Raft uses blockchain proof-of-work mining with SHA-256 cryptographic hashes and quadratic staking rewards."

    scores = RAGTriadEvaluator.evaluate(query, context, hallucinated_answer)
    # Faithfulness must be low because none of the claims exist in the context
    assert scores["faithfulness"] < 0.3
