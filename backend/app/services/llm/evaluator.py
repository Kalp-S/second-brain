import re


class RAGTriadEvaluator:
    """
    Automated RAG Triad evaluation:
    1. Context Relevance: Did the retrieval bring relevant passages or noise?
    2. Faithfulness / Groundedness: Are claims in the answer grounded in context?
    3. Answer Relevance: Does the generated response directly answer the query?
    """

    @staticmethod
    def evaluate(query: str, retrieved_context: str, generated_answer: str) -> dict[str, float]:
        if not retrieved_context.strip() or not generated_answer.strip():
            return {
                "context_relevance": 0.0,
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "composite_score": 0.0,
            }

        # 1. Context Relevance
        context_relevance = RAGTriadEvaluator._calc_context_relevance(query, retrieved_context)

        # 2. Faithfulness / Groundedness
        faithfulness = RAGTriadEvaluator._calc_faithfulness(retrieved_context, generated_answer)

        # 3. Answer Relevance
        answer_relevance = RAGTriadEvaluator._calc_answer_relevance(query, generated_answer)

        composite = round(
            (context_relevance * 0.3 + faithfulness * 0.4 + answer_relevance * 0.3), 3
        )

        return {
            "context_relevance": round(context_relevance, 3),
            "faithfulness": round(faithfulness, 3),
            "answer_relevance": round(answer_relevance, 3),
            "composite_score": composite,
        }

    @staticmethod
    def _tokenize(text: str) -> set:
        return set(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower()))

    @staticmethod
    def _calc_context_relevance(query: str, context: str) -> float:
        q_tokens = RAGTriadEvaluator._tokenize(query)
        if not q_tokens:
            return 1.0
        c_tokens = RAGTriadEvaluator._tokenize(context)
        overlap = len(q_tokens.intersection(c_tokens))
        return min(1.0, (overlap / len(q_tokens)) * 1.2)

    @staticmethod
    def _calc_faithfulness(context: str, answer: str) -> float:
        """Checks the proportion of key answer claims supported by context tokens."""
        c_tokens = RAGTriadEvaluator._tokenize(context)
        sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        supported_sentences = 0
        for sent in sentences:
            sent_tokens = RAGTriadEvaluator._tokenize(sent)
            if not sent_tokens:
                supported_sentences += 1
                continue
            overlap = len(sent_tokens.intersection(c_tokens))
            ratio = overlap / len(sent_tokens)
            if ratio >= 0.45:  # At least 45% token presence indicates grounded claim
                supported_sentences += 1

        return supported_sentences / len(sentences)

    @staticmethod
    def _calc_answer_relevance(query: str, answer: str) -> float:
        q_tokens = RAGTriadEvaluator._tokenize(query)
        a_tokens = RAGTriadEvaluator._tokenize(answer)
        if not q_tokens:
            return 1.0
        overlap = len(q_tokens.intersection(a_tokens))
        # Penalty if answer is extremely short or a refusal
        if len(answer.split()) < 5:
            return 0.2
        return min(1.0, (overlap / len(q_tokens)) * 1.1)
