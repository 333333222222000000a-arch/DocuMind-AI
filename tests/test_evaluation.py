import json
import unittest
from pathlib import Path

from rag_app.evaluation import (
    evaluate_retrieval_case,
    evaluate_retrieval_cases,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    unsupported_numeric_claims,
)


DATASET_PATH = Path(__file__).parent / "data" / "rag_eval_cases.json"


class RetrievalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        cls.cases = cls.dataset["cases"]

    def test_dataset_covers_requested_question_categories(self):
        categories = {case["category"] for case in self.cases}
        self.assertEqual(categories, {
            "direct_fact", "multi_passage", "missing_answer", "arabic_question",
            "english_question", "multiple_pdfs", "conflicting_evidence",
        })

    def test_precision_recall_and_mrr_are_deterministic(self):
        ranked = ["relevant-a", "noise", "relevant-b"]
        relevant = {"relevant-a", "relevant-b"}
        self.assertAlmostEqual(precision_at_k(ranked, relevant, 3), 2 / 3)
        self.assertEqual(recall_at_k(ranked, relevant, 3), 1.0)
        self.assertEqual(reciprocal_rank(ranked, relevant), 1.0)
        self.assertIsNone(recall_at_k(ranked, set(), 3))
        self.assertEqual(reciprocal_rank(ranked, set()), 0.0)

    def test_citation_metrics_use_expected_source_and_page_pairs(self):
        case = next(case for case in self.cases if case["id"] == "multiple_pdf_evidence")
        result = evaluate_retrieval_case(case, ["net_switch", "storage"], k=2)
        self.assertEqual(result["citation_precision"], 1.0)
        self.assertEqual(result["citation_recall"], 1.0)

    def test_dataset_evaluation_reports_unanswerable_retrievals(self):
        def deterministic_retriever(question, chunks):
            if "warranty" in question:
                return ["absent_cpu"]
            return [chunk["chunk_id"] for chunk in chunks]

        results = evaluate_retrieval_cases(self.cases, deterministic_retriever, k=3)
        self.assertEqual(len(results["case_results"]), len(self.cases))
        self.assertEqual(results["unanswerable_retrieval_rate"], 1.0)
        self.assertGreater(results["mean_recall_at_k"], 0.0)

    def test_numeric_grounding_check_detects_unsupported_numbers(self):
        evidence = "The server has an i5-14400F processor, RTX 5060 Ti 16GB, and 32GB RAM."
        supported = "i5-14400F, RTX 5060 Ti 16GB, and 32GB RAM [S1]."
        unsupported = "The server has 128GB RAM [S1]."
        self.assertEqual(unsupported_numeric_claims(supported, evidence), [])
        self.assertEqual(unsupported_numeric_claims(unsupported, evidence), ["128"])


if __name__ == "__main__":
    unittest.main()
