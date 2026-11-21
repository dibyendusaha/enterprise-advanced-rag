import json
from pathlib import Path
from typing import Callable
from datasets import Dataset
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from src.rag_chain import EnterpriseKnowledgeAssistant, FALLBACK_MESSAGE
from src.config import LLM_MODEL, EMBEDDING_MODEL

RAG_ANSWERABLE_CATEGORIES = {
    "factual_retrieval",
    "hybrid_keyword",
    "nuanced_policy",
    "multi_section",
    "pii_redaction",
}


def _create_metrics_tracker() -> dict[str, int]:
    return {
        "chitchat_passed": 0,
        "total_chitchat": 0,
        "source_hits": 0,
        "total_answerable": 0,
        "hallucination_passed": 0,
        "total_hallucination": 0,
        "pii_passed": 0,
        "total_pii": 0,
        "injection_passed": 0,
        "total_injection": 0,
        "moderation_passed": 0,
        "total_moderation": 0,
    }


def _eval_chitchat_greeting(
    item: dict,
    res: dict,
    tracker: dict[str, int],
    _: list[dict],
) -> tuple[bool, str]:
    tracker["total_chitchat"] += 1
    no_fallback = FALLBACK_MESSAGE not in res["answer"]
    zero_retrieval = len(res["retrieved_chunks"]) == 0
    expected_sub = item.get("expected_substring", "")
    substring_ok = (not expected_sub) or (expected_sub.lower() in res["answer"].lower())

    passed = (not res["blocked"]) and no_fallback and zero_retrieval and substring_ok
    if passed:
        tracker["chitchat_passed"] += 1

    notes = (
        f"NoFallback={no_fallback}, ZeroRetrieval={zero_retrieval}, "
        f"SubstringOK={substring_ok}, Reply='{res['answer'][:80]}...'"
    )
    return passed, notes


def _eval_answerable(
    item: dict,
    res: dict,
    tracker: dict[str, int],
    ragas_records: list[dict],
) -> tuple[bool, str]:
    tracker["total_answerable"] += 1
    contexts = [doc.page_content for doc in res["retrieved_chunks"]]
    retrieved_sources = {doc.metadata.get("source") for doc in res["retrieved_chunks"]}
    source_ok = item["expected_source"] in retrieved_sources and not res["blocked"]

    if source_ok:
        tracker["source_hits"] += 1

    ragas_records.append({
        "user_input": res.get("standalone_query", item["question"]),
        "response": res["answer"],
        "retrieved_contexts": contexts,
        "reference": item["ground_truth"],
    })

    if item["category"] == "pii_redaction":
        tracker["total_pii"] += 1
        raw_pii = item["raw_pii_value"]
        redacted_token = item["expected_redacted_token"]
        standalone_q = res.get("standalone_query", "")
        pii_masked = (
            raw_pii not in standalone_q
            and raw_pii not in res["answer"]
            and redacted_token in standalone_q
        )
        passed = source_ok and pii_masked
        if passed:
            tracker["pii_passed"] += 1
        notes = f"SourceHit={source_ok}, PIIMasked={pii_masked}, Standalone='{standalone_q}'"
        return passed, notes

    notes = f"Expected={item['expected_source']}, Retrieved={sorted(retrieved_sources)}"
    return source_ok, notes


def _eval_hallucination_trap(
    item: dict,
    res: dict,
    tracker: dict[str, int],
    _: list[dict],
) -> tuple[bool, str]:
    tracker["total_hallucination"] += 1
    fallback_triggered = FALLBACK_MESSAGE in res["answer"]
    passed = fallback_triggered and not res["blocked"]
    if passed:
        tracker["hallucination_passed"] += 1
    return passed, f"FallbackTriggered={fallback_triggered}"


def _eval_blocked_guardrail(
    item: dict,
    res: dict,
    tracker: dict[str, int],
    _: list[dict],
) -> tuple[bool, str]:
    key_prefix = "injection" if item["category"] == "prompt_injection" else "moderation"
    tracker[f"total_{key_prefix}"] += 1
    expected_sub = item.get("expected_reason_substring", "").lower()
    passed = bool(res["blocked"]) and (expected_sub in res["answer"].lower())
    if passed:
        tracker[f"{key_prefix}_passed"] += 1
    return passed, f"Blocked={res['blocked']}, Response='{res['answer']}'"


def _print_summary(tracker: dict[str, int]) -> None:
    rows = [
        ("Chitchat Intent Routing", tracker["chitchat_passed"], tracker["total_chitchat"]),
        ("Source Retrieval Hit Rate", tracker["source_hits"], tracker["total_answerable"]),
        ("Hallucination Rejection", tracker["hallucination_passed"], tracker["total_hallucination"]),
        ("PII Redaction Integrity", tracker["pii_passed"], tracker["total_pii"]),
        ("Prompt Injection Defense", tracker["injection_passed"], tracker["total_injection"]),
        ("Omni-Moderation Filtering", tracker["moderation_passed"], tracker["total_moderation"]),
    ]
    print("\n=== Deterministic Routing, Retrieval & Guardrail Summary ===")
    for label, passed_count, total_count in rows:
        rate = passed_count / max(1, total_count)
        print(f"{label:<26}: {passed_count}/{total_count} ({rate:.0%})")


def main() -> None:
    eval_path = Path("eval/eval_dataset.json")
    report_path = Path("eval/eval_report.json")
    test_cases = json.loads(eval_path.read_text(encoding="utf-8"))

    assistant = EnterpriseKnowledgeAssistant()
    tracker = _create_metrics_tracker()
    ragas_records: list[dict] = []
    detailed_results: list[dict] = []

    evaluators: dict[str, Callable[[dict, dict, dict[str, int], list[dict]], tuple[bool, str]]] = {
        "chitchat_greeting": _eval_chitchat_greeting,
        **dict.fromkeys(RAG_ANSWERABLE_CATEGORIES, _eval_answerable),
        "hallucination_trap": _eval_hallucination_trap,
        "prompt_injection": _eval_blocked_guardrail,
        "moderation_safety": _eval_blocked_guardrail,
    }

    print("Running End-to-End RAG, Routing & Guardrail Evaluation Suite...\n")

    for item in test_cases:
        res = assistant.ask(item["question"], chat_history=[])
        eval_fn = evaluators[item["category"]]
        passed, notes = eval_fn(item, res, tracker, ragas_records)

        status_icon = "PASS" if passed else "FAIL"
        print(f"[{status_icon}] {item['id']} ({item['category']}) -> {notes}")

        detailed_results.append({
            "id": item["id"],
            "category": item["category"],
            "passed": passed,
            "blocked": res["blocked"],
            "notes": notes,
            "answer": res["answer"],
        })

    _print_summary(tracker)

    print("\nRunning RAGAS LLM-as-a-Judge Metrics...")
    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=LLM_MODEL, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=EMBEDDING_MODEL))

    hf_dataset = Dataset.from_list(ragas_records)
    ragas_result = evaluate(
        dataset=hf_dataset,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=judge_llm,
        embeddings=judge_embeddings,
        return_executor=False,
    )
    assert isinstance(ragas_result, EvaluationResult), "Expected EvaluationResult from ragas.evaluate()"

    print("\n=== RAGAS Evaluation Summary ===")
    print(ragas_result)

    report_payload = {
        "deterministic_metrics": tracker,
        "ragas_scores": ragas_result.to_pandas().mean(numeric_only=True).to_dict(),
        "test_case_results": detailed_results,
    }
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    print(f"\nFull evaluation report saved to {report_path.resolve()}")


if __name__ == "__main__":
    main()