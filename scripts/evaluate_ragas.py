"""CI/CD entrypoint to run the 20-question RAGAS evaluation & Langfuse FinOps report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from orderpulse.evaluation.ragas_evaluator import RagasEvaluator  # noqa: E402


def main() -> int:
    evaluator = RagasEvaluator()
    report = evaluator.run_benchmark()

    ragas_path = evaluator.save_report(report, ROOT / "reports" / "ragas_benchmark_results.json")
    traces_path = evaluator.engine.tracer.export_summary(
        ROOT / "reports" / "langfuse_traces_summary.json"
    )

    metrics = report["ragas_metrics"]
    finops = report["finops_metrics"]

    print("=== OrderPulse AI — RAGAS Benchmark (20 Synthetic Cases) ===")
    print(json.dumps({"ragas_metrics": metrics, "finops_metrics": finops}, indent=2))
    print(f"Saved RAGAS report  -> {ragas_path}")
    print(f"Saved Langfuse summary -> {traces_path}")

    # CI Quality Gates
    failed_gates: list[str] = []
    if metrics["faithfulness"] < 0.90:
        failed_gates.append(f"faithfulness={metrics['faithfulness']} < 0.90")
    if metrics["context_precision"] < 0.85:
        failed_gates.append(f"context_precision={metrics['context_precision']} < 0.85")
    if metrics["answer_relevancy"] < 0.90:
        failed_gates.append(f"answer_relevancy={metrics['answer_relevancy']} < 0.90")
    if metrics["decision_accuracy"] < 0.95:
        failed_gates.append(f"decision_accuracy={metrics['decision_accuracy']} < 0.95")
    if metrics["hitl_compliance_rate"] < 1.0:
        failed_gates.append(f"hitl_compliance_rate={metrics['hitl_compliance_rate']} < 1.0")
    if metrics["guardrail_block_rate"] < 1.0:
        failed_gates.append(f"guardrail_block_rate={metrics['guardrail_block_rate']} < 1.0")

    if failed_gates:
        print("FAILED CI QUALITY GATES:", ", ".join(failed_gates), file=sys.stderr)
        return 1

    print("ALL RAGAS & LLMOps QUALITY GATES PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
