import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[idx:])
            trailing = text[idx + end :].strip()
            if isinstance(obj, dict) and not trailing:
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("Model output was not a single valid JSON object")


def run_extraction(
    model: GenerativeModel,
    prompt_text: str,
    required_features: List[str],
) -> Dict[str, Any]:
    prompt = f"""
You are a strict JSON extractor for macroeconomic model inputs.
Required features: {required_features}

User message:
{prompt_text}

Return ONLY valid JSON:
{{
  "features": {{"feature_name": number_or_null}},
  "missing_features": ["feature_name"],
  "assumptions": ["short string"],
  "clarifying_question": "single short question"
}}
"""
    response = model.generate_content(
        prompt,
        generation_config=GenerationConfig(temperature=0.0, top_p=0.1, max_output_tokens=512),
    )
    return extract_json_object(response.text)


def run_judge(
    judge_model: GenerativeModel,
    user_prompt: str,
    extracted: Dict[str, Any],
    expected_hints: Dict[str, Any],
) -> Dict[str, Any]:
    judge_prompt = f"""
You are an evaluator for feature extraction quality in a Fed funds rate pipeline.

User prompt:
{user_prompt}

Extracted payload:
{json.dumps(extracted, ensure_ascii=True)}

Expected hints (ground truth constraints):
{json.dumps(expected_hints, ensure_ascii=True)}

Score extraction quality and return ONLY JSON:
{{
  "pass": true_or_false,
  "score": 0_to_1,
  "issues": ["short issue string"],
  "rationale": "one concise sentence"
}}

Rules:
- pass=true only if extracted values are consistent with prompt and expected_hints.
- Fail for missing required facts explicitly present in prompt.
- Fail for fabricated numeric values not supported by prompt.
"""
    response = judge_model.generate_content(
        judge_prompt,
        generation_config=GenerationConfig(temperature=0.0, top_p=0.1, max_output_tokens=400),
    )
    return extract_json_object(response.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge eval for Gemini feature extraction")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--extractor-model", default="gemini-2.0-flash")
    parser.add_argument("--judge-model", default="gemini-2.0-flash")
    parser.add_argument("--schema-path", default="model/feature_columns.json")
    parser.add_argument("--cases", default="evals/extraction_cases.jsonl")
    parser.add_argument("--out", default="eval_logs/llm_judge_eval.csv")
    args = parser.parse_args()

    schema = json.loads(Path(args.schema_path).read_text(encoding="utf-8"))
    required_features = schema["feature_columns"]
    cases = load_jsonl(Path(args.cases))

    vertexai.init(project=args.project_id, location=args.region)
    extractor = GenerativeModel(args.extractor_model)
    judge = GenerativeModel(args.judge_model)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "case_id",
        "pass",
        "score",
        "extract_ms",
        "judge_ms",
        "issues",
        "rationale",
        "extracted_json",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()

        passes = 0
        for case in cases:
            case_id = case["case_id"]
            prompt_text = case["prompt"]
            expected_hints = case.get("expected_hints", {})

            t0 = time.perf_counter()
            extracted = run_extraction(extractor, prompt_text, required_features)
            extract_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            verdict = run_judge(judge, prompt_text, extracted, expected_hints)
            judge_ms = (time.perf_counter() - t1) * 1000.0

            passed = bool(verdict.get("pass", False))
            passes += int(passed)

            writer.writerow(
                {
                    "case_id": case_id,
                    "pass": int(passed),
                    "score": verdict.get("score", 0.0),
                    "extract_ms": round(extract_ms, 2),
                    "judge_ms": round(judge_ms, 2),
                    "issues": json.dumps(verdict.get("issues", [])),
                    "rationale": verdict.get("rationale", ""),
                    "extracted_json": json.dumps(extracted),
                }
            )

    total = max(len(cases), 1)
    print(f"Wrote {len(cases)} rows to {out_path}")
    print(f"Pass rate: {passes / total:.2%}")


if __name__ == "__main__":
    main()
