#!/usr/bin/env python3
"""Bounded, executable SciCode runtime-procedure reproduction."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from vllm import LLM, SamplingParams


PROCEDURE_BUNDLE = r"""
Reusable scientific-computing procedure

1. Interface audit. Before calling a helper, verify that it is defined, imported, or
passed in; check argument count/order and avoid shadowing a function with a variable.
Return exactly the requested type and shape.

2. Array and numerical audit. Before combining arrays, state the intended axes and
make ambiguous broadcasting explicit with reshape/newaxis. Use indexing='ij' when
meshgrid inputs map in order to output axes. Check reduction axes, dtype, units, and
boundary cases; use stable algebra and tolerances rather than fragile exact equality.

3. Dependency-flow audit. Treat each earlier function as a tested contract. Before
writing the current function, identify which earlier outputs feed which current
inputs. Preserve state updates in their specified order; do not silently recompute,
drop, or overwrite an intermediate. Add compact assertions for critical shapes and
finite values, then remove any debugging output from the final answer.

Apply only relevant clauses. These are decision rules, not substitutes for deriving
the task-specific formula. Output only executable Python in one code block.
""".strip()

SKIPS = {("13", 5), ("62", 0), ("76", 2)}


def load_records() -> list[dict]:
    ds = load_dataset("SciCode1/SciCode", data_files="problems_test.jsonl", split="train")
    return [dict(x) for x in ds]


def select_stratified(records: list[dict], n: int) -> list[dict]:
    # The public JSON release omits the paper's private domain/task-type labels.
    # Use two public attributes instead: benchmark-position quartile and chain length.
    ordered = sorted(records, key=lambda x: int(x["problem_id"]))
    rank = {r["problem_id"]: i for i, r in enumerate(ordered)}
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in ordered:
        quartile = min(3, 4 * rank[record["problem_id"]] // len(ordered))
        steps = len(record["sub_steps"])
        complexity = "single" if steps == 1 else ("short" if steps <= 4 else "long")
        groups[(f"id_q{quartile + 1}", complexity)].append(record)
    for values in groups.values():
        values.sort(key=lambda x: int(x["problem_id"]))
    keys = sorted(groups)
    chosen: list[dict] = []
    depth = 0
    while len(chosen) < n:
        progressed = False
        for key in keys:
            if depth < len(groups[key]) and len(chosen) < n:
                chosen.append(groups[key][depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return sorted(chosen, key=lambda x: int(x["problem_id"]))


def prompt_for(record: dict, step_idx: int, previous_code: list[str], with_procedure: bool) -> str:
    step = record["sub_steps"][step_idx]
    prior = []
    for j in range(step_idx):
        prior.append(record["sub_steps"][j]["step_description_prompt"])
        prior.append(previous_code[j])
    body = f"""You are solving one sub-step of a scientist-curated coding benchmark.
Use the scientific background and satisfy the exact function interface.

Required dependencies:
{record["required_dependencies"]}

Prior sub-steps and generated implementations:
{chr(10).join(prior) if prior else "(none)"}

Current sub-step:
{step["step_description_prompt"]}

Scientific background:
{step.get("step_background", "")}

Required function:
{step["function_header"]}

Return only the complete implementation for the current function in one Python code block.
"""
    if with_procedure:
        body = PROCEDURE_BUNDLE + "\n\n" + body
    return body


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.S | re.I)
    return (blocks[-1] if blocks else text).strip()


def score_step(record: dict, step_idx: int, all_code: list[str], test_h5: str) -> tuple[bool, str]:
    step = record["sub_steps"][step_idx]
    step_id = step["step_number"]
    script = [
        record["required_dependencies"],
        *all_code[: step_idx + 1],
        "from scicode.parse.parse import process_hdf5_to_tuple",
        f"targets = process_hdf5_to_tuple({step_id!r}, {len(step['test_cases'])}, {test_h5!r})",
    ]
    for i, test in enumerate(step["test_cases"]):
        script.append(f"target = targets[{i}]")
        script.append(test)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.py"
        path.write_text("\n\n".join(script))
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        try:
            result = subprocess.run(
                ["python3", str(path)],
                text=True,
                capture_output=True,
                timeout=180,
                env=env,
            )
            detail = (result.stderr or result.stdout)[-500:].replace("\n", " ")
            return result.returncode == 0, detail
        except subprocess.TimeoutExpired:
            return False, "timeout"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-h5", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    random.seed(config["seed"])
    started = time.time()

    print("CONFIG_JSON=" + json.dumps(config, sort_keys=True), flush=True)
    records = select_stratified(load_records(), config["n_problems"])
    subset = [
        {
            "problem_id": r["problem_id"],
            "id_quartile": f"q{min(4, 1 + 4 * sorted(int(x['problem_id']) for x in records).index(int(r['problem_id'])) // len(records))}",
            "complexity": "single" if len(r["sub_steps"]) == 1 else ("short" if len(r["sub_steps"]) <= 4 else "long"),
            "steps": len(r["sub_steps"]),
        }
        for r in records
    ]
    print("SCICODE_SUBSET_JSON=" + json.dumps(subset, sort_keys=True), flush=True)

    llm = LLM(
        model=config["model"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        seed=config["seed"],
        enforce_eager=False,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=config["temperature"],
        max_tokens=config["max_new_tokens"],
        seed=config["seed"],
        stop=["</s>", "<|im_end|>"],
    )
    codes: dict[str, list[str]] = {r["problem_id"]: [] for r in records}
    results: dict[str, list[bool]] = {r["problem_id"]: [] for r in records}
    max_steps = max(len(r["sub_steps"]) for r in records)
    for step_idx in range(max_steps):
        active = [
            r for r in records
            if step_idx < len(r["sub_steps"]) and (r["problem_id"], step_idx) not in SKIPS
        ]
        prompts = []
        for record in active:
            raw = prompt_for(
                record,
                step_idx,
                codes[record["problem_id"]],
                config["runtime_procedure"],
            )
            messages = [{"role": "user", "content": raw}]
            prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        for record, output in zip(active, outputs):
            code = extract_code(output.outputs[0].text)
            codes[record["problem_id"]].append(code)
            ok, detail = score_step(record, step_idx, codes[record["problem_id"]], args.test_h5)
            results[record["problem_id"]].append(ok)
            print(
                "STEP_RESULT_JSON="
                + json.dumps(
                    {
                        "problem_id": record["problem_id"],
                        "step": record["sub_steps"][step_idx]["step_number"],
                        "passed": ok,
                        "error_tail": detail,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    total_steps = sum(len(v) for v in results.values())
    passed_steps = sum(sum(v) for v in results.values())
    passed_main = sum(bool(v) and all(v) for v in results.values())
    summary = {
        "condition": config["condition"],
        "model": config["model"],
        "model_role": config["model_role"],
        "runtime_procedure": config["runtime_procedure"],
        "seed": config["seed"],
        "main_passed": passed_main,
        "main_total": len(records),
        "main_accuracy": passed_main / len(records),
        "substep_passed": passed_steps,
        "substep_total": total_steps,
        "substep_accuracy": passed_steps / total_steps,
        "elapsed_seconds": time.time() - started,
        "allocated_gpu_count": 4,
        "active_inference_gpu_count": 1,
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
    }
    print("FINAL_METRICS_JSON=" + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
