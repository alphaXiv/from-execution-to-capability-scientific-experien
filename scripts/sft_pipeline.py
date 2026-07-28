#!/usr/bin/env python3
"""Procedure-guided/no-procedure teacher generation, LoRA SFT, held-out eval."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path

import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from scicode_repro import extract_code, prompt_for, score_step, select_stratified


SKIPS = {("13", 5), ("62", 0), ("76", 2)}


def load_file(name: str) -> list[dict]:
    ds = load_dataset("SciCode1/SciCode", data_files=name, split="train")
    return [dict(x) for x in ds]


def teacher_examples(config: dict, test_h5: str) -> list[dict]:
    records = sorted(load_file("problems_dev.jsonl"), key=lambda x: int(x["problem_id"]))
    llm = LLM(
        model=config["teacher_model"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        seed=config["seed"],
    )
    tokenizer = llm.get_tokenizer()
    prompts, meta = [], []
    for record in records:
        prior_gold = []
        for step_idx, step in enumerate(record["sub_steps"]):
            if (record["problem_id"], step_idx) in SKIPS:
                prior_gold.append(step["ground_truth_code"])
                continue
            raw = prompt_for(record, step_idx, prior_gold, config["teacher_procedure"])
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw}], tokenize=False, add_generation_prompt=True
                )
            )
            meta.append((record, step_idx, list(prior_gold), raw))
            prior_gold.append(step["ground_truth_code"])
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=config["max_new_tokens"], seed=config["seed"]),
        use_tqdm=True,
    )
    retained = []
    total_pass = 0
    for (record, step_idx, prior_gold, raw), output in zip(meta, outputs):
        code = extract_code(output.outputs[0].text)
        ok, detail = score_step(record, step_idx, prior_gold + [code], test_h5)
        total_pass += int(ok)
        print(
            "TEACHER_STEP_JSON="
            + json.dumps(
                {
                    "step": record["sub_steps"][step_idx]["step_number"],
                    "passed": ok,
                    "error_tail": detail,
                    "teacher_procedure": config["teacher_procedure"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if ok:
            student_prompt = prompt_for(record, step_idx, prior_gold, False)
            retained.append(
                {
                    "step": record["sub_steps"][step_idx]["step_number"],
                    "prompt": student_prompt,
                    "response": code,
                }
            )
    retained.sort(key=lambda x: tuple(map(int, x["step"].split("."))))
    retained = retained[: config["max_train_examples"]]
    print(
        "TEACHER_SUMMARY_JSON="
        + json.dumps(
            {
                "candidate_steps": len(meta),
                "passing_steps": total_pass,
                "retained_steps": len(retained),
                "teacher_procedure": config["teacher_procedure"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return retained


def evaluate_adapter(config: dict, adapter_path: str, test_h5: str) -> dict:
    records = select_stratified(load_file("problems_test.jsonl"), config["n_problems"])
    llm = LLM(
        model=config["model"],
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,
        gpu_memory_utilization=0.90,
        seed=config["seed"],
        enable_lora=True,
        max_lora_rank=config["lora_rank"],
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=config["max_new_tokens"], seed=config["seed"])
    request = LoRARequest("consolidated", 1, adapter_path)
    codes = {r["problem_id"]: [] for r in records}
    results = {r["problem_id"]: [] for r in records}
    for step_idx in range(max(len(r["sub_steps"]) for r in records)):
        active = [
            r for r in records
            if step_idx < len(r["sub_steps"]) and (r["problem_id"], step_idx) not in SKIPS
        ]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_for(r, step_idx, codes[r["problem_id"]], False)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in active
        ]
        outputs = llm.generate(prompts, sampling, lora_request=request, use_tqdm=True)
        for record, output in zip(active, outputs):
            code = extract_code(output.outputs[0].text)
            codes[record["problem_id"]].append(code)
            ok, detail = score_step(record, step_idx, codes[record["problem_id"]], test_h5)
            results[record["problem_id"]].append(ok)
            print(
                "ADAPTER_STEP_JSON="
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
    total = sum(len(x) for x in results.values())
    passed = sum(sum(x) for x in results.values())
    main_passed = sum(bool(x) and all(x) for x in results.values())
    return {
        "condition": "procedure_guided_sft" if config["teacher_procedure"] else "no_procedure_sft",
        "teacher_procedure": config["teacher_procedure"],
        "substep_passed": passed,
        "substep_total": total,
        "substep_accuracy": passed / total,
        "main_passed": main_passed,
        "main_total": len(records),
        "main_accuracy": main_passed / len(records),
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpu_count": 4,
        "active_eval_gpu_count": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-h5", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    started = time.time()
    print("SFT_CONFIG_JSON=" + json.dumps(config, sort_keys=True), flush=True)
    examples = teacher_examples(config, args.test_h5)
    if not examples:
        raise RuntimeError("Teacher produced no verifier-passing examples")
    data_path = "/tmp/concretized.json"
    adapter_path = "/tmp/student-adapter"
    Path(data_path).write_text(json.dumps(examples))
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            "--nproc_per_node=1",
            "scripts/train_lora.py",
            "--data",
            data_path,
            "--output",
            adapter_path,
            "--config",
            args.config,
        ],
        check=True,
    )
    metrics = evaluate_adapter(config, adapter_path, args.test_h5)
    metrics["elapsed_seconds"] = time.time() - started
    metrics["train_examples"] = len(examples)
    print("FINAL_METRICS_JSON=" + json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
