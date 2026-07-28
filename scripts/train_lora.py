#!/usr/bin/env python3
"""Small deterministic DDP LoRA trainer used by the consolidation experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    examples = json.loads(Path(args.data).read_text())
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.manual_seed(config["seed"])

    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(local_rank)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["lora_rank"],
            lora_alpha=2 * config["lora_rank"],
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    encoded = []
    for example in examples:
        text = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": f"```python\n{example['response']}\n```"},
            ],
            tokenize=False,
        )
        item = tokenizer(text, truncation=True, max_length=4096, return_tensors="pt")
        encoded.append({k: v[0] for k, v in item.items()})

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        ids, masks = [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            ids.append(torch.cat([item["input_ids"], torch.full((pad,), tokenizer.pad_token_id)]))
            masks.append(torch.cat([item["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        input_ids = torch.stack(ids)
        attention_mask = torch.stack(masks)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return input_ids, attention_mask, labels

    sampler = DistributedSampler(encoded, shuffle=True, seed=config["seed"])
    loader = DataLoader(encoded, batch_size=1, sampler=sampler, collate_fn=collate)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["learning_rate"],
        weight_decay=0.0,
    )
    started = time.time()
    model.train()
    step = 0
    for epoch in range(config["epochs"]):
        sampler.set_epoch(epoch)
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(local_rank)
            attention_mask = attention_mask.to(local_rank)
            labels = labels.to(local_rank)
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                print(f"TRAIN_METRIC step={step} loss={loss.item():.6f}", flush=True)
            step += 1
    dist.barrier()
    if rank == 0:
        model.module.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print(
            "TRAIN_SUMMARY_JSON="
            + json.dumps(
                {
                    "examples": len(examples),
                    "epochs": config["epochs"],
                    "steps_per_rank": step,
                    "elapsed_seconds": time.time() - started,
                    "lora_rank": config["lora_rank"],
                    "world_size": dist.get_world_size(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
