# SciConsolidate claim-by-claim reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/from-execution-to-capability-scientific-experien/blob/main/notebooks/sciconsolidate_reproduction.py)

This repository reproduces two claims from [*From Execution to Capability: Scientific Experience Consolidation via Procedural Knowledge Synthesis* (arXiv:2607.24459)](https://arxiv.org/abs/2607.24459): that runtime cross-task procedures benefit a stronger scientific-coding model more than a weaker model, and that procedure-guided concretized supervision improves a weaker model after procedures are removed.

**Assessment: not reproduced under this bounded public reconstruction.** On the primary public SciCode slice, targeted procedures changed executable substep pass rate by −1.89 points for both the 7B and 32B substitutes (paper: weaker −0.30, stronger +3.85). Across three exact eight-example LoRA pairs, guided supervision averaged 7.55% versus 8.81% for no-procedure controls, a −1.26-point difference (paper: +3.89).

We used official executable SciCode tests, Qwen2.5-Coder-Instruct 7B/14B/32B substitutes, reconstructed answer-free procedures, verifier-gated teacher examples, and procedure-free adapter deployment. The original repository, prompts, trajectories, split metadata, checkpoints, and recipe were unavailable; tuning used rank-16 LoRA rather than full-model SFT. All runs used Kubernetes on NVIDIA RTX PRO 6000 Blackwell GPUs, with 16 GPUs allocated concurrently at peak and 3.01 hours elapsed from first submit to final result.

- [Detailed illustrated report](reports/sciconsolidate/report.md)
- [Self-contained marimo tutorial](notebooks/sciconsolidate_reproduction.py)
- [Machine-readable measurements](results/metrics.json)
- Local notebook: `marimo edit notebooks/sciconsolidate_reproduction.py`

## Experiment log

Every experiment inherited the exact run command shown by `orx exp status`: `bash scripts/run.sh`. Each Kubernetes job allocated four NVIDIA RTX PRO 6000 Blackwell GPUs; the stabilized implementation actively used one GPU per job after tensor-parallel/distributed shared-memory failures.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| `main` | Polished publication surface | Not run as an experiment (publication surface) | README, report, figures, notebook | Presentation only |
| [7B baseline](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/k8s-shell-fixed-weaker-baseline) | No runtime procedure | `bash scripts/run.sh` | 5/53 substeps; 0/12 main | K8s, 4× GPU allocated |
| [7B targeted](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/runtime-round-weaker-plus-procedures) | Reconstructed procedures | `bash scripts/run.sh` | 4/53; 0/12 | K8s, 4× GPU allocated |
| [7B generic](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/runtime-weak-7b-generic-control) | Length-matched reminder | `bash scripts/run.sh` | 5/53; 0/12 | K8s, 4× GPU allocated |
| [14B baseline](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/runtime-intermediate-14b-baseline) | Intermediate-capacity control | `bash scripts/run.sh` | 15/53; 3/12 | K8s, 4× GPU allocated |
| [14B targeted](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/runtime-intermediate-14b-plus-procedures) | Reconstructed procedures | `bash scripts/run.sh` | 10/53; 2/12 | K8s, 4× GPU allocated |
| [14B generic](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/runtime-intermediate-14b-generic-control) | Length-matched reminder | `bash scripts/run.sh` | 16/53; 3/12 | K8s, 4× GPU allocated |
| [32B baseline](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/runtime-round-stronger-baseline) | No runtime procedure | `bash scripts/run.sh` | 11/53; 1/12 | K8s, 4× GPU allocated |
| [32B targeted](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/runtime-round-stronger-plus-procedures) | Reconstructed procedures | `bash scripts/run.sh` | 10/53; 1/12 | K8s, 4× GPU allocated |
| [32B generic](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/generic-instruction-length-control) | Length-matched reminder | `bash scripts/run.sh` | 15/53; 2/12 | K8s, 4× GPU allocated |
| [Guided seed 42](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/procedure-guided-lora-consolidation) / [control](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/eight-example-no-procedure-lora-control) | Exact eight-example LoRA pair | `bash scripts/run.sh` | Guided 3/53 vs control 4/53 | K8s, 4× GPU each |
| [Guided seed 7](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/procedure-guided-lora-seed-7) / [control](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/eight-example-no-procedure-lora-seed-7) | Independent exact pair | `bash scripts/run.sh` | Guided 5/53 vs control 3/53 | K8s, 4× GPU each |
| [Guided seed 123](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/procedure-guided-lora-seed-123) / [control](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/eight-example-no-procedure-lora-seed-123) | Independent exact pair | `bash scripts/run.sh` | Guided 4/53 vs control 7/53 | K8s, 4× GPU each |
| [Low-rate guided](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/procedure-guided-lora-low-learning-rate) / [control](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/no-procedure-lora-low-learning-rate) | Learning-rate robustness | `bash scripts/run.sh` | Tie: 4/53 and 1/12 | K8s, 4× GPU each |
| [Gold-code diagnostics](https://github.com/alphaXiv/from-execution-to-capability-scientific-experien/tree/orx/gold-supervision-lora-upper-bound) | Verified-target control at two rates | `bash scripts/run.sh` | 3/53 at both rates | K8s, 4× GPU each |

## Reproduce the protocol

The committed Kubernetes manifest and runner implement the experiment contract:

```bash
bash scripts/run.sh
```

Change experimental conditions in committed `experiment.json`, not in the command. The runner downloads the public SciCode JSON and official HDF5 tests, performs verifier-backed evaluation, and prints structured terminal metrics. Model inference and tuning require a CUDA environment with substantial memory; the notebook is the lightweight way to inspect all published evidence.
