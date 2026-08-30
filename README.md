# privacy_perserving_pllm

Code for the MSc thesis ***Context-Grounded Personalised Question-Answering under
Centralised and Federated Training***.

## What this project does

When someone asks a language model an ordinary question, the information needed
to answer it *well for that person* is usually not in the question. It is buried
in what they said in earlier conversations a habit, a constraint, a
preference they mentioned once and never repeated. A model that is handed the
relevant snippet of that history still has to notice which part of it matters,
work out what it implies, and use it without inventing personal details the
evidence does not support.

That history is also the most sensitive data a user holds. Pooling it on a
server to train on it is exactly what privacy argues against.

This project asks two questions:

1. **Which training objective** best teaches a model to use retrieved
   conversational evidence when answering?
2. **Does that answer survive federation**, when each user's history never
   leaves their device and only gradients are shared?

Five objectives are compared, holding the base model (Qwen3-0.6B + LoRA), the
data (PersonaMem-v2), the retrieval interface and the training budget fixed, so
that any difference is attributable to the learning signal alone. Every
objective is trained twice — once centralised, once under federated gradient
averaging — and scored with the same two metrics.

Each training example is a tuple:

| Symbol | Meaning |
|---|---|
| `q` | the user's current query |
| `c+` | the relevant snippet from that user's history |
| `c-` | an off-topic snippet from the *same* user, mined by lowest TF-IDF similarity |
| `y+` | the correctly personalised answer |
| `y-` | a general answer — fluent and correct, but ignores `c+` |

The five objectives differ in which of these fields the loss consumes:

| Objective | Loss | Notebook stem |
|---|---|---|
| Supervised fine-tuning | cross-entropy on `y+` given `(q, c+)` | `*_sft_*` |
| Answer-contrastive | SFT + ORPO odds ratio, `y+` vs `y-` under `c+` | `*_orpo_*` |
| Context-contrastive | SFT + ORPO odds ratio, `c+` vs `c-` under `y+` | `*_ctx_contrast_*` |
| Dual-contrastive | SFT + both odds-ratio terms | `*_combined_orpo_*` |
| Draft-revision | generation + weighted revision of `y-` into `y+` | `*_gen_edit_*` |

Every notebook opens with a header cell stating the exact loss or metric it
implements, together with the constants it uses.

## Folder structure

| Directory | Contents |
|---|---|
| `zero_shot/` | Persona subset construction, subset merging, and the untrained zero-shot baseline |
| `centralised/` | The five objectives trained with all data pooled on one server |
| `fed_grad_avg/` | The same five objectives under federated gradient averaging, one client per persona |
| `evaluation/` | Inference from a trained adapter, then the four scoring notebooks |
| `annotation/` | Interactive tool for manually marking relevant spans in a snippet |

## Order to run

### 1. Build the data

```
zero_shot/create_persona_subsets.ipynb     # filter PersonaMem-v2 into per-persona subsets
zero_shot/combine_all_subsets.ipynb        # merge them into pooled train/test splits
```

`create_persona_subsets` keeps personas with enough training and held-out
examples. Each surviving persona becomes one federated client later, so this
step defines the client population as well as the data.

### 2. Train

Run whichever objectives you need — they are independent of each other, and both
directories contain the same five strategies:

```
centralised/centralized_sft_snippet.ipynb
centralised/centralized_orpo_snippet.ipynb
centralised/centralized_ctx_contrast_orpo_snippet.ipynb
centralised/centralized_combined_orpo_snippet.ipynb
centralised/centralized_gen_edit_snippet.ipynb

fed_grad_avg/federated_sft_gradavg_snippet.ipynb
fed_grad_avg/federated_orpo_gradavg_snippet.ipynb
fed_grad_avg/federated_ctx_contrast_orpo_gradavg_snippet.ipynb
fed_grad_avg/federated_combined_orpo_gradavg_snippet.ipynb
fed_grad_avg/federated_gen_edit_gradavg_snippet.ipynb
```

Each writes a LoRA adapter. For the zero-shot reference point, no training is
needed — run `zero_shot/zero_shot_50_snippet.ipynb` instead.

### 3. Generate answers

```
evaluation/inference_global_adapter.ipynb
```

**Run this before any scoring notebook.** It decodes the free-form answers that
three of the four metrics consume; without its output they have nothing to score.

### 4. Score

```
evaluation/mc_choice_answer_logprob.ipynb            # Acc_MC — multiple-choice answer log-likelihood
evaluation/llm_judge_pref_metric_deepseek.ipynb      # PSR    — LLM-as-a-judge success rate
evaluation/length_retention_judge_deepseek.ipynb     # PSR(f) vs snippet length
evaluation/persona_sat_rank_correlation_deepseek.ipynb  # per-persona rank correlation
```

`length_retention` and `persona_sat_rank_correlation` both build on the judge
labels, so run `llm_judge_pref_metric_deepseek` before them.
`mc_choice_answer_logprob` is independent of the judge and can run at any point
after training.

> **Two multiple-choice notebooks.** `mc_choice_answer_logprob.ipynb` is the one
> reported in the thesis: it scores each candidate by the length-normalised
> log-likelihood of the **answer text**.
> `mc_choice_success_rate.ipynb` is an earlier variant that scores the **option
> letter** (A/B/C/D) instead. Both are kept for reference; use the former.

## Requirements

Training was run on a single Colab GPU, one run at a time.

The judge notebooks call the DeepSeek API and read the key from the environment:

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

No key is stored in this repository.

## Data

[PersonaMem-v2](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2)
(`bowen-upenn/PersonaMem-v2`), text split.
# privacy_perserving_pllm
