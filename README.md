# LGCA Release Code

Code supporting *"LGCA: A Computationally Efficient Frozen-Backbone
Cross-Attention Adaptation for Vietnamese Bone-Radiograph Question
Answering"* (IEEE Access). This folder contains the model, training,
inference, and evaluation code used to produce every number reported in
the paper. **It does not contain the clinical dataset itself** — see
[Data (not included)](#data-not-included) for why, and what you need to
supply to run any of this.

If you are the thesis author preparing this for public release: replace
this placeholder repository link once published: `<REPO_URL>`.

## Folder structure

```
release_code/
├── model_llava/llava/   LGCA architecture (the actual model code)
├── training/            Training scripts (LGCA + LoRA baselines)
├── inference/            Inference / generation scripts
├── metrics/              All metric computation (BLEU/ROUGE/BERTScore,
│                         proxy diagnostic accuracy, leakage audit,
│                         bootstrap significance test)
├── data_prep/            Dataset construction from raw hospital records
├── rag/                  Retrieval-augmented generation (BM25/CLIP)
├── external_qwenvl/      Qwen2-VL + LoRA external-competitor baseline,
│                         and the Qwen2-VL + cross-attention variant used
│                         for the English replication
└── docs/                 GENERATE_BUG_INVESTIGATION.md — see "Known
                           inference bug" below before using
                           infer_baseline/infer_lora_ft
```

## Data (not included)

The bone-radiograph images and clinical text were copied verbatim from
a single Vietnamese hospital's own records, generated during real,
routine patient care — not collected for this study — with personal
identifying information removed before the data were provided to the
authors. Use of the dataset for this research is documented in a
written authorization signed by the hospital director and a
representative of the authors' university (University of Science,
VNU-HCM) (see "Ethics and Data Statement" in the paper for the full
account, including the scope of that authorization). This repository
therefore ships no patient data, images, or de-identified clinical text
of any kind.

To run anything here, you need your own data in this shape:

- `bone_data/` — image root; `image` fields in the JSONL files below are
  paths relative to this root (e.g.
  `CH151209737/30.03.2023/img-02772-00001.jpg`).
- `SPLIT2_OFFICIAL_BACKUP/dataset_{train,val,test}.jsonl` — one JSON
  object per line: `{"image": "...", "question": "...", "answer": "..."}`.
  This is the **reconstructed split** (6,807 train / 704 val / 1,287
  test) used for every parameter-matched, ablation, and modality-ablation
  table in the paper (`sec:param-matched` onward). `data_prep/preprocess_data.py`
  shows how an equivalent split was built from raw hospital CSV export +
  image directory (`--train_ratio 0.75 --val_ratio 0.10 --test_ratio
  0.15 --seed 42`); `data_prep/preprocess_merged.py` is the more complete
  reconstruction of the official 8,798-sample dataset (merges CSV-parsing
  logic with multi-visit/multi-image detection), which is what you should
  start from if rebuilding from scratch.
- The **original split** used for Table I/II (`tab:main`) and the
  external-competitor comparison is a separate, now-unavailable
  infrastructure snapshot (see the paper's `sec:param-matched` for why
  the two splits exist and are not numerically interchangeable); this
  repo only lets you reproduce results on the reconstructed split.

## Setup

Tested with Python 3.11 on a single CUDA GPU (24–50 GB VRAM; A40/A5000
used throughout). Pinned dependency versions (newer `bitsandbytes`/
`transformers` combinations hit real bugs during this project — see
below):

```bash
pip install bitsandbytes==0.41.0 peft==0.4.0 accelerate==0.21.0 \
    transformers==4.36.2 sentencepiece==0.1.99 tokenizers \
    einops==0.6.1 einops-exts==0.0.4 timm==0.9.12 \
    huggingface_hub pillow protobuf scipy torch
```

**bitsandbytes CUDA library patch** (needed on newer CUDA images where
`bitsandbytes==0.41.0` doesn't ship a matching prebuilt `.so`):

```bash
cd "$(python -c 'import bitsandbytes, os; print(os.path.dirname(bitsandbytes.__file__))')"
cp libbitsandbytes_cuda122.so libbitsandbytes_cuda124.so
cp libbitsandbytes_cuda122.so libbitsandbytes_cuda130.so
cp libbitsandbytes_cuda122.so libbitsandbytes_cuda132.so
```

**Known transient import error**: the very first `import bitsandbytes`
in a fresh process can raise `ModuleNotFoundError: No module named
'triton.language'`. This is a harmless import-ordering race, not a
missing package — simply retry the same import once (or run
`python -c "import triton.language; import bitsandbytes"` first).

**`PYTHONPATH`**: run everything from the directory containing
`model_llava/llava/` with `export PYTHONPATH=<path-to-model_llava>`, or
copy/symlink `model_llava/llava/` next to the script you're running —
every script does `from llava...` imports assuming `llava/` is importable
from the current directory.

### BERTScore / BioBERT truncation bug

`compute_metrics.py`'s default `--bert_model dmis-lab/biobert-base-cased-v1.2`
hits `OverflowError: int too big to convert` under recent `transformers`
releases (BioBERT's tokenizer config leaves `model_max_length` unset,
and the new Rust tokenizer backend can't handle the resulting sentinel
value). Fix by monkey-patching `bert_score`'s sentence-encode function
to pass an explicit `max_length=512` before calling `bert_score.score()`:

```python
import bert_score.utils as bu, bert_score.scorer as bs

def _patched_sent_encode(tokenizer, sent):
    sent = sent.strip()
    if sent == "":
        return tokenizer.build_inputs_with_special_tokens([])
    return tokenizer.encode(sent, add_special_tokens=True, max_length=512, truncation=True)

bu.sent_encode = _patched_sent_encode
bs.sent_encode = _patched_sent_encode
# ... then import bert_score / call compute_metrics.py's functions as normal
```

Also note: `compute_metrics_clean_subset.py` internally splits its input
into leak/clean/overall subsets and will crash with
`ZeroDivisionError: Fraction(0, 0)` if you pass it a file that is
*already* fully clean (0 leaked rows) — use `compute_metrics.py` directly
on pre-filtered files instead of `compute_metrics_clean_subset.py` in
that case.

## The `LGCA_LAYOUT` environment variable — read this before training or
running inference

`model_llava/llava/model/language_model/llava_mistral_xattn.py` reads
`LGCA_LAYOUT` to pick the depth-placement schedule:

| Value (or unset) | Schedule | Modules |
|---|---|---|
| unset / not set | three-zone default (paper's main configuration) | 17 |
| `uniform` | evenly spaced across all 32 layers | 17 |
| `top_only` | concentrated in the top 17 layers | 17 |
| `fewer` | half-density subsample of the three-zone schedule | 9 |

**This variable must be set identically for training and for inference
on the resulting checkpoint.** A checkpoint trained with one layout and
evaluated with `LGCA_LAYOUT` set to a different value will load with
mismatched module shapes / silently wrong results — this exact failure
mode happened once during this project's own experimentation (an
abandoned pod named `lgca-e5-mismatched-infer`). Always export
`LGCA_LAYOUT` once per shell session (or `unset LGCA_LAYOUT` for the
default) and keep the train and inference commands in the same script so
they can't drift apart.

## Reproducing each table

All of the following assume the reconstructed split described above and
`--base_model microsoft/llava-med-v1.5-mistral-7b`.

### LGCA training (any layout)

```bash
export PYTHONPATH=/path/to/model_llava
unset LGCA_LAYOUT   # or: export LGCA_LAYOUT=uniform|top_only|fewer

python training/train_xattn_valsel.py \
  --stage 1 \
  --base_model microsoft/llava-med-v1.5-mistral-7b \
  --data_path SPLIT2_OFFICIAL_BACKUP/dataset_train.jsonl \
  --val_data_path SPLIT2_OFFICIAL_BACKUP/dataset_val.jsonl \
  --image_root bone_data \
  --output_dir ./output_<tag> \
  --seed 42 \
  --save_steps 250 --early_stopping_patience 20
```

This is the **validation-selected checkpoint protocol** used for every
LGCA number in the paper from `sec:param-matched` onward: evaluate every
250 steps (5 evaluations over the fixed 3-epoch/1,275-step schedule),
keep the lowest-eval_loss checkpoint (`load_best_model_at_end=True`,
`metric_for_best_model="eval_loss"`). Early stopping (`patience=20`
evaluations) is configured but, with only 5 evaluations in the standard
schedule, never actually triggers — it only becomes relevant if you
raise `--num_epochs` well past 3 (see "equal-time-budget" note below).
`training/train_xattn.py` is the earlier, non-validation-selected
version (`save_strategy="epoch"`, no val-based selection) that
`train_xattn_valsel.py` extends; use it only if you specifically want
the final-epoch checkpoint instead.

An optional `--max_train_seconds <N>` flag (added for the paper's
equal-training-time-budget pilot, Section "Does extended training change
LGCA's quality ceiling?") stops training early once wall-clock time
exceeds `N` seconds, regardless of `--num_epochs`. Omitted by default —
inert unless you pass it.

The **learned-gate control** (confirms the cold-start argument,
`sec:learned-gate`) is `training/train_xattn_learned_gate.py` — identical
to `train_xattn.py` except for the model class (learned $\tanh$-gate
instead of the fixed $\alpha=0.05$ coefficient).

Seeds used throughout: **42, 43, 44**.

### LoRA $r{=}256$ parameter-matched control

```bash
python training/train_lora_baseline_v3.py \
  --base_model microsoft/llava-med-v1.5-mistral-7b \
  --data_path SPLIT2_OFFICIAL_BACKUP/dataset_train.jsonl \
  --val_data_path SPLIT2_OFFICIAL_BACKUP/dataset_val.jsonl \
  --image_root bone_data \
  --output_dir ./output_lora256_<seed> \
  --num_epochs 3 --batch_size 1 --grad_accum 16 --max_length 512 \
  --lora_r 256 --lora_alpha 512 --lora_dropout 0.1 \
  --early_stopping_patience 20 --save_steps 250 \
  --seed 42
```

`--lora_r 256` with all seven target projection modules yields
**708,837,376 trainable parameters** — matches LGCA's ≈723M within the
margin reported in the paper. `--batch_size 1 --grad_accum 16` on 6,807
training rows gives exactly 1,275 steps over 3 epochs, matching LGCA's
schedule for a fair step-for-step comparison. The script includes an
anti-EOS-degeneration penalty (`--first_token_eos_penalty`, default
`1.0`) — **do not set this to 0 or train with fewer steps than this
schedule provides**: an earlier, much shorter run (testing an
equal-*LGCA*-time-budget comparison rather than equal steps) produced a
completely degenerate, repetition-collapsed model, because 708M
trainable parameters need close to the full 1,275-step schedule to
produce coherent output at all. That direction of comparison was
abandoned for this reason; only the reverse direction (give LGCA LoRA's
longer time budget) was pursued, and is the `--max_train_seconds` flag
on `train_xattn_valsel.py` above.

### Inference

```bash
export PYTHONPATH=/path/to/model_llava
unset LGCA_LAYOUT   # match whatever was used for training this checkpoint

python inference/run_inference_ablation.py \
  --model_type xattn --modality full \
  --model_path ./output_<tag>/stage1 \
  --base_model microsoft/llava-med-v1.5-mistral-7b \
  --data_path SPLIT2_OFFICIAL_BACKUP/dataset_test.jsonl \
  --image_root bone_data \
  --output_file results_<tag>_full_outputs.jsonl
```

`--modality` is `full` / `image_only` / `text_only` (used for the
modality-ablation tables); `--model_type` is `xattn` / `lora_ft` /
`baseline`. For `lora_ft`, point `--model_path` at the **merged** output
directory from `train_lora_baseline_v3.py` (it auto-merges LoRA into the
base model unless `--load_in_8bit`/`--load_in_4bit` was used, in which
case it saves an adapter-only directory and inference must load it via
`peft.PeftModel.from_pretrained` instead).

`inference/run_inference.py` is the single-model-type counterpart
(`--model_type baseline|xattn|lora_ft`, no modality ablation) and defines
`load_baseline_model`/`load_lora_ft_model`/`load_xattn_model`, which
`run_inference_ablation.py` imports — keep both files in the same
directory.

#### Known inference bug — read before trusting `baseline`/`lora_ft` numbers

`infer_xattn()` never hits this; it decodes by splitting on the literal
string `"Answer:"` rather than slicing by token position, which happens
to sidestep the whole bug class. `infer_baseline()`/`infer_lora_ft()`
historically did not: `LlavaMistralForCausalLM.generate()`
(`model_llava/llava/model/language_model/llava_mistral.py`) computes
`inputs_embeds` internally for the images-present path and passes it to
`transformers`' `generate()` without an explicit `attention_mask`,
which can yield silently empty (`100%` empty) generations. The slicing
half of this bug is already fixed in `inference/run_inference.py`'s
`infer_lora_ft()` (decodes `output_ids[0]` directly, no
`input_ids.shape[1]:` slice — see the comment there); the
`attention_mask=None` half is **not** fixed in `llava_mistral.py` as
shipped here. If you need baseline/LoRA-FT inference and get
suspiciously empty or garbled output, use
`inference/run_lora_ft_fixed.py` / `inference/run_baseline_fixed_full.py`
/ `inference/run_baseline_textonly_fixed.py` instead — these
monkey-patch the fix at runtime without modifying the shared model code,
and were the scripts actually used to produce this project's final
baseline/LoRA-FT numbers after the bug was found. Full write-up,
including how a *third*, independent bug (string-sort instead of
numeric-sort checkpoint selection) compounded with this one:
`docs/GENERATE_BUG_INVESTIGATION.md`.

### Metrics

```bash
python metrics/proxy_classification_metrics.py \
  --files results_a.jsonl results_b.jsonl \
  --names configA configB \
  --output_json out.json --output_md out.md
```

Keyword-lexicon-based disease-group classification (reused from
`data_prep/classify_patients.py`) + laterality side-matching; no
embedding model, purely rule-based — this is the paper's primary
"diagnostic-group proxy accuracy" metric, explicitly not a clinically
validated correctness measure (see the paper's Discussion).

```bash
python metrics/compute_metrics.py \
  --files results_a.jsonl results_b.jsonl \
  --names configA configB \
  --bert_model roberta-large \
  --output_file out.json
```

BLEU-1/2/4, ROUGE-1/2/L, METEOR, BERTScore-P/R/F1. Apply the BioBERT
monkey-patch above first if you need `dmis-lab/biobert-base-cased-v1.2`
specifically (the model actually used for the paper's reported BERT-F1
numbers) rather than `roberta-large`.

```bash
python metrics/leakage_audit.py \
  --ref_file results_a.jsonl \
  --compare_files results_a.jsonl results_b.jsonl \
  --compare_names configA configB \
  --output_md audit.md --output_json audit.json
```

Flags a sample as leaked if its reference answer appears verbatim, or
$\geq\!85\%$-token-contained, in its own question text (`--containment_threshold 0.85`,
the value used throughout the paper) — this reproduces the 295/1,287
(original split) and 230/1,287 (reconstructed split) leakage counts
reported in `sec:leakage`/`sec:param-matched`.

```bash
python metrics/bootstrap_ci.py --help   # patient-clustered bootstrap significance test
```

### Checkpoints already on Hugging Face

Three of the fresh re-run checkpoints from the parameter-matched
reproducibility check (Table "Per-seed breakdown", `tab:param-matched-perseed`)
are uploaded to `itachi2706/lgca-checkpoints` on the Hugging Face Hub,
under `checkpoints/default3zone_seed43`, `checkpoints/default3zone_seed44`,
and `checkpoints/default3zone_seed42_v2` (3-zone default layout, standard
protocol, `train_xattn_valsel.py`, no `LGCA_LAYOUT` override). Download
with `huggingface_hub.snapshot_download(repo_id="itachi2706/lgca-checkpoints", allow_patterns=["checkpoints/<name>/*"])`.

## RAG (retrieval-augmented) variant

`rag/build_rag_index.py` / `rag/build_clip_index.py` build BM25 / CLIP
retrieval indices over the training set; `rag/rag_retriever.py` queries
them; `rag/build_rag_dataset.py` / `rag/build_rag_train_data.py` /
`rag/build_rag_train_data_strict.py` build the RAG-aware training data
(the `_strict` variant additionally excludes same-patient and
near-duplicate-answer candidates from the retrieval context, used for
the stricter-filtering re-run discussed in the paper's Discussion
section on RAG-aware training-set contamination); `rag/run_inference_rag.py`
/ `rag/run_rag_inference.py` run Pre-RAG/Post-RAG inference. The paper's
finding: none of the BM25/CLIP configurations tested improved BERTScore
over the no-retrieval baseline on either split tested — report this as
"the specific configurations tested did not help," not as a general
finding against retrieval augmentation.

## External competitor (Qwen2-VL + LoRA) and English replication

`external_qwenvl/train_lora_qwenvl.py` + `external_qwenvl/run_inference_qwenvl.py`
fine-tune/evaluate Qwen2-VL-7B + LoRA as the independently fine-tuned
external competitor (`sec:external`). `external_qwenvl/qwenvl_xattn/`
is a from-scratch port of the LGCA cross-attention mechanism onto the
Qwen2-VL backbone, used for the English-language replication
(`sec:english`) to check whether the architecture's relative ordering
against LoRA holds on a different backbone/language — treat this as
supplementary, capacity-confounded evidence only (it does not repeat the
parameter-matched control), per the paper's own framing.

## What is *not* in this repository

- The clinical dataset (see above).
- Compiled/binary artifacts (`.pyc`, `__pycache__` — stripped).
- Exploratory/one-off scripts not needed to reproduce any reported
  number (debugging notebooks, scratch preprocessing variants, a few
  abandoned feature branches such as an anatomy-constrained-graph
  module that is not discussed in the paper).
- The full commit/experiment history behind every number in the paper —
  this release is the current, verified-correct script set, not an
  archive of every intermediate (and sometimes buggy) version tried
  along the way. Where a documented bug materially affected a past
  result, that is called out above and in `docs/GENERATE_BUG_INVESTIGATION.md`
  rather than silently omitted.
