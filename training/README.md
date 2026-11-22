# training

Training scripts for LGCA and the LoRA baselines compared against it.

| Script | What it trains |
|---|---|
| `train_xattn_valsel.py` | **LGCA, validation-selected checkpoint** — the protocol used for every LGCA number in the paper from the parameter-matched comparison onward. Evaluates every 250 steps, keeps the lowest-`eval_loss` checkpoint. Reads `LGCA_LAYOUT` (see root README) to pick the layer-placement schedule. |
| `train_xattn.py` | LGCA, earlier non-validation-selected version (`save_strategy="epoch"`). `train_xattn_valsel.py` extends this; use this one only if you specifically want the final-epoch checkpoint. |
| `train_xattn_learned_gate.py` | LGCA with a learned $\tanh$-gate instead of the fixed $\alpha{=}0.05$ coefficient — the cold-start-failure control described in the paper (`sec:learned-gate`). |
| `train_lora_baseline_v3.py` | LoRA fine-tuning of the same frozen backbone (`llava_mistral.py`). `--lora_r 256` with all 7 target projection modules gives ≈709M trainable parameters, matched against LGCA's ≈723M for the parameter-matched control. Includes an anti-EOS-degeneration penalty (`--first_token_eos_penalty`) — do not disable it or shorten the schedule (see root README for why). |

Seeds used throughout the paper: **42, 43, 44**. See the root
[README](../README.md) for exact commands, flags, and the data layout each
script expects (none of which is included in this repository — see
"Data (not included)" there).
