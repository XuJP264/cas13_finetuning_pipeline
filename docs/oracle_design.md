# Cas13 Cascade Oracle Design

The cascade oracle is a configurable engineering scaffold for ranking candidate
Cas13-like proteins. It is not a functional validation assay.

Pipeline order:

1. Hard filters check amino acid alphabet, length range, HEPN `RxxxxH` motifs,
   and lightweight low-complexity signals.
2. PLM naturalness scoring uses ProGen3 or another configured protein language
   model. PPL or mean log-likelihood only measures model naturalness; it does
   not prove Cas13 function.
3. The Cas13 identity proxy estimates whether a sequence resembles Cas13 using
   a mock heuristic or a configured lightweight classifier.
4. ESMFold structure scoring uses pLDDT, pTM, and optional TM-to-reference as
   structure plausibility signals. These values do not prove nuclease activity.
5. The weighted reward aggregator combines z-scored terms, HEPN motif count,
   diversity, and soft penalties.

ESM-2 pseudo-PLL is included as an interface. Exact single-mask pseudo-PLL for
full-length Cas13 proteins is expensive because it masks each residue position;
future production use should prefer an OFS-style approximation or offline
precomputation.

Unit tests use mock and disabled scorers only. Real ProGen3, ESM-2, and ESMFold
models must be configured explicitly and are not downloaded by tests.

## Integration Boundary

The current RL training and debug entry points use the `cas13_rl` package:

- `scripts/02_rl_debug_mac.py` calls `cas13_rl.rl_trainer.run_from_config`.
- `scripts/12_train_rl_ppo.py` calls `cas13_rl.ppo.run_mock_ppo`.
- Existing online RL reward composition remains in `src/cas13_rl/reward.py`.

The cascade oracle added under `src/cas13_ft/oracle/` is currently an offline
candidate scoring and reranking tool, exposed by:

- `scripts/08_score_candidates_oracle.py`
- `cas13_rl.cascade_oracle`, a thin compatibility adapter that re-exports
  `cas13_ft.oracle.Cas13Oracle` for RL-side imports.

It is not yet wired into `src/cas13_rl/rl_trainer.py`. Keeping it separate lets
the project validate hard filters, mock identity scoring, structure plausibility,
and reward aggregation without changing PPO/debug trainer behavior.

There are therefore two oracle surfaces today:

- Legacy online RL oracle wrappers in `src/cas13_rl/oracle_*.py`, used by the
  RL trainer/debug scripts.
- New cascade oracle in `src/cas13_ft/oracle/`, recommended for offline
  candidate scoring, filtering, and reranking before training or evaluation.

For new offline candidate triage, prefer the cascade oracle. For current RL
debug/training runs, keep using the existing `cas13_rl` trainer path until a
separate integration task explicitly swaps the trainer oracle backend.

`src/cas13_rl/reward.py` was changed in an earlier RL reward-composer task to
implement ProteinRL-style property rewards. That change is separate from this
cascade oracle package. Compatibility tests keep the old `compute_rewards`
surface available, while the cascade oracle has its own aggregator in
`src/cas13_ft/oracle/reward.py`.
