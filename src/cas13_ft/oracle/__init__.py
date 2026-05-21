from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from .cache import OracleSQLiteCache
from .cas13_identity import Cas13IdentityScorer
from .esm2_pll_scorer import ESM2PseudoPLLScorer
from .esmfold_scorer import ESMFoldScorer
from .filters import apply_hard_filters
from .progen3_scorer import ProGen3Scorer
from .reward import combine_oracle_scores
from .types import HardFilterResult, OracleConfig, OracleScore


def _config_hash(config: dict[str, Any]) -> str:
    raw = json.dumps(config, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dict_to_config(data: dict[str, Any]) -> OracleConfig:
    return OracleConfig(
        hard_filters=data.get("hard_filters", {}),
        naturalness=data.get("naturalness", {}),
        cas13_identity=data.get("cas13_identity", {}),
        structure=data.get("structure", {}),
        diversity=data.get("diversity", {}),
        reward=data.get("reward", {}),
        cache=data.get("cache", {}),
    )


def _simple_hamming_diversity(sequences: list[str]) -> list[float]:
    out = []
    for i, seq in enumerate(sequences):
        others = [other for j, other in enumerate(sequences) if j != i]
        if not others:
            out.append(0.0)
            continue
        vals = []
        for other in others:
            width = max(len(seq), len(other), 1)
            matches = sum(1 for a, b in zip(seq, other) if a == b)
            vals.append(1.0 - matches / float(width))
        out.append(float(mean(vals)))
    return out


class _MockNaturalnessScorer:
    def score_sequences(self, sequences: list[str], batch_size: int = 1) -> list[dict[str, float | int]]:
        rows = []
        for seq in sequences:
            text = str(seq or "")
            rare = sum(1 for ch in text if ch in "WCM") / max(1, len(text))
            mean_ll = -2.0 - rare - abs(len(text) - 1000) / 5000.0
            rows.append({"mean_log_likelihood": float(mean_ll), "nll": float(-mean_ll), "ppl": float(2.718281828 ** (-mean_ll)), "num_tokens": len(text)})
        return rows


class Cas13Oracle:
    def __init__(self, config: OracleConfig | dict[str, Any]):
        self.config = config if isinstance(config, OracleConfig) else _dict_to_config(config)
        self.cache = OracleSQLiteCache(self.config.cache.get("path", "data/oracle_cache.sqlite")) if self.config.cache.get("path") else None
        self.naturalness_scorer = self._build_naturalness_scorer()
        id_cfg = dict(self.config.cas13_identity)
        id_cfg.setdefault("length_range", self.config.hard_filters.get("length_range", [800, 1400]))
        self.identity_scorer = Cas13IdentityScorer(**id_cfg) if id_cfg.get("enabled", True) else Cas13IdentityScorer(mode="disabled")
        struct_cfg = dict(self.config.structure)
        struct_cfg.pop("enabled", None)
        norm = struct_cfg.pop("normalize", None)
        if norm is not None:
            struct_cfg["normalize"] = norm
        self.structure_scorer = ESMFoldScorer(**struct_cfg) if self.config.structure.get("enabled", False) else ESMFoldScorer(mode="disabled")

    @classmethod
    def from_config(cls, path_or_dict: str | Path | dict[str, Any]) -> "Cas13Oracle":
        if isinstance(path_or_dict, (str, Path)):
            with Path(path_or_dict).open("r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
            return cls(data)
        return cls(path_or_dict)

    def close(self) -> None:
        if self.cache is not None:
            self.cache.close()

    def _build_naturalness_scorer(self):
        cfg = self.config.naturalness
        if not cfg.get("enabled", True):
            return None
        scorer = cfg.get("scorer", "progen3")
        if scorer == "mock":
            return _MockNaturalnessScorer()
        if scorer == "esm2_pseudo_pll":
            return ESM2PseudoPLLScorer(mode=cfg.get("mode", "mock"), model_name=cfg.get("model_name", "facebook/esm2_t33_650M_UR50D"))
        if scorer == "progen3":
            return ProGen3Scorer(
                model_name=cfg.get("model_name", "Profluent-Bio/progen3-1b"),
                device=cfg.get("device", "auto"),
                dtype=cfg.get("dtype", "auto"),
                cache_dir=cfg.get("cache_dir"),
            )
        raise ValueError(f"Cas13Oracle naturalness.scorer unsupported: {scorer!r}")

    def _score_naturalness(self, sequences: list[str]) -> list[dict[str, Any] | None]:
        if self.naturalness_scorer is None:
            return [None for _ in sequences]
        cfg = self.config.naturalness
        batch_size = int(cfg.get("batch_size", 1))
        use_cache = bool(cfg.get("use_cache", False) and self.cache is not None)
        scorer_name = f"naturalness:{cfg.get('scorer', 'progen3')}"
        version = "v1"
        chash = _config_hash(cfg)
        results: list[dict[str, Any] | None] = [None] * len(sequences)
        misses: list[str] = []
        miss_indices: list[int] = []
        for i, seq in enumerate(sequences):
            cached = self.cache.get_cached(seq, scorer_name, version, chash) if use_cache else None
            if cached is None:
                misses.append(seq)
                miss_indices.append(i)
            else:
                results[i] = cached
        if misses:
            scored = self.naturalness_scorer.score_sequences(misses, batch_size=batch_size)
            for idx, seq, row in zip(miss_indices, misses, scored):
                if isinstance(row, dict):
                    payload = row
                else:
                    payload = {"mean_log_likelihood": row, "nll": None, "ppl": None, "num_tokens": len(seq)}
                results[idx] = payload
                if use_cache:
                    self.cache.set_cached(seq, scorer_name, version, chash, payload)
        return results

    def _score_structure(self, records: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
        cfg = self.config.structure
        if not cfg.get("enabled", False):
            return [None for _ in records]
        use_cache = bool(cfg.get("use_cache", False) and self.cache is not None)
        scorer_name = f"structure:{cfg.get('mode', 'disabled')}"
        version = "v1"
        chash = _config_hash(cfg)
        results: list[dict[str, Any] | None] = [None] * len(records)
        misses: list[dict[str, Any]] = []
        miss_indices: list[int] = []
        for i, record in enumerate(records):
            seq = str(record.get("sequence", ""))
            cached = self.cache.get_cached(seq, scorer_name, version, chash) if use_cache else None
            if cached is None:
                misses.append(record)
                miss_indices.append(i)
            else:
                results[i] = cached
        if misses:
            scored = self.structure_scorer.score_sequences(misses)
            for idx, record, row in zip(miss_indices, misses, scored):
                payload = row or {"ptm": None, "plddt_mean": None, "tm_to_reference": None, "metadata": {"structure_missing": True}}
                results[idx] = payload
                if use_cache:
                    self.cache.set_cached(str(record.get("sequence", "")), scorer_name, version, chash, payload)
        return results

    def score_records(self, records: list[dict[str, Any]]) -> list[OracleScore]:
        normalized = [{"id": row.get("id"), "sequence": str(row.get("sequence", "")).upper()} for row in records]
        hard_results = [apply_hard_filters(row["sequence"], self.config.hard_filters) for row in normalized]
        pass_indices = [i for i, result in enumerate(hard_results) if result.passed]
        pass_records = [normalized[i] for i in pass_indices]
        pass_sequences = [row["sequence"] for row in pass_records]

        naturalness = self._score_naturalness(pass_sequences) if pass_sequences else []
        identity = self.identity_scorer.score_sequences(pass_sequences) if pass_sequences else []
        structure = self._score_structure(pass_records) if pass_records else []
        diversity_all = _simple_hamming_diversity([row["sequence"] for row in normalized]) if self.config.diversity.get("enabled", True) else [None for _ in normalized]

        by_index: dict[int, dict[str, Any]] = {}
        for local, original in enumerate(pass_indices):
            by_index[original] = {
                "naturalness": naturalness[local],
                "identity": identity[local],
                "structure": structure[local],
            }

        scores: list[OracleScore] = []
        for i, record in enumerate(normalized):
            hard = hard_results[i]
            payload = by_index.get(i, {})
            nat_row = payload.get("naturalness") or {}
            identity_score = payload.get("identity")
            struct_row = payload.get("structure") or {}
            naturalness_score = nat_row.get("mean_log_likelihood") if isinstance(nat_row, dict) else nat_row
            plddt = struct_row.get("plddt_mean") if isinstance(struct_row, dict) else None
            ptm = struct_row.get("ptm") if isinstance(struct_row, dict) else None
            tm_ref = struct_row.get("tm_to_reference") if isinstance(struct_row, dict) else None
            combined = combine_oracle_scores(
                hard_filter=hard,
                naturalness_score=naturalness_score,
                cas13_identity_score=identity_score,
                plddt_mean=plddt,
                ptm=ptm,
                tm_to_reference=tm_ref,
                diversity_score=diversity_all[i],
                reward_config=self.config.reward,
                naturalness_config=self.config.naturalness.get("normalize", {}),
                cas13_identity_config=self.config.cas13_identity.get("normalize", {}),
                structure_config=self.config.structure.get("normalize", {}),
            )
            metadata = {
                **combined.get("metadata", {}),
                "hard_filter": {
                    "hepn_positions": hard.hepn_positions,
                    "low_complexity": hard.low_complexity,
                },
            }
            if isinstance(nat_row, dict):
                metadata["naturalness"] = {k: v for k, v in nat_row.items() if k != "mean_log_likelihood"}
            if isinstance(struct_row, dict):
                metadata["structure"] = struct_row.get("metadata", {})
                if struct_row.get("pdb_path") is not None:
                    metadata["pdb_path"] = struct_row.get("pdb_path")
            scores.append(
                OracleScore(
                    sequence=record["sequence"],
                    sequence_id=record.get("id"),
                    passed_hard_filters=hard.passed,
                    hard_fail_reasons=hard.reasons,
                    length=hard.length,
                    hepn_motif_count=len(hard.hepn_positions),
                    naturalness_score=float(naturalness_score) if naturalness_score is not None else None,
                    naturalness_z=combined["naturalness_z"],
                    cas13_identity_score=float(identity_score) if identity_score is not None else None,
                    cas13_identity_z=combined["cas13_identity_z"],
                    plddt_mean=float(plddt) if plddt is not None else None,
                    ptm=float(ptm) if ptm is not None else None,
                    tm_to_reference=float(tm_ref) if tm_ref is not None else None,
                    structure_z=combined["structure_z"],
                    diversity_score=float(diversity_all[i]) if diversity_all[i] is not None else None,
                    penalty=float(combined["penalty"]),
                    final_reward=float(combined["final_reward"]),
                    metadata=metadata,
                )
            )
        return scores


def oracle_score_to_dict(score: OracleScore) -> dict[str, Any]:
    return asdict(score)


__all__ = ["Cas13Oracle", "OracleScore", "OracleConfig", "HardFilterResult", "oracle_score_to_dict"]

