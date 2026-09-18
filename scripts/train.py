"""Fine-tune a GLiNER2 boundary extractor (e.g. newmindai/Mursit-Base) on GLiNER2 JSONL.

Everything comes from a JSON config (see configs/recipes/*.json):
  model     -> ExtractorConfig (encoder, boundary_head copied from a checkpoint + overrides,
               token_pooling, tokenization, add_bos, attn_implementation, max_len)
  training  -> gliner2.training.trainer.TrainingConfig kwargs
  eval      -> batch_size / threshold for the span-F1 hook (scripts/span_metrics.py)
  warm_start_head -> optional checkpoint whose non-encoder tensors initialise the head
  model.special_token_init -> "mean" (default: what resize_token_embeddings gives), "random" or
               "gliner25": how the ten schema-token rows ([E], [R], [C], [SEP_TEXT], ...) are initialised
               when the encoder is not gliner2.5's own
  training.schema_token_lr -> train those ten rows in their own optimizer group at this learning
               rate (instead of encoder_lr); the update is folded into the embedding matrix after
               every step, so checkpoints stay ordinary
The trainer only knows cuda/cpu; DeviceTrainer places the model on --device (mps) itself,
mirroring the trainer's own CPU branch, without touching gliner2/training.

Usage:
  python scripts/train.py --config configs/recipes/entities_kvkk_labeldiv_6ep.json \\
      --train datasets/kvkk/pilot/train_50000_labeldiv.jsonl --eval datasets/kvkk/pilot/validation_2000.jsonl \\
      --device auto --out results/entities --run-name my-run
  --from-pretrained overrides model.from_pretrained in the config (chain recipes without editing JSON);
  --device auto = cuda if available, else mps, else cpu.
  --set key=value overrides any training key (repeatable).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
import types
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("train_pilot")


def load_boundary_head(source_dir: Path, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    head = dict(json.loads((Path(source_dir) / "config.json").read_text(encoding="utf-8"))["boundary_head"])
    head.update(overrides or {})
    return head


def build_extractor_config(model_section: Mapping[str, Any]):
    from gliner2 import ExtractorConfig

    section = dict(model_section)
    head = load_boundary_head(section.pop("boundary_head_from"), section.pop("boundary_head_overrides", None))
    return ExtractorConfig(architecture="boundary", boundary_head=head, **section)


def build_model(config, seed: int):
    import torch

    from gliner2.inference.engine import BoundaryExtractor

    torch.manual_seed(seed)
    return BoundaryExtractor(config)


def warm_start_head(model, checkpoint_dir: Path) -> tuple[int, int]:
    """Copy every non-encoder tensor of a saved extractor into ``model`` (shape must match)."""
    from safetensors.torch import load_file

    state = load_file(str(Path(checkpoint_dir) / "model.safetensors"))
    target = model.state_dict()
    loaded, skipped = 0, 0
    subset = {}
    for key, value in state.items():
        if key.startswith("encoder."):
            continue
        if key in target and target[key].shape == value.shape:
            subset[key] = value
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(subset, strict=False)
    return loaded, skipped


def schema_token_ids(model) -> list:
    """Vocabulary ids of the processor's added schema tokens ([SEP_STRUCT], [SEP_TEXT], [P], [C], [E], [R], ...)."""
    tok = model.processor.tokenizer
    ids = [tok.convert_tokens_to_ids(t) for t in (tok.additional_special_tokens or [])]
    return sorted(i for i in ids if i is not None and i >= 0)


def schema_token_stats(model) -> dict[str, Any]:
    """How distinct the schema-token rows are: norms, cosine to the mean embedding, pairwise cosine."""
    import torch

    ids = schema_token_ids(model)
    W = model.encoder.get_input_embeddings().weight.detach().float()
    base = min(ids)
    reg = W[:base]
    rows = W[ids]
    mean = reg.mean(0)
    cos_mean = torch.nn.functional.cosine_similarity(rows, mean.expand_as(rows), dim=1)
    pair = torch.nn.functional.cosine_similarity(rows[0].expand_as(rows[1:]), rows[1:], dim=1)
    return {
        "ids": ids,
        "regular_row_norm": round(reg.norm(dim=1).mean().item(), 4),
        "row_norms": [round(x, 4) for x in rows.norm(dim=1).tolist()],
        "cos_to_mean": [round(x, 4) for x in cos_mean.tolist()],
        "cos_first_to_others": [round(x, 4) for x in pair.tolist()],
    }


def init_schema_tokens(
    model, mode: str, seed: int, source: Path = PROJECT_ROOT / "models/gliner2.5-multi-v1"
) -> dict[str, Any]:
    """Re-initialise the schema-token rows of the encoder embedding.

    "mean"     leave what transformers' resize_token_embeddings produced (mean of the vocabulary plus
               noise): every schema row is the same vector — the state every Mursit checkpoint so far
               ended in, because encoder_lr 1e-5 never moved them
    "random"   N(0, std of the regular rows): distinct rows
    "gliner25" copy gliner2.5-multi-v1's trained rows for the same token names (needs the same hidden
               size); different embedding space, but distinct and structured
    """
    import torch

    ids = schema_token_ids(model)
    emb = model.encoder.get_input_embeddings()
    W = emb.weight
    base = min(ids)
    info: dict[str, Any] = {"mode": mode, "ids": ids}
    if mode == "mean":
        return info
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        if mode == "random":
            std = W[:base].float().std().item()
            W[ids] = (torch.randn(len(ids), W.shape[1], generator=g) * std).to(W.dtype)
            info["std"] = round(std, 5)
        elif mode == "gliner25":
            from safetensors import safe_open

            names = [model.processor.tokenizer.convert_ids_to_tokens(i) for i in ids]
            src_tok = json.loads((Path(source) / "tokenizer.json").read_text(encoding="utf-8"))
            src_ids = {t["content"]: t["id"] for t in src_tok.get("added_tokens", [])}
            with safe_open(str(Path(source) / "model.safetensors"), "pt") as f:
                key = next(k for k in f.keys() if k.endswith("embeddings.word_embeddings.weight"))
                src = f.get_tensor(key)
            if src.shape[1] != W.shape[1]:
                raise ValueError(f"special_token_init=gliner25 needs hidden {src.shape[1]}, encoder has {W.shape[1]}")
            missing = [n for n in names if n not in src_ids]
            if missing:
                raise ValueError(f"schema tokens not in {source}: {missing}")
            W[ids] = src[[src_ids[n] for n in names]].to(W.dtype)
            info["source"] = str(source)
        else:
            raise ValueError(f"unknown special_token_init {mode!r}")
    return info


class DeviceTrainer:
    """Factory so the import stays lazy: DeviceTrainer(model, config, device=..., log_dir=..., tensorboard=...)
    returns an ExtractorTrainer subclass that (a) places the model on the requested device and
    (b) mirrors every _log_metrics call into <log_dir>/metrics.jsonl and, if the tensorboard
    package is installed and tensorboard=True, into <log_dir>/tensorboard. Training logic untouched."""

    def __new__(
        cls,
        model,
        config,
        device: str | None = None,
        log_dir: Path | None = None,
        tensorboard: bool = False,
        schema_token_lr: float | None = None,
        **kwargs,
    ):
        import torch

        from gliner2.training.trainer import ExtractorTrainer

        writer = None
        # Only rank 0 logs scalars, so only rank 0 should create the writer —
        # otherwise every DDP rank drops an empty events file into the run dir.
        if tensorboard and log_dir is not None and int(os.environ.get("RANK", 0)) == 0:
            try:
                from torch.utils.tensorboard import SummaryWriter

                writer = SummaryWriter(log_dir=str(Path(log_dir) / "tensorboard"))
            except ImportError:
                logger.warning("tensorboard is not installed (pip install tensorboard); skipping TensorBoard logging")

        class _DeviceTrainer(ExtractorTrainer):
            requested_device = device
            metrics_path = (Path(log_dir) / "metrics.jsonl") if log_dir is not None else None
            tb_writer = writer
            schema_lr = schema_token_lr

            def _create_optimizer(self):
                opt = super()._create_optimizer()
                if self.schema_lr is None:
                    return opt
                # The ten schema-token rows live inside the encoder embedding matrix, so they sit in
                # the encoder_lr group. A row slice cannot be its own parameter, so: a zero delta over
                # those rows is added in a forward hook (gradients reach the delta), the delta gets
                # its own AdamW group at schema_lr, and after every optimizer step it is folded into
                # the matrix and zeroed. Net effect: those rows train at schema_lr; the saved model
                # is an ordinary checkpoint.
                emb = self.model.encoder.get_input_embeddings()
                ids = schema_token_ids(self.model)
                base = min(ids)
                idx = torch.tensor(ids, device=emb.weight.device)
                delta = torch.nn.Parameter(
                    torch.zeros(len(ids), emb.weight.shape[1], device=emb.weight.device, dtype=emb.weight.dtype)
                )

                def add_delta(module, args, out):
                    tok = args[0]
                    mask = tok >= base
                    if not bool(mask.any()):
                        return out
                    out = out.clone()
                    out[mask] = out[mask] + delta.to(out.dtype)[tok[mask] - base]
                    return out

                emb.register_forward_hook(add_delta)
                opt.add_param_group({"params": [delta], "lr": self.schema_lr, "weight_decay": 0.0})
                orig_step = opt.step

                def step(self_opt, closure=None):
                    result = orig_step(closure) if closure is not None else orig_step()
                    with torch.no_grad():
                        emb.weight.data[idx] += delta.data.to(emb.weight.dtype)
                        delta.data.zero_()
                    return result

                # bound method, because torch's LRScheduler wraps optimizer.step via step.__func__
                opt.step = types.MethodType(step, opt)
                self.schema_token_delta = delta
                logger.info(
                    "schema-token rows %s train at lr %s in their own group (folded into the embedding after every step)",
                    ids,
                    self.schema_lr,
                )
                return opt

            def _setup_device(self):
                if self.requested_device in (None, "auto", "cpu", "cuda"):
                    return super()._setup_device()
                self.device = torch.device(self.requested_device)
                self.is_distributed = False
                if self.config.fp16 or self.config.bf16:
                    logger.warning("Mixed precision disabled on %s", self.device)
                    self.config.fp16 = False
                    self.config.bf16 = False
                self.model.to(self.device)
                self.model.float()

            def _log_metrics(self, metrics, prefix: str = ""):
                super()._log_metrics(metrics, prefix)
                if hasattr(metrics, "to_dict"):
                    metrics = metrics.to_dict()
                if not metrics or not self.is_main_process:
                    return
                numeric = {
                    k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                step = int(getattr(self, "global_step", 0))
                if self.metrics_path is not None:
                    self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(
                            json.dumps(
                                {
                                    "time": dt.datetime.now().isoformat(timespec="seconds"),
                                    "step": step,
                                    "prefix": prefix,
                                    **numeric,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                if self.tb_writer is not None:
                    for key, value in numeric.items():
                        self.tb_writer.add_scalar(f"{prefix}/{key}" if prefix else key, value, step)
                    self.tb_writer.flush()

        return _DeviceTrainer(model, config, **kwargs)


def guard_metrics(hook, metric_for_best: str = "exact_f1_micro", greater_is_better: bool = True):
    """Never let a metrics-hook crash kill a multi-hour training run.

    ExtractorTrainer calls ``compute_metrics`` inside ``_evaluate``; an exception
    there propagates out of ``train()`` before any checkpoint is saved (run D lost
    2000 steps to exactly this). On failure, log the full traceback and report the
    worst possible value for ``metric_for_best`` instead: training continues, and
    the trainer neither falls back to comparing eval_loss against F1 values nor
    overwrites the ``best`` checkpoint on a broken eval.
    """
    sentinel = float("-inf") if greater_is_better else float("inf")

    def guarded(model, eval_dataset):
        try:
            return hook(model, eval_dataset)
        except Exception:
            guarded.failures += 1
            logger.exception("metrics hook failed; continuing training without span metrics for this eval")
            if metric_for_best.startswith("eval_"):
                return {}  # the trainer computed this metric itself; never clobber it
            return {metric_for_best: sentinel}

    guarded.failures = 0
    return guarded


def run(args) -> dict[str, Any]:
    import torch
    from span_metrics import make_compute_metrics

    from gliner2.training.trainer import TrainingConfig

    # torchrun sets LOCAL_RANK/RANK; without torchrun both default to single-process.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main = int(os.environ.get("RANK", 0)) == 0

    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    run_name = args.run_name or spec["run_name"]
    out_dir = Path(args.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(spec.get("seed", 42))

    model_spec = dict(spec["model"])
    special_token_init = model_spec.pop("special_token_init", "mean")
    if getattr(args, "from_pretrained", None):
        model_spec["from_pretrained"] = str(args.from_pretrained)
    pretrained = model_spec.get("from_pretrained")
    if pretrained:
        # Full-checkpoint fine-tune (encoder + trained head together), e.g. the
        # original gliner2.5-multi-v1 — as opposed to the encoder-swap path below,
        # which builds a fresh encoder and only warm-starts head tensors.
        import torch

        from gliner2 import AutoExtractor

        torch.manual_seed(seed)
        model = AutoExtractor.from_pretrained(pretrained)
        config = model.config
        warm_info = {"from": str(pretrained), "mode": "full checkpoint (encoder + head)"}
        logger.info("loaded full pretrained extractor from %s", pretrained)
    else:
        config = build_extractor_config(model_spec)
        model = build_model(config, seed)
        warm = args.warm_start_head or spec.get("warm_start_head")
        warm_info = None
        if warm:
            loaded, skipped = warm_start_head(model, Path(warm))
            warm_info = {"from": str(warm), "tensors_loaded": loaded, "tensors_skipped": skipped}
            logger.info("warm-started head from %s: %d tensors loaded, %d skipped", warm, loaded, skipped)

    schema_before = schema_token_stats(model)
    init_info = init_schema_tokens(model, special_token_init, seed)
    schema_init = schema_token_stats(model)
    logger.info(
        "schema tokens: init=%s | cos(first row, others) before %s -> after init %s",
        special_token_init,
        schema_before["cos_first_to_others"][:3],
        schema_init["cos_first_to_others"][:3],
    )

    training = dict(spec["training"])
    for item in getattr(args, "set", None) or []:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        training[key] = value
        logger.info("training override from --set: %s = %r", key, value)
    schema_token_lr = training.pop("schema_token_lr", None)
    if args.device != "cuda" and (training.get("bf16") or training.get("fp16")):
        logger.warning(
            "device %s: mixed precision is CUDA-only, training in fp32 (config asked for bf16=%s fp16=%s)",
            args.device,
            training.get("bf16"),
            training.get("fp16"),
        )
        training["bf16"] = False
        training["fp16"] = False
    training["output_dir"] = str(out_dir / "checkpoints")
    if args.max_steps is not None:
        training["max_steps"] = args.max_steps
    training.setdefault("seed", seed)
    if local_rank >= 0:
        training["local_rank"] = local_rank  # activates the trainer's DDP path (nccl + DistributedSampler)
    tconfig = TrainingConfig(**training)
    eval_spec = spec.get("eval", {})
    hook = make_compute_metrics(
        batch_size=int(eval_spec.get("batch_size", 16)),
        threshold=float(eval_spec.get("threshold", 0.5)),
        errors_path=(out_dir / "eval_errors.jsonl") if is_main else None,
    )
    guarded_hook = guard_metrics(
        hook, metric_for_best=tconfig.metric_for_best, greater_is_better=tconfig.greater_is_better
    )
    trainer = DeviceTrainer(
        model,
        tconfig,
        device=args.device,
        log_dir=out_dir,
        tensorboard=args.tensorboard,
        schema_token_lr=schema_token_lr,
        compute_metrics=guarded_hook,
    )

    # sanity: what the encoder sees for the first training record
    first = json.loads(Path(args.train).open(encoding="utf-8").readline())
    labels = list(first["output"]["entities"])
    batch = model.processor.collate_fn_inference([(first["input"], {"entities": {l: "" for l in labels}})])
    pieces = model.processor.tokenizer.convert_ids_to_tokens(batch.input_ids[0].tolist())
    n_params = sum(p.numel() for p in model.parameters())
    info = {
        "run_name": run_name,
        "config_file": str(args.config),
        "train": str(args.train),
        "eval": str(args.eval),
        "device": str(trainer.device),
        "seed": seed,
        "params": n_params,
        "encoder": type(model.encoder).__name__,
        "attn": getattr(model.encoder.config, "_attn_implementation", None),
        "tokenization": model.processor.tokenization,
        "add_bos": model.processor.add_bos,
        "token_pooling": model.processor.token_pooling,
        "first_record_pieces": pieces[:40],
        "warm_start": warm_info,
        "extractor_config": config.to_dict(),
        "training_config": tconfig.__dict__,
        "schema_tokens": {
            "init": init_info,
            "lr": schema_token_lr,
            "before_init": schema_before,
            "after_init": schema_init,
        },
        "started": dt.datetime.now().isoformat(timespec="seconds"),
    }
    if is_main:
        (out_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info(
        "run %s | device %s | params %s | encoder %s (%s) | pieces %s",
        run_name,
        trainer.device,
        f"{n_params:,}",
        info["encoder"],
        info["attn"],
        pieces[:16],
    )

    started = time.time()
    trainer.train(train_data=str(args.train), eval_data=str(args.eval) if args.eval else None)
    elapsed = time.time() - started
    info["schema_tokens"]["after_training"] = schema_token_stats(model)
    result = {
        "elapsed_s": round(elapsed, 1),
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
        "metrics_hook_failures": guarded_hook.failures,  # >0 means last_report may be stale
        "last_report": hook.last_report,
    }
    info.update(result)
    if is_main:
        (out_dir / "run.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    logger.info("done in %.1f s", elapsed)
    if hook.last_report:
        m = hook.last_report["micro"]
        logger.info("last eval: micro P %.3f R %.3f F1 %.3f", m["precision"], m["recall"], m["f1"])
    return info


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["mps", "cpu", "cuda", "auto"],
        help="auto = cuda if available, else mps, else cpu",
    )
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "pilot")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--warm-start-head", type=Path, default=None)
    parser.add_argument(
        "--from-pretrained",
        type=Path,
        default=None,
        help="checkpoint dir; overrides model.from_pretrained in the config",
    )
    parser.add_argument(
        "--tensorboard", action="store_true", help="also write TensorBoard scalars (needs the tensorboard package)"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a training key, e.g. --set gradient_accumulation_steps=4 --set max_len=512 (repeatable)",
    )
    args = parser.parse_args(argv)
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.device = resolve_device(args.device)
    run(args)


def resolve_device(device: str) -> str:
    """'auto' -> cuda / mps / cpu (the library's own auto-detection never picks MPS)."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
