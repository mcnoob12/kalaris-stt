"""Fine-tune a NeMo transducer. Run from the training environment (.venv-train).

Stages run in order, each starting from the last one's .nemo file:
    head: new tokenizer, decoder and joint; encoder frozen
    full: everything trains, with lower learning rates near the input

Usage:
    .venv-train/bin/python training/finetune.py training/configs/smoke.yaml --stage head
    .venv-train/bin/python training/finetune.py training/configs/smoke.yaml --stage full
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf, open_dict


def restore_tdt_loss(model) -> None:
    """Put the TDT loss back after change_vocabulary.

    NeMo 3.0's change_vocabulary always rebuilds a plain RNNTLoss, sized without
    the duration outputs. TDT models would then train on the wrong loss with no
    error, so this rebuilds it the way the model's constructor does.
    """
    print(f"Loss before rebuild: {model.loss._loss.__class__.__name__}, blank {model.loss._loss.blank}")
    loss_name, loss_kwargs = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    num_classes = model.joint.num_classes_with_blank - 1
    if loss_name == "tdt":
        num_classes -= model.joint.num_extra_outputs
    model.loss = RNNTLoss(
        num_classes=num_classes,
        loss_name=loss_name,
        loss_kwargs=loss_kwargs,
        reduction=model.cfg.get("rnnt_reduction", "mean_batch"),
    )
    if model.joint.fuse_loss_wer or (
        model.decoding.joint_fused_batch_size is not None and model.decoding.joint_fused_batch_size > 0
    ):
        model.joint.set_loss(model.loss)
        model.joint.set_wer(model.wer)
    print(f"Loss: {loss_name}, {num_classes} classes, {model.joint.num_extra_outputs} extra outputs")


def check_tdt_setup(model) -> None:
    """Stop before any training if a TDT model isn't set up as TDT."""
    loss_name, _ = model.extract_rnnt_loss_cfg(model.cfg.get("loss", None))
    if loss_name != "tdt":
        return
    expected = model.joint.num_classes_with_blank - 1 - model.joint.num_extra_outputs
    implementation = model.loss._loss.__class__.__name__
    if "TDT" not in implementation or model.loss._loss.blank != expected:
        raise SystemExit(
            f"TDT setup is wrong: loss is {implementation}, blank "
            f"{model.loss._loss.blank}, expected {expected}. Stopping before training."
        )


def layerwise_param_groups(model, base_lr: float, decay: float) -> list[dict]:
    """Lower learning rates toward the input. The decoder and joint get the full rate."""
    layers = list(model.encoder.layers)
    groups = [{"params": list(model.encoder.pre_encode.parameters()), "lr": base_lr * decay ** (len(layers) + 1)}]
    for index, layer in enumerate(layers):
        groups.append({"params": list(layer.parameters()), "lr": base_lr * decay ** (len(layers) - index)})
    known = {id(p) for group in groups for p in group["params"]}
    groups.append({"params": [p for p in model.parameters() if id(p) not in known], "lr": base_lr})
    return [g for g in groups if g["params"]]


class FrozenEncoderEval(Callback):
    """Keep a frozen encoder in eval mode, since Lightning switches everything to train mode."""

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        pl_module.encoder.eval()


class CollapseGuard(Callback):
    """Stop paying for a run whose model has stopped learning to transcribe.

    A public Parakeet fine-tune collapsed into repeating one token at 109.5% WER
    after a tokenizer swap. WER over the limit after the patience runs out means
    the same failure, so the run stops instead of burning GPU hours.
    """

    def __init__(self, limit: float, patience: int, metric: str = "val_wer"):
        self.limit, self.patience, self.metric, self.seen = limit, patience, metric, 0

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking or self.metric not in trainer.callback_metrics:
            return
        self.seen += 1
        wer = float(trainer.callback_metrics[self.metric]) * 100
        print(f"validation {self.seen}: {self.metric} {wer:.2f}%")
        if self.seen >= self.patience and wer >= self.limit:
            print(f"CollapseGuard: {wer:.1f}% after {self.seen} validations. Stopping.")
            trainer.should_stop = True


class LossLog(Callback):
    """Print training loss, because a flat WER alone can't separate slow from broken."""

    def __init__(self, every: int):
        self.every = every

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.global_step % self.every == 0 and isinstance(outputs, dict) and "loss" in outputs:
            print(f"step {trainer.global_step}: loss {float(outputs['loss']):.3f}", flush=True)


def data_config(manifest: str, data: dict, *, shuffle: bool) -> dict:
    return {
        "manifest_filepath": manifest,
        "sample_rate": 16000,
        "batch_size": data["batch_size"],
        "shuffle": shuffle,
        "num_workers": data.get("num_workers", 2),
        "pin_memory": False,
        "use_start_end_token": False,
        "trim_silence": False,
        "max_duration": data.get("max_duration", 20.0),
        "min_duration": 0.1,
    }


def load_model(stage: str, config: dict, output: Path):
    if stage == "head":
        base = config["base_model"]
        model = ASRModel.restore_from(base, map_location="cpu") if Path(base).exists() else ASRModel.from_pretrained(base, map_location="cpu")
        model.change_vocabulary(new_tokenizer_dir=config["tokenizer_dir"], new_tokenizer_type="bpe")
        restore_tdt_loss(model)
        model.encoder.freeze()
    else:
        start = config.get("init_from") or str(output / "head.nemo")
        model = ASRModel.restore_from(start, map_location="cpu")
        restore_tdt_loss(model)
        model.encoder.unfreeze()
    check_tdt_setup(model)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--stage", required=True, choices=["head", "full"])
    parser.add_argument("--resume", help="Lightning checkpoint to resume from")
    parser.add_argument("--max-seconds", type=int, help="Stop cleanly after this much wall-clock time")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stage = config["stages"][args.stage]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.get("seed", 0))
    model = load_model(args.stage, config, output)
    if not config.get("spec_augment", True):
        model.spec_augmentation = None

    model.setup_training_data(OmegaConf.create(data_config(config["train_manifest"], config["data"], shuffle=True)))
    model.setup_validation_data(OmegaConf.create(data_config(config["dev_manifest"], config["data"], shuffle=False)))
    with open_dict(model.cfg):
        model.cfg.train_ds = OmegaConf.create(data_config(config["train_manifest"], config["data"], shuffle=True))
        model.cfg.validation_ds = OmegaConf.create(data_config(config["dev_manifest"], config["data"], shuffle=False))

    if args.stage == "full" and stage.get("layer_decay"):
        groups = layerwise_param_groups(model, stage["lr"], stage["layer_decay"])
        model.setup_optimizer_param_groups = lambda: setattr(model, "_optimizer_param_groups", groups)

    # A constant or cosine schedule, since the stock warmup can reset to a high rate.
    model.setup_optimization(OmegaConf.create({
        "name": "adamw",
        "lr": stage["lr"],
        "weight_decay": stage.get("weight_decay", 1e-3),
        "betas": [0.9, 0.98],
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": stage.get("warmup_steps", 0),
            "min_lr": stage.get("min_lr", 1e-6),
            "max_steps": stage["max_steps"],
        },
    }))

    callbacks = [
        CollapseGuard(config["collapse_guard"]["wer_limit"], config["collapse_guard"]["patience"]),
        LossLog(stage.get("log_every", 10)),
        ModelCheckpoint(dirpath=output / f"checkpoints_{args.stage}", save_last=True, every_n_train_steps=stage["checkpoint_every"], save_top_k=0),
    ]
    if args.stage == "head":
        callbacks.append(FrozenEncoderEval())

    trainer = pl.Trainer(
        accelerator=config.get("accelerator", "auto"),
        devices=1,
        precision=config.get("precision", "32-true"),
        max_steps=stage["max_steps"],
        max_epochs=-1,
        max_time={"seconds": args.max_seconds} if args.max_seconds else None,
        val_check_interval=None,
        check_val_every_n_epoch=stage.get("validate_every_epochs", 1),
        log_every_n_steps=stage.get("log_every", 10),
        gradient_clip_val=1.0,
        accumulate_grad_batches=config["data"].get("accumulate", 1),
        callbacks=callbacks,
        enable_progress_bar=False,
        logger=False,
    )
    model.set_trainer(trainer)
    trainer.fit(model, ckpt_path=args.resume)

    path = output / f"{args.stage}.nemo"
    model.save_to(str(path))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
