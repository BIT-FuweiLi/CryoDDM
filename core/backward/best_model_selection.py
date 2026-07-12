from pathlib import Path
import math
import shutil


BEST_MODEL_RULE = "post20_lowest_s2_loss_with_10_epoch_stability_v1"
MIN_BEST_EPOCH_EXCLUSIVE = 20
STABILITY_FOLLOWING_EPOCHS = 10
MAX_S2_RELATIVE_JUMP = 0.10


def select_best_epoch(
    losses,
    min_epoch_exclusive=MIN_BEST_EPOCH_EXCLUSIVE,
    stable_following_epochs=STABILITY_FOLLOWING_EPOCHS,
    max_relative_jump=MAX_S2_RELATIVE_JUMP,
):
    records = _normalize_losses(losses)
    by_epoch = {record["epoch"]: record for record in records}
    candidates = []
    for record in records:
        epoch = record["epoch"]
        if epoch <= min_epoch_exclusive:
            continue
        if _has_stable_following_epochs(
            by_epoch,
            epoch,
            stable_following_epochs,
            max_relative_jump,
        ):
            candidates.append(record)
    if not candidates:
        return None
    selected = min(candidates, key=lambda record: (record["s2_loss"], record["epoch"]))
    return {
        "epoch": selected["epoch"],
        "s2_loss": selected["s2_loss"],
        "rule": BEST_MODEL_RULE,
    }


def materialize_best_model(losses, model_dir):
    selection = select_best_epoch(losses)
    if selection is None:
        return None
    model_dir = Path(model_dir)
    source = model_dir / f"{selection['epoch']}.pth"
    if not source.is_file():
        raise FileNotFoundError(f"Selected epoch model does not exist: {source}")
    destination = model_dir / "best_model.pth"
    temporary = model_dir / "best_model.pth.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return selection


def _normalize_losses(losses):
    records = []
    for item in losses:
        try:
            epoch = int(item["epoch"])
            s2_loss = float(item["s2_loss"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(s2_loss):
            continue
        records.append({"epoch": epoch, "s2_loss": s2_loss})
    return sorted(records, key=lambda record: record["epoch"])


def _has_stable_following_epochs(by_epoch, epoch, following, max_relative_jump):
    candidate = by_epoch[epoch]["s2_loss"]
    previous = candidate
    for next_epoch in range(epoch + 1, epoch + following + 1):
        record = by_epoch.get(next_epoch)
        if record is None:
            return False
        current = record["s2_loss"]
        relative_change = abs(current - previous) / max(abs(previous), 1e-12)
        if relative_change > max_relative_jump:
            return False
        if candidate > 0 and current > candidate * (1 + max_relative_jump):
            return False
        previous = current
    return True
