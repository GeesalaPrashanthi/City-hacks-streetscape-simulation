import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Train an advanced sidewalk condition classifier.")
    parser.add_argument("--data-dir", default="dataset_masked", help="Dataset root with class folders.")
    parser.add_argument("--fallback-data-dir", default="dataset", help="Fallback dataset path if data-dir missing.")
    parser.add_argument("--model-out", default="sidewalk_classifier_advanced.pt", help="Checkpoint output path.")
    parser.add_argument("--results-out", default="training_results_advanced.json", help="Metrics JSON output path.")
    parser.add_argument("--arch", choices=["convnext_tiny", "efficientnet_v2_s"], default="convnext_tiny")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument("--lr-backbone", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-mode", choices=["fair", "stratified"], default="fair")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Used in stratified mode.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Used in stratified mode.")
    parser.add_argument(
        "--val-per-class",
        type=int,
        default=0,
        help="Used in fair mode. 0 means auto from minority class.",
    )
    parser.add_argument(
        "--test-per-class",
        type=int,
        default=0,
        help="Used in fair mode. 0 means auto from minority class.",
    )
    parser.add_argument(
        "--extra-test-ratio",
        type=float,
        default=0.15,
        help="Used in fair mode. Fraction of remaining samples per class reserved as extra real-world test.",
    )
    parser.add_argument(
        "--equal-train-per-class",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Used in fair mode. Force equal train count per class and move surplus majority samples to extra test."
        ),
    )
    parser.add_argument(
        "--weighted-sampler",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use weighted random sampling to reduce class imbalance during training.",
    )
    return parser.parse_args()


class PathDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_data_dir(primary: Path, fallback: Path):
    if primary.exists():
        return primary
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No dataset found at {primary} or {fallback}")


def group_by_class(samples):
    by_class = defaultdict(list)
    for path, label in samples:
        by_class[label].append((path, label))
    return by_class


def stratified_split(samples, val_ratio=0.15, test_ratio=0.15, seed=42):
    by_class = group_by_class(samples)
    rng = random.Random(seed)
    train_samples = []
    val_samples = []
    test_samples = []

    for label, label_samples in by_class.items():
        rng.shuffle(label_samples)
        n = len(label_samples)
        n_val = int(val_ratio * n)
        n_test = int(test_ratio * n)
        n_train = n - n_val - n_test

        train_samples.extend(label_samples[:n_train])
        val_samples.extend(label_samples[n_train:n_train + n_val])
        test_samples.extend(label_samples[n_train + n_val:])

        print(f"Class {label}: train={n_train} val={n_val} test={n_test}")

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    split_meta = {
        "mode": "stratified",
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    return train_samples, val_samples, test_samples, [], split_meta


def fair_split(
    samples,
    num_classes,
    val_per_class,
    test_per_class,
    extra_test_ratio,
    equal_train_per_class,
    seed=42,
):
    by_class = group_by_class(samples)
    rng = random.Random(seed)

    for label in range(num_classes):
        if label not in by_class:
            raise ValueError(f"Missing class label {label} in dataset.")
        rng.shuffle(by_class[label])

    min_count = min(len(items) for items in by_class.values())
    auto_eval = max(1, int(0.15 * min_count))
    if val_per_class <= 0:
        val_per_class = auto_eval
    if test_per_class <= 0:
        test_per_class = auto_eval

    if val_per_class + test_per_class >= min_count:
        raise ValueError(
            "val_per_class + test_per_class must be smaller than minority class count. "
            f"Got val={val_per_class} test={test_per_class} min_count={min_count}."
        )

    train_samples = []
    val_samples = []
    test_samples = []
    extra_test_samples = []
    remaining_by_class = {}

    for label in range(num_classes):
        items = by_class[label]
        val_cls = items[:val_per_class]
        test_cls = items[val_per_class:val_per_class + test_per_class]
        remaining_by_class[label] = items[val_per_class + test_per_class:]

        val_samples.extend(val_cls)
        test_samples.extend(test_cls)

    if equal_train_per_class:
        train_target = min(len(items) for items in remaining_by_class.values())
        for label in range(num_classes):
            remaining = remaining_by_class[label]
            train_cls = remaining[:train_target]
            extra_cls = remaining[train_target:]
            print(
                f"Class {label}: train={len(train_cls)} val={val_per_class} "
                f"test_bal={test_per_class} test_extra={len(extra_cls)}"
            )
            train_samples.extend(train_cls)
            extra_test_samples.extend(extra_cls)
    else:
        for label in range(num_classes):
            remaining = remaining_by_class[label]
            if len(remaining) <= 1:
                extra_n = 0
            else:
                extra_n = min(int(len(remaining) * extra_test_ratio), len(remaining) - 1)
            extra_cls = remaining[:extra_n]
            train_cls = remaining[extra_n:]
            print(
                f"Class {label}: train={len(train_cls)} val={val_per_class} "
                f"test_bal={test_per_class} test_extra={len(extra_cls)}"
            )
            train_samples.extend(train_cls)
            extra_test_samples.extend(extra_cls)

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)
    rng.shuffle(extra_test_samples)

    split_meta = {
        "mode": "fair",
        "min_class_count": min_count,
        "val_per_class": val_per_class,
        "test_per_class": test_per_class,
        "extra_test_ratio": extra_test_ratio,
        "equal_train_per_class": equal_train_per_class,
    }
    return train_samples, val_samples, test_samples, extra_test_samples, split_meta


def build_model(arch: str, num_classes: int):
    if arch == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )
        return model

    if arch == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported arch: {arch}")


def make_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.RandomPerspective(distortion_scale=0.22, p=0.25),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def count_labels(samples, num_classes):
    counts = [0] * num_classes
    for _, label in samples:
        counts[label] += 1
    return counts


def per_class_recall(correct, totals):
    recalls = []
    for c, t in zip(correct, totals):
        recalls.append(c / t if t > 0 else 0.0)
    return recalls


def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    per_cls_correct = [0] * num_classes
    per_cls_total = [0] * num_classes

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch = labels.size(0)
            total += batch
            total_loss += loss.item() * batch
            correct += (preds == labels).sum().item()

            for cls in range(num_classes):
                cls_mask = labels == cls
                cls_count = cls_mask.sum().item()
                if cls_count == 0:
                    continue
                per_cls_total[cls] += cls_count
                per_cls_correct[cls] += (preds[cls_mask] == labels[cls_mask]).sum().item()

    loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    recalls = per_class_recall(per_cls_correct, per_cls_total)
    balanced_acc = sum(recalls) / len(recalls) if recalls else 0.0

    return {
        "loss": loss,
        "acc": acc,
        "balanced_acc": balanced_acc,
        "per_class_correct": per_cls_correct,
        "per_class_total": per_cls_total,
        "per_class_recall": recalls,
    }


def metrics_to_dict(metrics, classes):
    return {
        "loss": metrics["loss"],
        "acc": metrics["acc"],
        "balanced_acc": metrics["balanced_acc"],
        "per_class_recall": {classes[i]: metrics["per_class_recall"][i] for i in range(len(classes))},
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    data_dir = resolve_data_dir(Path(args.data_dir), Path(args.fallback_data_dir))
    print(f"Using data dir: {data_dir}")

    base_ds = datasets.ImageFolder(data_dir)
    classes = base_ds.classes
    samples = [(Path(path), label) for path, label in base_ds.samples]
    num_classes = len(classes)

    print(f"Classes: {classes}")
    print(f"Total images: {len(samples)}")

    if args.split_mode == "fair":
        train_samples, val_samples, test_samples, extra_test_samples, split_meta = fair_split(
            samples=samples,
            num_classes=num_classes,
            val_per_class=args.val_per_class,
            test_per_class=args.test_per_class,
            extra_test_ratio=args.extra_test_ratio,
            equal_train_per_class=args.equal_train_per_class,
            seed=args.seed,
        )
    else:
        train_samples, val_samples, test_samples, extra_test_samples, split_meta = stratified_split(
            samples=samples,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    print(
        f"Train: {len(train_samples)} | Val: {len(val_samples)} | "
        f"Test(bal/main): {len(test_samples)} | Test(extra): {len(extra_test_samples)}"
    )

    train_tf, eval_tf = make_transforms(args.img_size)
    train_ds = PathDataset(train_samples, transform=train_tf)
    val_ds = PathDataset(val_samples, transform=eval_tf)
    test_ds = PathDataset(test_samples, transform=eval_tf)
    extra_test_ds = PathDataset(extra_test_samples, transform=eval_tf) if extra_test_samples else None

    train_class_counts = count_labels(train_samples, num_classes)
    val_class_counts = count_labels(val_samples, num_classes)
    test_class_counts = count_labels(test_samples, num_classes)
    print(f"Train class distribution: {dict(zip(classes, train_class_counts))}")
    print(f"Val class distribution:   {dict(zip(classes, val_class_counts))}")
    print(f"Test class distribution:  {dict(zip(classes, test_class_counts))}")
    if extra_test_samples:
        extra_counts = count_labels(extra_test_samples, num_classes)
        print(f"Extra test distribution:  {dict(zip(classes, extra_counts))}")

    if args.weighted_sampler:
        sample_weights = [1.0 / max(train_class_counts[label], 1) for _, label in train_samples]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    extra_test_loader = (
        DataLoader(extra_test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if extra_test_ds is not None
        else None
    )

    model = build_model(args.arch, num_classes=num_classes).to(device)

    class_weights = torch.tensor(
        [1.0 / max(count, 1) ** 0.5 for count in train_class_counts],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_balanced_acc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_total += batch_size
            running_correct += (logits.argmax(dim=1) == labels).sum().item()

        scheduler.step()

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["acc"],
                "val_balanced_acc": val_metrics["balanced_acc"],
            }
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.3f} "
            f"Balanced: {val_metrics['balanced_acc']:.3f}"
        )

        if val_metrics["balanced_acc"] > best_val_balanced_acc:
            best_val_balanced_acc = val_metrics["balanced_acc"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "arch": args.arch,
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "val_balanced_acc": best_val_balanced_acc,
                    "img_size": args.img_size,
                    "split_mode": args.split_mode,
                },
                args.model_out,
            )
            print(f"  -> Best model saved (val_balanced_acc={best_val_balanced_acc:.3f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  -> Early stopping (no improvement for {args.patience} epochs)")
                break

    checkpoint = torch.load(args.model_out, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    print(
        f"\nMain test | Loss: {test_metrics['loss']:.4f} "
        f"| Acc: {test_metrics['acc']:.3f} | Balanced: {test_metrics['balanced_acc']:.3f}"
    )
    print("Per-class test recall:")
    for idx, cls in enumerate(classes):
        correct = test_metrics["per_class_correct"][idx]
        total = test_metrics["per_class_total"][idx]
        recall = test_metrics["per_class_recall"][idx]
        print(f"  {cls}: {correct}/{total} ({recall:.3f})")

    extra_test_metrics = None
    if extra_test_loader is not None:
        extra_test_metrics = evaluate(model, extra_test_loader, criterion, device, num_classes)
        print(
            f"\nExtra test | Loss: {extra_test_metrics['loss']:.4f} "
            f"| Acc: {extra_test_metrics['acc']:.3f} | Balanced: {extra_test_metrics['balanced_acc']:.3f}"
        )

    results = {
        "arch": args.arch,
        "data_dir": str(data_dir),
        "classes": classes,
        "split_mode": args.split_mode,
        "split_meta": split_meta,
        "best_val_balanced_acc": best_val_balanced_acc,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "test_size": len(test_samples),
        "extra_test_size": len(extra_test_samples),
        "train_class_distribution": dict(zip(classes, train_class_counts)),
        "val_class_distribution": dict(zip(classes, val_class_counts)),
        "test_class_distribution": dict(zip(classes, test_class_counts)),
        "test_metrics": metrics_to_dict(test_metrics, classes),
        "extra_test_metrics": metrics_to_dict(extra_test_metrics, classes) if extra_test_metrics else None,
        "history": history,
    }
    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Best checkpoint: {args.model_out}")
    print(f"Results saved: {args.results_out}")


if __name__ == "__main__":
    main()
