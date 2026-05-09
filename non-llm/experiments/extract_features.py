from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from src.utils import ensure_dir, save_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def resolve_data_dir(value: str) -> Path:
    if value != "auto":
        return Path(value)
    candidates = [
        Path("data/ytfaces/aligned_images_DB"),
        Path("data/aligned_images_DB"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_clip_dirs(data_dir: Path) -> list[Path]:
    clips: list[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_dir():
            continue
        if any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir()):
            clips.append(path)
    return clips


def select_clip_dirs(
    clip_dirs: list[Path],
    data_dir: Path,
    max_subjects: int | None,
    max_clips_per_subject: int | None,
) -> list[Path]:
    by_subject: dict[str, list[Path]] = {}
    for clip_dir in clip_dirs:
        rel = clip_dir.relative_to(data_dir)
        if not rel.parts:
            continue
        by_subject.setdefault(rel.parts[0], []).append(clip_dir)

    subjects = sorted(by_subject, key=lambda subject: (-len(by_subject[subject]), subject))
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    selected: list[Path] = []
    for subject in subjects:
        subject_clips = sorted(by_subject[subject])
        if max_clips_per_subject is not None:
            subject_clips = subject_clips[:max_clips_per_subject]
        selected.extend(subject_clips)
    return sorted(selected)


def build_model(device: torch.device) -> torch.nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval().to(device)
    return model


def build_transform() -> transforms.Compose:
    weights = models.ResNet18_Weights.DEFAULT
    return weights.transforms()


def extract_clip_features(
    clip_dir: Path,
    model: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    max_frames: int | None,
) -> torch.Tensor:
    frames = sorted(path for path in clip_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_frames is not None:
        frames = frames[:max_frames]
    feats: list[torch.Tensor] = []
    batch: list[torch.Tensor] = []
    with torch.no_grad():
        for frame in frames:
            image = Image.open(frame).convert("RGB")
            batch.append(transform(image))
            if len(batch) == batch_size:
                tensor = torch.stack(batch).to(device)
                feat = model(tensor).flatten(1).cpu()
                feats.append(feat)
                batch = []
        if batch:
            tensor = torch.stack(batch).to(device)
            feat = model(tensor).flatten(1).cpu()
            feats.append(feat)
    if not feats:
        raise RuntimeError(f"no frames found in {clip_dir}")
    return torch.cat(feats, dim=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ResNet-18 pool5 features for YouTube Faces clips.")
    parser.add_argument(
        "--data-dir",
        default="auto",
        help="Raw aligned image root. 'auto' checks data/ytfaces/aligned_images_DB then data/aligned_images_DB.",
    )
    parser.add_argument("--out-dir", default="data/ytfaces/features", help="Output feature root.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame cap per clip for debugging.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Extract only the subjects with the most clips.")
    parser.add_argument("--max-clips-per-subject", type=int, default=None, help="Optional clip cap per subject.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features.pt files.")
    parser.add_argument("--limit-clips", type=int, default=None, help="Optional clip cap for smoke extraction.")
    return parser.parse_args()


def normalize_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist. Download YouTube Faces aligned_images_DB first.")

    device = normalize_device(args.device)
    model = build_model(device)
    transform = build_transform()
    clip_dirs = find_clip_dirs(data_dir)
    clip_dirs = select_clip_dirs(
        clip_dirs,
        data_dir=data_dir,
        max_subjects=args.max_subjects,
        max_clips_per_subject=args.max_clips_per_subject,
    )
    if args.limit_clips is not None:
        clip_dirs = clip_dirs[: args.limit_clips]

    rows = []
    for clip_dir in tqdm(clip_dirs, desc="extracting clips"):
        rel = clip_dir.relative_to(data_dir)
        clip_out = ensure_dir(out_dir / rel)
        out_path = clip_out / "features.pt"
        if out_path.exists() and not args.overwrite:
            rows.append({"clip_id": str(rel), "status": "skipped", "features_path": str(out_path)})
            continue
        try:
            features = extract_clip_features(
                clip_dir,
                model=model,
                transform=transform,
                device=device,
                batch_size=args.batch_size,
                max_frames=args.max_frames,
            )
            torch.save(features, out_path)
            rows.append({"clip_id": str(rel), "status": "ok", "n_frames": int(features.shape[0]), "features_path": str(out_path)})
        except Exception as exc:
            rows.append({"clip_id": str(rel), "status": "failed", "error": repr(exc)})

    save_json(
        out_dir / "extraction_summary.json",
        {
            "data_dir": str(data_dir),
            "out_dir": str(out_dir),
            "device": str(device),
            "n_clips": len(clip_dirs),
            "max_subjects": args.max_subjects,
            "max_clips_per_subject": args.max_clips_per_subject,
            "max_frames": args.max_frames,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
