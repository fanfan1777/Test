import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
_IMAGE_DIR_CANDIDATES: Sequence[str] = ("images", "trainval-image", "train-image", "test-image")
_MASK_DIR_CANDIDATES: Sequence[str] = ("masks", "trainval-mask", "train-mask", "test-mask")


def _normalize_root(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return root


def _possible_filenames(name: str):
    name = name.strip()
    if not name:
        return
    suffix = Path(name).suffix
    if suffix:
        yield name
    else:
        for ext in _IMAGE_EXTS:
            yield f"{name}{ext}"


def _resolve_file(root: Path, folders: Sequence[str], name: str) -> str:
    for folder in folders:
        folder_path = root / folder
        if not folder_path.is_dir():
            continue
        for candidate in _possible_filenames(name):
            candidate_path = folder_path / candidate
            if candidate_path.exists():
                return str(candidate_path)
    raise FileNotFoundError(f"Could not locate '{name}' inside {root}")


def _load_from_filelist(root: Path, list_path: Path) -> Tuple[List[str], List[str]]:
    lines = [line.strip() for line in list_path.read_text().splitlines() if line.strip()]
    images = [_resolve_file(root, _IMAGE_DIR_CANDIDATES, name) for name in lines]
    masks = [_resolve_file(root, _MASK_DIR_CANDIDATES, name) for name in lines]
    return images, masks


def _iter_valid_files(folder: Path):
    for entry in folder.iterdir():
        if entry.is_file() and entry.suffix.lower() in _IMAGE_EXTS:
            yield entry


def _collect_pairs(image_dir: Path, mask_dir: Path) -> Tuple[List[str], List[str], List[str]]:
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing image/mask directories: {image_dir}, {mask_dir}")
    image_map = {p.stem: p for p in _iter_valid_files(image_dir)}
    mask_map = {p.stem: p for p in _iter_valid_files(mask_dir)}
    common_keys = sorted(set(image_map).intersection(mask_map))
    if not common_keys:
        raise RuntimeError(f"No overlapping filenames between {image_dir} and {mask_dir}")
    images = [str(image_map[key]) for key in common_keys]
    masks = [str(mask_map[key]) for key in common_keys]
    return images, masks, common_keys


def _auto_split_trainval(root: Path, split: str, val_ratio: float, split_seed: int) -> Tuple[List[str], List[str]]:
    image_dir = root / "trainval-image"
    mask_dir = root / "trainval-mask"
    images, masks, keys = _collect_pairs(image_dir, mask_dir)
    rng = random.Random(split_seed)
    rng.shuffle(keys)

    if len(keys) == 1:
        split_point = 1 if split == "val" else 0
    else:
        split_point = max(1, int(round(len(keys) * val_ratio)))
        split_point = min(split_point, len(keys) - 1)

    val_subset = keys[:split_point]
    val_keys = set(val_subset)
    if split == "val":
        selected_keys = val_subset
    else:
        selected_keys = [key for key in keys if key not in val_keys]

    index_lookup = {key: idx for idx, key in enumerate(keys)}
    selected_indices = [index_lookup[key] for key in selected_keys]
    return [images[i] for i in selected_indices], [masks[i] for i in selected_indices]


def _load_test_split(root: Path) -> Tuple[List[str], List[str]]:
    image_dir = root / "test-image"
    mask_dir = root / "test-mask"
    images, masks, _ = _collect_pairs(image_dir, mask_dir)
    return images, masks


def load_split(path: str,
               split: str = "train",
               val_name: Optional[str] = None,
               val_ratio: float = 0.1,
               split_seed: int = 42) -> Tuple[List[str], List[str]]:
    """Return ordered image/mask paths for a specific split.

    Falls back to TN3K-style directory layouts when explicit txt manifests are absent.
    """
    split = split.lower()
    root = _normalize_root(path)

    # Preferred path: explicit file list
    manifest_name = f"{split}.txt"
    if split == "val" and val_name:
        manifest_name = f"val_{val_name}.txt"
    manifest_path = root / manifest_name
    if manifest_path.is_file():
        return _load_from_filelist(root, manifest_path)

    # TN3K-style directories (trainval/test)
    if split in ("train", "val") and (root / "trainval-image").is_dir():
        return _auto_split_trainval(root, split, val_ratio, split_seed)
    if split == "test" and (root / "test-image").is_dir():
        return _load_test_split(root)

    raise FileNotFoundError(
        f"Unable to resolve split '{split}' in {root}. Missing manifest '{manifest_name}' or expected directories.")


def load_data(path: str,
              val_name: Optional[str] = None,
              val_ratio: float = 0.1,
              split_seed: int = 42) -> Tuple[Tuple[List[str], List[str]], Tuple[List[str], List[str]]]:
    """Backward-compatible helper that returns (train, val) splits."""
    train = load_split(path, split="train", val_name=val_name, val_ratio=val_ratio, split_seed=split_seed)
    val = load_split(path, split="val", val_name=val_name, val_ratio=val_ratio, split_seed=split_seed)
    return train, val
