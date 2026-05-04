import argparse
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODELS = [
    "cropvlm",
    "openai_clip_vit_b32",
    "bioclip",
    "bioclip2",
    "biotrove_clip",
    "remoteclip",
    "siglip2",
]


class ImageFolderPaths(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.classes:
            for path in sorted((self.root / class_name).iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    self.samples.append((path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        return Image.open(path).convert("RGB"), label, str(path)


def pil_collate(batch):
    images, labels, paths = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long), list(paths)


def display_name(class_name: str) -> str:
    return class_name.replace("_", " ")


def normalize(features: torch.Tensor) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[0]
    return F.normalize(features.float(), dim=-1)


class Adapter:
    name = ""
    family = ""
    checkpoint: Optional[str] = None
    load_message: Optional[str] = None

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        raise NotImplementedError

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        raise NotImplementedError


class OpenAIClipAdapter(Adapter):
    def __init__(self, device: torch.device, checkpoint: Optional[str] = None):
        import clip

        self.name = "CropVLM" if checkpoint else "OpenAI CLIP ViT-B/32"
        self.family = "openai_clip"
        self.device = device
        self.clip = clip
        self.model, self.preprocess = clip.load("ViT-B/32", device=str(device))
        if checkpoint:
            checkpoint_path = Path(checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"CropVLM checkpoint not found: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=device)
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
            self.model.load_state_dict(state)
            self.checkpoint = str(checkpoint_path)
        self.model.eval()

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.clip.tokenize(list(prompts), truncate=True).to(self.device)
        with torch.no_grad():
            return normalize(self.model.encode_text(tokens))

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            return normalize(self.model.encode_image(batch))


class OpenClipAdapter(Adapter):
    def __init__(
        self,
        model_name: str,
        pretrained: Optional[str],
        device: torch.device,
        hf_checkpoint: Optional[Tuple[str, str]] = None,
    ):
        import open_clip

        self.name = model_name
        self.family = "open_clip"
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.open_clip = open_clip

        if hf_checkpoint:
            from huggingface_hub import hf_hub_download

            repo, filename = hf_checkpoint
            checkpoint = hf_hub_download(repo, filename)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=None)
            ckpt = torch.load(checkpoint, map_location="cpu")
            state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            if any(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            self.load_message = str(self.model.load_state_dict(state, strict=False))
            self.checkpoint = checkpoint
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )

        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(device).eval()

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(list(prompts)).to(self.device)
        with torch.no_grad():
            return normalize(self.model.encode_text(tokens))

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(image) for image in images]).to(self.device)
        with torch.no_grad():
            return normalize(self.model.encode_image(batch))


class Siglip2Adapter(Adapter):
    def __init__(self, device: torch.device):
        from transformers import AutoModel, AutoProcessor

        self.name = "google/siglip2-base-patch16-224"
        self.family = "transformers_siglip2"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name).to(device).eval()

    def encode_text(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(text=list(prompts), padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                features = self.model.get_text_features(**inputs)
            else:
                features = self.model(**inputs).text_embeds
            return normalize(features)

    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt").to(self.device)
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
            else:
                features = self.model(**inputs).image_embeds
            return normalize(features)


def build_adapter(model_key: str, device: torch.device, cropvlm_checkpoint: str) -> Adapter:
    if model_key == "cropvlm":
        return OpenAIClipAdapter(device, checkpoint=cropvlm_checkpoint)
    if model_key == "openai_clip_vit_b32":
        return OpenAIClipAdapter(device)
    if model_key == "bioclip":
        return OpenClipAdapter("hf-hub:imageomics/bioclip", None, device)
    if model_key == "bioclip2":
        return OpenClipAdapter("hf-hub:imageomics/bioclip-2", None, device)
    if model_key == "biotrove_clip":
        return OpenClipAdapter(
            "ViT-B-16",
            None,
            device,
            hf_checkpoint=("BGLab/BioTrove-CLIP", "biotroveclip-vit-b-16-from-bioclip-epoch-8.pt"),
        )
    if model_key == "remoteclip":
        return OpenClipAdapter(
            "ViT-B-32",
            None,
            device,
            hf_checkpoint=("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt"),
        )
    if model_key == "siglip2":
        return Siglip2Adapter(device)
    raise KeyError(
        f"Unknown model '{model_key}'. Supported models: {', '.join(DEFAULT_MODELS)}. "
        "TULIP, EVA-CLIP, and LongCLIP are intentionally excluded."
    )


def per_class_stats(per_class: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values = [item["accuracy"] for item in per_class.values() if item.get("accuracy") is not None]
    if not values:
        return {
            "per_class_accuracy_mean": None,
            "per_class_accuracy_std": None,
            "per_class_accuracy_std_population": None,
            "num_classes_with_accuracy": 0,
        }
    mean = sum(values) / len(values)
    sample_std = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    population_std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return {
        "per_class_accuracy_mean": mean,
        "per_class_accuracy_std": sample_std,
        "per_class_accuracy_std_population": population_std,
        "num_classes_with_accuracy": len(values),
    }


def evaluate_model(args: argparse.Namespace, dataset: ImageFolderPaths, model_key: str) -> Dict[str, Any]:
    started_at = time.time()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    prompts = [args.prompt_template.format(display_name(class_name)) for class_name in dataset.classes]
    result: Dict[str, Any] = {
        "model_key": model_key,
        "dataset": str(dataset.root),
        "num_images": len(dataset),
        "num_classes": len(dataset.classes),
        "classes": dataset.classes,
        "class_prompts": dict(zip(dataset.classes, prompts)),
        "prompt_template": args.prompt_template,
        "device": str(device),
        "status": "started",
        "started_at_unix": started_at,
    }

    try:
        adapter = build_adapter(model_key, device, args.cropvlm_checkpoint)
        result["model_name"] = adapter.name
        result["family"] = adapter.family
        result["checkpoint"] = adapter.checkpoint
        result["load_message"] = adapter.load_message
        text_features = adapter.encode_text(prompts).to(device)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pil_collate,
        )

        class_total = [0 for _ in dataset.classes]
        class_correct = [0 for _ in dataset.classes]
        confusion = [[0 for _ in dataset.classes] for _ in dataset.classes]
        predictions: List[Dict[str, Any]] = []
        correct = 0

        for images, labels, paths in tqdm(loader, desc=model_key):
            image_features = adapter.encode_images(images)
            logits = image_features @ text_features.T
            pred = logits.argmax(dim=-1).detach().cpu()
            scores = logits.max(dim=-1).values.detach().cpu()
            for true_idx, pred_idx, score, path in zip(labels.tolist(), pred.tolist(), scores.tolist(), paths):
                class_total[true_idx] += 1
                class_correct[true_idx] += int(true_idx == pred_idx)
                confusion[true_idx][pred_idx] += 1
                correct += int(true_idx == pred_idx)
                if args.save_predictions:
                    predictions.append(
                        {
                            "path": path,
                            "true_class": dataset.classes[true_idx],
                            "pred_class": dataset.classes[pred_idx],
                            "correct": true_idx == pred_idx,
                            "score": float(score),
                        }
                    )

        per_class = {}
        for idx, class_name in enumerate(dataset.classes):
            total = class_total[idx]
            per_class[class_name] = {
                "correct": class_correct[idx],
                "total": total,
                "accuracy": class_correct[idx] / total if total else None,
            }

        result.update(
            {
                "status": "ok",
                "accuracy": correct / len(dataset) if len(dataset) else None,
                "correct": correct,
                "per_class": per_class,
                "confusion_matrix": confusion,
                "predictions": predictions if args.save_predictions else None,
            }
        )
        result.update(per_class_stats(per_class))
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    result["elapsed_seconds"] = time.time() - started_at
    return result


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="ImageFolder-style dataset root.")
    parser.add_argument("--output", default="outputs/zero_shot_results.json")
    parser.add_argument("--cropvlm-checkpoint", default="models/CropVLM.pth")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prompt-template", default="{}")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    excluded = {"tulip", "eva_clip", "eva_clip_official", "longclip"}
    requested = [model for model in args.models if model not in excluded]
    skipped = [model for model in args.models if model in excluded]

    dataset = ImageFolderPaths(args.dataset)
    results = [evaluate_model(args, dataset, model_key) for model_key in requested]
    ok = [result for result in results if result.get("status") == "ok"]
    failed = [result for result in results if result.get("status") != "ok"]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset.root),
        "num_images": len(dataset),
        "num_classes": len(dataset.classes),
        "classes": dataset.classes,
        "requested_models": args.models,
        "evaluated_models": requested,
        "skipped_models": skipped,
        "num_models": len(results),
        "num_successful": len(ok),
        "num_failed": len(failed),
        "models": {
            result["model_key"]: {
                "status": result.get("status"),
                "accuracy": result.get("accuracy"),
                "correct": result.get("correct"),
                "num_images": result.get("num_images"),
                "per_class_accuracy_mean": result.get("per_class_accuracy_mean"),
                "per_class_accuracy_std": result.get("per_class_accuracy_std"),
                "per_class_accuracy_std_population": result.get("per_class_accuracy_std_population"),
                "num_classes_with_accuracy": result.get("num_classes_with_accuracy"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "error": result.get("error"),
            }
            for result in results
        },
        "model_results": {result["model_key"]: result for result in results},
        "results": results,
    }
    write_json(Path(args.output), summary)
    print(Path(args.output))


if __name__ == "__main__":
    main()
