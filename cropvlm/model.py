from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


CROP_CLASSES = [
    "apple",
    "avocado",
    "banana",
    "barley",
    "bell pepper",
    "broccoli",
    "cacao",
    "canola",
    "cauliflower",
    "cherry",
    "chilli",
    "coconut",
    "coffee",
    "corn",
    "cotton",
    "cucumber",
    "eggplant",
    "kiwi",
    "lemon",
    "mango",
    "olive",
    "orange",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "pumpkin",
    "rice",
    "soyabean",
    "strawberry",
    "sugarcane",
    "sunflower",
    "tea",
    "tomato",
    "watermelon",
    "wheat",
]


def _normalize(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=-1)


class CropVLMClassifier:
    """Small zero-shot wrapper around the CropVLM/OpenAI CLIP ViT-B/32 model."""

    def __init__(
        self,
        checkpoint: str,
        class_names: Sequence[str] = CROP_CLASSES,
        device: str | None = None,
        prompt_template: str = "{}",
    ):
        import clip

        self.clip = clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.prompt_template = prompt_template
        self.class_names = list(class_names)

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CropVLM checkpoint not found: {checkpoint_path}")

        self.model, self.preprocess = clip.load(
            "ViT-B/32",
            device=str(self.device),
            download_root=str(Path.home() / ".cache" / "clip"),
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state)
        self.model.eval()
        self.set_classes(self.class_names)

    def set_classes(self, class_names: Sequence[str]) -> None:
        self.class_names = [c.strip() for c in class_names if c.strip()]
        prompts = [self.prompt_template.format(c) for c in self.class_names]
        tokens = self.clip.tokenize(prompts, truncate=True).to(self.device)
        with torch.no_grad():
            self.text_features = _normalize(self.model.encode_text(tokens))

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return _normalize(self.model.encode_image(batch))

    def predict(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        return [(label, probability) for label, probability, _ in self.predict_with_scores(image, top_k=top_k)]

    def predict_scores(self, image: Image.Image) -> Dict[str, float]:
        image_features = self.encode_image(image)
        logits = (image_features @ self.text_features.T).squeeze(0)
        return {name: float(score) for name, score in zip(self.class_names, logits.tolist())}

    def predict_with_scores(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float, float]]:
        image_features = self.encode_image(image)
        cosine_scores = (image_features @ self.text_features.T).squeeze(0)
        logit_scale = self.model.logit_scale.exp().clamp(max=100)
        probabilities = (logit_scale * cosine_scores).softmax(dim=-1)
        k = min(top_k, len(self.class_names))
        probs, indices = probabilities.topk(k)
        return [
            (self.class_names[idx], float(prob), float(cosine_scores[idx]))
            for prob, idx in zip(probs.tolist(), indices.tolist())
        ]


def load_cropvlm(
    checkpoint: str,
    class_names: Sequence[str] = CROP_CLASSES,
    device: str | None = None,
    prompt_template: str = "{}",
) -> CropVLMClassifier:
    return CropVLMClassifier(
        checkpoint=checkpoint,
        class_names=class_names,
        device=device,
        prompt_template=prompt_template,
    )


def parse_class_names(text: str | Iterable[str]) -> List[str]:
    if isinstance(text, str):
        raw = text.replace(",", "\n").splitlines()
    else:
        raw = list(text)
    return [name.strip() for name in raw if name.strip()]
