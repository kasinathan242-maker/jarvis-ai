"""JARVIS Lite — Custom Semantic Embeddings Training Pipeline.

Trains a custom semantic vector encoder using Triplet Cosine Margin Loss,
and exports an optimized ONNX model for zero-daemon ChromaDB vector embeddings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training.train_intent import SimpleTokenizer


class TripletDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        triplets: list[dict[str, str]],
        tokenizer: SimpleTokenizer,
        max_len: int = 48,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.anchors = [
            torch.tensor(tokenizer.encode(t["anchor"], max_len=max_len), dtype=torch.long)
            for t in triplets
        ]
        self.positives = [
            torch.tensor(tokenizer.encode(t["positive"], max_len=max_len), dtype=torch.long)
            for t in triplets
        ]
        self.negatives = [
            torch.tensor(tokenizer.encode(t["negative"], max_len=max_len), dtype=torch.long)
            for t in triplets
        ]

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.anchors[idx], self.positives[idx], self.negatives[idx]


class SemanticEmbeddingModel(nn.Module):
    """Dense Semantic Vector Encoder producing L2-normalized 768-dimensional embeddings."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, out_dim: int = 768) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dense = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Tanh(),
            nn.Linear(embed_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len]
        embeds = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        summed = (embeds * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / lengths
        projected = self.dense(pooled)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized


def train_embedding_model(
    dataset_path: Path | str,
    output_dir: Path | str,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    margin: float = 0.5,
) -> dict[str, Any]:
    """Trains the embedding model and exports PyTorch and ONNX models."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)

    triplets = data["triplets"]

    # Build vocabulary from all texts
    all_texts: list[str] = []
    for t in triplets:
        all_texts.extend([t["anchor"], t["positive"], t["negative"]])

    tokenizer = SimpleTokenizer(max_vocab=3000)
    tokenizer.fit(all_texts)

    dataset = TripletDataset(triplets, tokenizer, max_len=48)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SemanticEmbeddingModel(vocab_size=len(tokenizer.vocab) + 10, embed_dim=128, out_dim=768)
    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for _epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # Save artifacts
    torch.save(model.state_dict(), output_dir / "model.pt")

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer.to_dict(), f, indent=2)

    config = {
        "embedding_dim": 768,
        "vocab_size": len(tokenizer.vocab) + 10,
        "max_seq_len": 48,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Export to ONNX
    model.eval()
    dummy_input = torch.zeros((1, 48), dtype=torch.long)
    onnx_path = output_dir / "model.onnx"

    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            str(onnx_path),
            input_names=["input_tokens"],
            output_names=["embeddings"],
            dynamic_axes={"input_tokens": {0: "batch_size"}, "embeddings": {0: "batch_size"}},
            opset_version=17,
            dynamo=False,
        )
    except Exception as exc:
        print(f"Warning: ONNX export skipped ({exc}), PyTorch weights model.pt are active.")

    return {
        "status": "SUCCESS",
        "output_dir": str(output_dir),
        "onnx_path": str(onnx_path),
        "embedding_dim": 128,
    }


if __name__ == "__main__":
    from training.dataset_generator import generate_embedding_pairs

    base_dir = Path(__file__).resolve().parent
    data_path = base_dir / "data" / "embedding_pairs.json"
    generate_embedding_pairs(data_path)

    res = train_embedding_model(data_path, base_dir.parent / "models" / "embeddings")
    print(f"Embedding model training complete: {res}")
