"""Tiny randomly initialized byte-level causal Transformer (PyTorch optional)."""
import hashlib
from pathlib import Path


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("Install PyTorch first: python3 -m pip install -e '.[train]'") from error
    return torch, nn


def build_model(config):
    torch, nn = require_torch()
    dim = int(config["embedding_dim"])
    block = int(config["block_size"])
    heads = int(config["heads"])
    layers = int(config["layers"])
    if dim % heads:
        raise ValueError("embedding_dim must be divisible by heads")

    class BabyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(256, dim)
            self.pos = nn.Embedding(block, dim)
            layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=float(config.get("dropout", 0.0)), batch_first=True,
                norm_first=True, activation="gelu",
            )
            self.layers = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)
            self.out = nn.Linear(dim, 256)

        def forward(self, tokens):
            _, length = tokens.shape
            if length > block:
                raise ValueError("Input exceeds block_size")
            positions = torch.arange(length, device=tokens.device)
            hidden = self.token(tokens) + self.pos(positions)
            causal = torch.ones((length, length), device=tokens.device, dtype=torch.bool).triu(1)
            hidden = self.layers(hidden, mask=causal)
            return self.out(self.norm(hidden))
    return BabyGPT()


def get_device(choice):
    torch, _ = require_torch()
    if choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if choice not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be auto, cpu, cuda, or mps")
    return choice


def load_checkpoint(path, map_location="cpu"):
    torch, _ = require_torch()
    # weights_only=True ensures no untrusted Python pickle classes are loaded.
    content = torch.load(path, map_location=map_location, weights_only=True)
    if content.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = build_model(content["model_config"])
    model.load_state_dict(content["weights"])
    return model, content


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_text(model, prompt, block_size, max_new_bytes=100, temperature=0.9, device="cpu", seed=42, greedy=False):
    torch, _ = require_torch()
    torch.manual_seed(seed)
    data = list(prompt.encode("utf-8", errors="replace")) or [10]
    model.to(device).eval()
    with torch.no_grad():
        for _ in range(max_new_bytes):
            x = torch.tensor([data[-block_size:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1, :] / max(temperature, 0.05)
            token = int(logits.argmax().item() if greedy else torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            data.append(token)
    return bytes(data).decode("utf-8", errors="replace")
