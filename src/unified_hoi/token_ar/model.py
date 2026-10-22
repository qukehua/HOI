"""Pretrained causal language backbone + geometry prefix + discrete HOI vocabulary."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .sequence import Vocabulary


class TokenHOI(nn.Module):
    def __init__(self, backbone, vocab):
        super().__init__()
        self.backbone, self.vocab = backbone, vocab
        hidden = backbone.config.hidden_size
        if 1024 % vocab.geometry_tokens:
            raise ValueError("geometry_tokens must divide the 1024 BPS anchors")
        self.motion_embeddings = nn.Embedding(vocab.size, hidden)
        self.motion_head = nn.Linear(hidden, vocab.size, bias=False)
        self.geometry = nn.Sequential(nn.Linear(3072 // vocab.geometry_tokens, 512), nn.GELU(),
                                      nn.Linear(512, hidden))
        self.geometry_positions = nn.Parameter(torch.randn(vocab.geometry_tokens, hidden) * .02)
        self.gradient_checkpointing = False

    def configure_training(self, scope="full", last_layers=4, gradient_checkpointing=True):
        if scope not in ("full", "last_layers", "frozen"):
            raise ValueError("train_scope must be full, last_layers or frozen; no task-specific adapters")
        self.backbone.requires_grad_(scope == "full")
        if scope == "last_layers":
            layers = getattr(self.backbone, "layers", None)
            if layers is None or not 1 <= last_layers <= len(layers):
                raise ValueError("Backbone has no supported layers or last_layers is invalid")
            for layer in layers[-last_layers:]:
                layer.requires_grad_(True)
            if getattr(self.backbone, "norm", None) is not None:
                self.backbone.norm.requires_grad_(True)
        self.backbone.config.use_cache = False
        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def embeddings(self, input_ids, object_bps=None):
        is_motion = input_ids >= self.vocab.text_size
        words = self.backbone.get_input_embeddings()(input_ids.masked_fill(is_motion, 0))
        ids = (input_ids - self.vocab.text_size).masked_fill(~is_motion, self.vocab.pad)
        motions = self.motion_embeddings(ids).to(words.dtype)
        result = torch.where(is_motion[..., None], motions, words)
        if object_bps is not None:
            geometry_mask = input_ids == self.vocab.token(self.vocab.geometry)
            if not (geometry_mask.sum(-1) == self.vocab.geometry_tokens).all():
                raise ValueError("Geometry prefix slots missing or duplicated")
            points = object_bps.reshape(len(input_ids), self.vocab.geometry_tokens, -1)
            geometry = self.geometry(points.to(self.geometry[0].weight.dtype)) + self.geometry_positions
            result = result.clone()
            result[geometry_mask] = geometry.to(result.dtype).reshape(-1, result.shape[-1])
        return result

    def forward(self, input_ids, attention_mask, object_bps, labels=None):
        output = self.backbone(inputs_embeds=self.embeddings(input_ids, object_bps),
                               attention_mask=attention_mask, use_cache=False, return_dict=True)
        hidden = output.last_hidden_state
        if labels is None:
            return {"logits": self.motion_head(hidden.to(self.motion_head.weight.dtype))}
        valid = labels[:, 1:] != -100
        if not valid.any():
            raise ValueError("No generated-token labels")
        # Avoid allocating full language-vocabulary logits (150k+) for motion-only targets.
        logits = self.motion_head(hidden[:, :-1][valid].to(self.motion_head.weight.dtype)).float()
        loss_sum = F.cross_entropy(logits, labels[:, 1:][valid], reduction="sum")
        return {"loss": loss_sum / valid.sum(), "loss_sum": loss_sum, "token_count": valid.sum()}

    @torch.no_grad()
    def generate_codes(self, prefix, order, object_bps, seed=42, temperature=1., top_k=50):
        if not math.isfinite(temperature) or temperature < 0 or top_k < 0:
            raise ValueError("Invalid temperature/top_k")
        self.eval()
        device = self.motion_embeddings.weight.device
        ids = torch.tensor([prefix], dtype=torch.long, device=device)
        attention = torch.ones_like(ids)
        generator = torch.Generator(device=device).manual_seed(seed)
        codes = {k: [] for k in ("human", "object", "contact")}
        past = None
        for key in order:
            embedding = self.embeddings(ids, object_bps if past is None else None)
            result = self.backbone(inputs_embeds=embedding, attention_mask=attention,
                                   past_key_values=past, use_cache=True, return_dict=True)
            logits = self.motion_head(result.last_hidden_state[:, -1].to(self.motion_head.weight.dtype))[0].float()
            start = self.vocab.offset(key)
            logits = logits[start:start + self.vocab.codebook_size]
            if temperature == 0:
                index = int(logits.argmax())
            else:
                logits = logits / temperature
                if top_k:
                    threshold = logits.topk(min(top_k, len(logits))).values[-1]
                    logits = logits.masked_fill(logits < threshold, -torch.inf)
                index = int(torch.multinomial(logits.softmax(-1), 1, generator=generator))
            codes[key].append(index)
            ids = torch.tensor([[self.vocab.token(start + index)]], device=device)
            attention = torch.cat((attention, torch.ones_like(ids)), -1)
            past = result.past_key_values
        return codes


def build_model(config, codec_config, backbone_config=None, pretrained=True):
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    if backbone_config is not None:
        kind = backbone_config["model_type"]
        hf_config = AutoConfig.for_model(kind, **{k: v for k, v in backbone_config.items() if k != "model_type"})
        tokenizer = None
    else:
        name = config["backbone"]
        revision = config.get("backbone_revision")
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        hf_config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=False)

    # Official Qwen3.5 releases contain a vision encoder and an auxiliary MTP module.
    # HOI uses its pretrained text backbone with our own geometry prefix and output head.
    composite = hf_config.model_type == "qwen3_5"
    if composite:
        hf_config = hf_config.text_config
    dtype = getattr(torch, config.get("load_dtype", "float32"))
    attention = config.get("attention_backend", "sdpa")
    if backbone_config is None and pretrained:
        kwargs = {}
        if composite:
            kwargs["key_mapping"] = {r"^model\.language_model\.": "", r"^language_model\.": ""}
        backbone, loading = AutoModel.from_pretrained(
            name, config=hf_config, revision=revision, dtype=dtype, attn_implementation=attention,
            trust_remote_code=False, output_loading_info=True, **kwargs)
        # A wrong multimodal-to-text mapping must never silently train random weights.
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError(f"Incomplete pretrained backbone: {loading}")
    else:
        backbone = AutoModel.from_config(hf_config, attn_implementation=attention).to(dtype)
    vocab = Vocabulary(backbone.get_input_embeddings().num_embeddings, codec_config.codebook_size,
                       config.get("geometry_tokens", 8))
    if config.get("max_sequence_tokens", 1024) > backbone.config.max_position_embeddings:
        raise ValueError("Requested context exceeds backbone context")
    model = TokenHOI(backbone, vocab)
    return model, tokenizer
