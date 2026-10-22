"""Task masks select observed prefixes and causal, interleaved missing-modality targets."""
from dataclasses import dataclass

import torch

from . import MODALITIES, MODES


@dataclass(frozen=True)
class Vocabulary:
    text_size: int
    codebook_size: int
    geometry_tokens: int = 8
    pad: int = 0
    geometry: int = 1
    begin: int = 2
    target: int = 3
    end: int = 4
    missing: int = 5
    modality_start: int = 6
    mode_start: int = 9
    code_start: int = 16

    @property
    def size(self):
        return self.code_start + 3 * self.codebook_size

    def offset(self, modality):
        return self.code_start + MODALITIES.index(modality) * self.codebook_size

    def token(self, local):
        return self.text_size + local


class SequenceBuilder:
    def __init__(self, tokenizer, vocab, max_text_tokens=128, max_sequence_tokens=1024):
        self.tokenizer, self.vocab = tokenizer, vocab
        self.max_text_tokens, self.max_sequence_tokens = max_text_tokens, max_sequence_tokens

    def prefix(self, codes, mode, text, frames, fps, gender, dataset):
        if mode not in MODES:
            raise ValueError("Expected seven whole-modality H/O/C masks; 1=generate, 0=observed")
        v = self.vocab
        header = f"Dataset: {dataset}; frames: {frames}; fps: {fps:g}; gender: {gender}. Text: {text}"
        words = self.tokenizer.encode(header, add_special_tokens=False, truncation=True,
                                      max_length=self.max_text_tokens)
        if any(i < 0 or i >= v.text_size for i in words):
            raise ValueError("Tokenizer and language backbone vocabulary disagree")
        result = [v.token(v.begin)] + [v.token(v.geometry)] * v.geometry_tokens + words
        result.append(v.token(v.mode_start + MODES.index(mode)))
        for i, key in enumerate(MODALITIES):
            result.append(v.token(v.modality_start + i))
            if mode[i] == "0":
                ids = codes[key]
                if any(x < 0 or x >= v.codebook_size for x in ids):
                    raise ValueError("Motion code outside its codebook")
                result.extend(v.token(v.offset(key) + x) for x in ids)
            else:
                result.append(v.token(v.missing))
        result.append(v.token(v.target))
        return result

    @staticmethod
    def order(mode, token_count):
        return [key for _ in range(token_count) for key, bit in zip(MODALITIES, mode) if bit == "1"]

    def example(self, codes, mode, text, frames, fps, gender, dataset, training=True, token_count=None):
        source = self.prefix(codes, mode, text, frames, fps, gender, dataset)
        if token_count is None:
            token_count = len(next(iter(codes.values())))
        order = self.order(mode, token_count)
        if len(source) + len(order) + 1 > self.max_sequence_tokens:
            raise ValueError("Sequence exceeds configured context; reduce window/text length or increase context")
        if not training:
            return {"input_ids": source, "labels": [-100] * len(source)}, order
        if any(len(codes[k]) != token_count for k in MODALITIES):
            raise ValueError("All modality tokenizers must use the same temporal stride")
        cursor = {k: 0 for k in MODALITIES}
        target = []
        for key in order:
            target.append(self.vocab.offset(key) + codes[key][cursor[key]])
            cursor[key] += 1
        target.append(self.vocab.end)
        return {"input_ids": source + [self.vocab.token(i) for i in target],
                "labels": [-100] * len(source) + target}, order

    def collate(self, examples, device="cpu"):
        size = max(len(e["input_ids"]) for e in examples)
        ids = torch.full((len(examples), size), self.vocab.token(self.vocab.pad), dtype=torch.long, device=device)
        labels = torch.full_like(ids, -100)
        attention = torch.zeros_like(ids)
        for i, example in enumerate(examples):
            n = len(example["input_ids"])
            ids[i, :n] = torch.tensor(example["input_ids"], device=device)
            labels[i, :n] = torch.tensor(example["labels"], device=device)
            attention[i, :n] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attention}


def build_training_batch(batch, codecs, builder, modes=None, text_dropout=0.):
    encoded = codecs.encode(batch)
    examples = []
    for i in range(len(batch["human"])):
        mode = modes[i] if modes is not None else MODES[int(torch.randint(len(MODES), ()))]
        text = batch["text"][i] if torch.rand(()) >= text_dropout else ""
        codes = {k: encoded[k][i].tolist() for k in MODALITIES}
        example, _ = builder.example(codes, mode, text, int(batch["valid_frames"][i].sum()),
                                     float(batch["fps"][i]), "male" if batch["gender"][i] else "female",
                                     codecs.config.dataset)
        examples.append(example)
    return builder.collate(examples, batch["human"].device)
