"""
Dataset for Qwen2-VL + Gated Cross-Attention training.

Key difference vs train_lora_qwenvl.py's QwenVLVQADataset: the chat template
built here has NO image content block, so apply_chat_template never inserts
<|vision_start|><|image_pad|>...<|vision_end|> placeholder tokens into
input_ids. Image pixels are processed completely separately (via the
processor's image_processor only) and consumed exclusively by
Qwen2VLXAttnForCausalLM's cross-attention path — never by the main
self-attention sequence.
"""

import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


class QwenVLXAttnDataset(Dataset):
    def __init__(self, data_path, image_root, processor, max_length=512):
        with open(data_path, encoding="utf-8") as f:
            all_samples = [json.loads(l) for l in f if l.strip()]

        self.image_root = image_root
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

        valid = []
        skipped_img = 0
        for s in all_samples:
            if not os.path.exists(os.path.join(image_root, s["image"])):
                skipped_img += 1
                continue
            valid.append(s)
        self.samples = valid
        if skipped_img:
            print(f"[Dataset] Skipped {skipped_img} samples with missing images")
        print(f"[Dataset] {len(self.samples)} samples ready ({data_path})")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _messages(question, answer=None):
        # No image content block on purpose — see module docstring.
        msgs = [{"role": "user", "content": question}]
        if answer is not None:
            msgs.append({"role": "assistant", "content": answer})
        return msgs

    def __getitem__(self, idx):
        sample = self.samples[idx]
        question = sample["question"]
        answer = sample["answer"]

        text_full = self.processor.apply_chat_template(
            self._messages(question, answer), tokenize=False, add_generation_prompt=False
        )
        text_prompt = self.processor.apply_chat_template(
            self._messages(question), tokenize=False, add_generation_prompt=True
        )

        full_ids = self.tokenizer(
            text_full, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).input_ids[0]
        prompt_ids = self.tokenizer(
            text_prompt, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).input_ids[0]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = full_ids.clone()
        labels[:prompt_len] = IGNORE_INDEX

        image = Image.open(os.path.join(self.image_root, sample["image"])).convert("RGB")
        image_enc = self.processor.image_processor(images=image, return_tensors="pt")
        # image_enc.pixel_values: (N_patches, C*patch*patch) already flattened by
        # Qwen2-VL's image processor; image_grid_thw: (1, 3) = [t, h, w] patch grid.
        pixel_values = image_enc.pixel_values
        image_grid_thw = image_enc.image_grid_thw

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(full_ids),
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


def make_collate_fn(pad_token_id: int):
    def collate_fn(batch):
        input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
        # batch_size=1 in every script here (matches train_lora_baseline*.py / train_lora_qwenvl.py) —
        # cat works directly since each sample already has its own (N_i, D) pixel_values / (1,3) grid.
        pixel_values = torch.cat([b["pixel_values"] for b in batch], dim=0)
        image_grid_thw = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
        return {
            "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask,
            "pixel_values": pixel_values, "image_grid_thw": image_grid_thw,
        }
    return collate_fn
