#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data.py
# @Time      :2024/07/07 16:38:34
# @Author    :Lifeng
# @Description :
import torch
import pandas as pd
from typing import Dict
from typing import Any, List, Tuple
from transformers import AutoProcessor
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from dataclasses import dataclass


@dataclass
class QaImageOutput:
    q_input_ids: torch.Tensor
    pixel_values: torch.Tensor
    a_input_ids: torch.Tensor


class LlavaDataset(Dataset):
    def __init__(self, dataset_dir) -> None:
        super().__init__()
        self.chat_data, self.image_dir = self.build_dataset(dataset_dir)

    def build_dataset(self, data_dir) -> Tuple[List[Dict], Path]:
        data_dir = Path(data_dir)
        chat_file = data_dir.joinpath("chat.json")
        image_dir = data_dir.joinpath("images_dl")

        chat_data = pd.read_json(chat_file).to_dict(orient="records")

        return chat_data, image_dir

    def __len__(self) -> int:
        return len(self.chat_data)

    def __getitem__(self, index) -> Tuple[str, str, Path]:
        cur_data = self.chat_data[index]
        conversations = cur_data["conversations"]

        humman_input = conversations[0].get("value")
        chatbot_output = conversations[1].get("value")

        image_path = self.image_dir.joinpath(cur_data.get("image"))

        return humman_input, chatbot_output, image_path


def build_qaimage(
    processor: AutoProcessor, q_text: str, a_text: str, image_path: Path
) -> QaImageOutput:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": q_text},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_file = image_path

    raw_image = Image.open(image_file)
    inputs = processor(prompt, raw_image, return_tensors="pt")

    a_input_ids = processor.tokenizer(
        a_text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )["input_ids"]

    res = QaImageOutput(
        q_input_ids=inputs.get("input_ids"),
        pixel_values=inputs.get("pixel_values"),
        a_input_ids=a_input_ids,
    )
    return res


class TrainLLavaModelCollator:
    def __init__(self, processor: AutoProcessor, IGNORE_TOKENS: int) -> None:
        self.processor = processor
        self.ingore_tokens = IGNORE_TOKENS

    def convert_one_piece(self, q_input_ids: torch.Tensor, a_input_ids: torch.Tensor):
        input_ids = torch.concat(
            [
                q_input_ids,
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )
        labels = torch.concat(
            [
                torch.full(q_input_ids.shape, self.ingore_tokens),
                a_input_ids,
                torch.tensor(self.processor.tokenizer.eos_token_id).reshape(1, -1),
            ],
            axis=1,
        )

        return input_ids, labels

    def __call__(self, features: List) -> dict[str, Any]:
        input_ids_list = []
        labels_list = []
        pixel_values = []
        max_input_len_list = []

        for feature in features:
            qaimage_output = build_qaimage(
                self.processor, feature[0], feature[1], feature[2]
            )
            temp_input_ids, temp_labels = self.convert_one_piece(
                qaimage_output.q_input_ids, qaimage_output.a_input_ids
            )
            max_input_len_list.append(temp_input_ids.shape[1])
            input_ids_list.append(temp_input_ids)
            labels_list.append(temp_labels)
            pixel_values.append(qaimage_output.pixel_values)

        max_input_len = max(max_input_len_list)

        final_input_ids = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.processor.tokenizer.pad_token_id,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(input_ids_list)
            ]
        )

        final_labels = torch.concat(
            [
                torch.concat(
                    [
                        torch.full(
                            (1, max_input_len - max_input_len_list[index]),
                            self.ingore_tokens,
                        ),
                        value,
                    ],
                    axis=1,
                )
                for index, value in enumerate(labels_list)
            ]
        )

        final_pixel_values = torch.concat(pixel_values, axis=0)

        attention_mask = torch.ones_like(input=final_input_ids)
        attention_mask[final_input_ids == self.processor.tokenizer.pad_token_id] = 0

        return {
            "input_ids": final_input_ids,
            "labels": final_labels,
            "pixel_values": final_pixel_values,
            "attention_mask": attention_mask,
        }


if __name__ == "__main__":
    data_dir = "/root/autodl-tmp/Models/LLaVA-CC3M-Pretrain-595K"
    llavadataset = LlavaDataset(dataset_dir=data_dir)
    print(len(llavadataset))
    print(llavadataset[100])
