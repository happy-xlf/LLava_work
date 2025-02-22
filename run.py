import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from transformers import (
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
)


from train_llava.data import LlavaDataset, TrainLLavaModelCollator

from train_llava.util import print_trainable_parameters

logger = logging.getLogger(__name__)

# import debugpy

# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="show_model/model001")
    train_type: Optional[str] = field(
        default="use_lora",
        metadata={
            "help": """
            1. use_lora:使用lora训练,
            2. none:全量参数训练;
            3. freeze_vision:只冻结vision_tower进行训练
            """
        },
    )


@dataclass
class DataArguments:
    build_data_from_web: bool = field(
        default=False, metadata={"help": "是否使用web获得数据"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    web_host_ip: str = field(default="0.0.0.0", metadata={"help": "web端的数据ip"})
    # source_length: int = field(default=128)
    # target_length: int = field(default=512)


def load_model_processor(modelargs: ModelArguments):
    model = LlavaForConditionalGeneration.from_pretrained(
        modelargs.model_name_or_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = LlavaProcessor.from_pretrained(modelargs.model_name_or_path)

    if modelargs.train_type == "use_lora":
        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"],
        )
        model = get_peft_model(model, config)
        # model.print_trainable_parameters()

    elif modelargs.train_type == "none":
        logging.warning("使用全量参数进行训练")

        pass
    elif modelargs.train_type == "freeze_vision":
        logging.warning("冻结vision_tower网络层，剩下的网络权重进行训练")

        for param in model.vision_tower.parameters():
            param.requires_grad = False
    print_trainable_parameters(model)

    return model, processor


def load_dataset_collator(processor, dataargs: DataArguments):
    if dataargs.build_data_from_web:
        # llava_dataset = DatasetReceiveByWeb(
        #     dataargs.web_host_ip,
        # )
        # logging.warning("从网络层进行数据初始化")

        # if len(llava_dataset) <= 0:
        #     raise ValueError("数据出现问题，无法进行web数据初始化")
        # data_collator = TrainLlavaModelCollatorByWeb(processor, -100)
        pass
    else:

        llava_dataset = LlavaDataset(
            dataargs.data_path  # "data/liuhaotian/LLaVA-CC3M-Pretrain-595K"
        )
        data_collator = TrainLLavaModelCollator(processor, -100)

    return llava_dataset, data_collator


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model, processor = load_model_processor(model_args)
    train_dataset, data_collator = load_dataset_collator(processor, data_args)

    if data_args.build_data_from_web:
        # trainer = WebTrainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset,
        #     eval_dataset=None,
        #     data_collator=data_collator,
        # )
        pass
    else:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=None,
            data_collator=data_collator,
        )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    train()
