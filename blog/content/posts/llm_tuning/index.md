---
title: <LLM Fine Tuning> 1. Supervised Fine-Tuning Using QLoRA
date: 2025-05-02T10:47:20Z
lastmod: 2025-05-02T10:47:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: cover.png
categories:
  - LLM
  - Supervised fine tuning
tags:
  - python
  - PEFT
  - QLoRA
  - Huggingface
  - Pytorch

# nolastmod: true
draft: false
---

I tried Tuning LLMs with RAG. Now it's time to directly tune LLMs using various methods.

# Load KoAlpaca Model

- Load KoAlpaca 5.8B model for fine-tuning

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
model = AutoModelForCausalLM.from_pretrained("beomi/KoAlpaca-Polyglot-5.8B")
```

# Quantization Using bitsandbytes

To quantize the model to np4 we need `bitsandbytes` library

pip version of bitsandbytes doesn't work when downloaded in an environment with CUDA>=12.4

- https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1152

So we need to build `bitsandbytes` library from scratch

```sh
$ git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/

# need python>=3.9, cmake>=3.22.1
$ cmake -DCOMPUTE_BACKEND=cuda -S .
$ cmake --build . --config Release

$ pip install -e .
```

## Using Quantization in Huggingface

Now that we've installed a working `bitsandbytes` library, we can use quantization in huggingface's `transformers` library.

We will fine-tune our model using `QLora` method.

### 1. Define BitsAndBytesConfig

```python
from transformers import BitsandBytesConfig

bnb_config = BitsAndBytesConfig(
	load_in_4bit=True,
	bnb_4bit_quant_type="nf4",
	bnb_4bit_compute_dtype=torch.bfloat16,
	bnb_4bit_use_double_quant=True,
	bnb_4bit_quant_storage=torch.bfloat16,
)
```

### 2. Pass Our Config to quantization_config option When Loading Model

```python
from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbits_training

model = AutoModelForCausalLM.from_pretrained(
	model_name,
	quantization_config=bnb_config,
	device_map={'': f'cuda:{local_rank}'}
)

# Call to make model ready for quantized training
model = prepare_model_for_kbit_training(model)
```

### Set Lora Config And Setup PEFT Model

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
	r=16,
	lora_alpha=32,
	target_modules = ['query_key_value'],
	lora_dropout=0.05,
	bias="none",
	task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
```

Now we can train our model with Huggingface's various Fine-Tuning Trainers such as `SFTTrainer`

Full example code:

```python
import json
import os

import pandas as pd
from pathlib import Path
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from torch.optim import AdamW


def formatting_prompts_func(example):
    return f"### 명령어: {get_instruction_prompt()}\n\n###맥락: {example['prompt']}\n\n### 답변: {example['completion']}<|endoftext|>"

def get_instruction_prompt():
    return f"""
당신은 팀 리더의 명령을 수행하여 협동 멀티 플레이어 게임을 클리어해야하는 게이머입니다.
리더의 아래 오더를 보고 지금 취해야 하는 행동이 무엇인지 알려주세요.
"""

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    prompts_list = list()
    completions_list = list()
    dataset = pd.read_csv('npc_action_tuning_dataset.csv')

    for row in dataset.itertuples():
        prompts_list.append(row.commands)
        completions_list.append(row.actions)

    train_dataset = Dataset.from_dict({
        'prompt': prompts_list,
        'completion': completions_list
    })

    train_dataset = train_dataset.map(
        lambda x: {'text': f"### 명령어: {get_instruction_prompt()}\n\n### 맥락: {x['prompt']}\n\n### 답변: {x['completion']}<|endoftext|>" }
    )

    model_name = "EleutherAI/polyglot-ko-5.8b"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={'': f'cuda:{local_rank}'}
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = train_dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
    columns_to_remove = ['prompt', 'completion', 'text']
    train_dataset = train_dataset.remove_columns(columns_to_remove)

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules = ['query_key_value'],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    response_template = "### 답변:"
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
        mlm=False
    )

    training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        output_dir="./distributed_output",
        num_train_epochs=50,
        learning_rate=2e-4,
        save_steps=500,
        max_seq_length=128,
        logging_steps=10,
        gradient_checkpointing=False,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./distributed_output/final_model")
```
