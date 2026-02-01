import os
import json
import torch
from pathlib import Path
from PIL import Image
from datasets import load_dataset, Features, Sequence, Value, Array2D, Array3D
from transformers import (
    LayoutLMv3Processor, 
    LayoutLMv3ForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification
)
import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from huggingface_hub import login

login("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")

# 1. SETUP PATHS & LABELS
DATA_DIR = Path(r"C:\Users\wgsom\Desktop\AI_Project\cv\ner_layoutlm")
MODEL_ID = "microsoft/layoutlmv3-base"
LABEL_LIST = [
    "O", "B-INVOICE_NO", "I-INVOICE_NO", "B-DATE", "I-DATE", 
    "B-CLIENT", "I-CLIENT", "B-TOTAL", "I-TOTAL", "B-CURRENCY", "I-CURRENCY"
]
ID2LABEL = {v: k for v, k in enumerate(LABEL_LIST)}
LABEL2ID = {k: v for v, k in enumerate(LABEL_LIST)}

# 2. LOAD DATASET
data_files = {
    "train": str(DATA_DIR / "train.jsonl"),
    "test": str(DATA_DIR / "test.jsonl"),
    "val": str(DATA_DIR / "val.jsonl"),
}
dataset = load_dataset("json", data_files=data_files)

# 3. INITIALIZE PROCESSOR
# apply_ocr=False because we already have our OCR data
processor = LayoutLMv3Processor.from_pretrained(MODEL_ID, apply_ocr=False)

def preprocess_data(examples):
    images = [Image.open(path).convert("RGB") for path in examples["image_path"]]
    words = examples["tokens"]
    boxes = examples["bboxes"]
    word_labels = examples["ner_tags"]

    encoding = processor(
        images, 
        words, 
        boxes=boxes, 
        word_labels=word_labels, 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )
    return encoding

# Map the preprocessing across the dataset
train_dataset = dataset["train"].map(preprocess_data, batched=True, remove_columns=dataset["train"].column_names)
val_dataset = dataset["val"].map(preprocess_data, batched=True, remove_columns=dataset["val"].column_names)

# 4. COMPUTE METRICS (for evaluation)
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [LABEL_LIST[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABEL_LIST[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }

# 5. INITIALIZE MODEL
model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_ID, 
    num_labels=len(LABEL_LIST),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# 6. TRAINING ARGUMENTS (Optimized for 1660 Ti)
training_args = TrainingArguments(
    output_dir="./layoutlmv3-oaca",
    max_steps=500,                  # Start with 1000 steps
    per_device_train_batch_size=2,   # Keep small for 6GB VRAM
    gradient_accumulation_steps=2,   # Effectively makes batch size 4
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=100,
    logging_steps=5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,                       # Critical for GPU speed/memory
    push_to_hub=False,
)

# 7. START TRAINING
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
    data_collator=DataCollatorForTokenClassification(processor.tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()