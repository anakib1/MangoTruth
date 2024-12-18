from dataclasses import asdict
from uuid import uuid4
import argparse

import torch
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import EarlyStoppingCallback

from detectors.metrics import Conclusion
from detectors.neptune.nexus import NeptuneNexus
from detectors.utils.training import calculate_classification
from detectors.utils.training import report_classification


def preprocess(row, tokenizer):
    return tokenizer(row["output"], truncation=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = torch.softmax(torch.as_tensor(predictions), dim=1)[:, 1].numpy()

    res = calculate_classification(labels, predictions)

    return asdict(res)


def compute_final_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = torch.softmax(torch.as_tensor(predictions), dim=1)[:, 1].numpy()
    global results
    results = report_classification(labels, predictions)
    return asdict(results)


def perform_evaluation(eval_data, model, data_collator):
    training_args = TrainingArguments("test_trainer", report_to="none",
                                      per_device_eval_batch_size=32)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=eval_data,
        data_collator=data_collator,
        eval_dataset=eval_data,
        compute_metrics=compute_final_metrics
    )

    trainer.evaluate()

    return results


def process_data_split(split, selection_size, tokenizer):
    split = split.filter(lambda x: len(x['output']) < 15000)
    human = split.filter(lambda x: x['label'] == 0)
    ai = split.filter(lambda x: x['label'] == 3).map(lambda x: {"label": 1})

    selection_size = min(selection_size, len(human), len(ai))

    split = concatenate_datasets([human.shuffle().take(selection_size), ai.shuffle().take(selection_size)])

    return split.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=['prompt', 'user_id'])


def main():
    parser = argparse.ArgumentParser(description='Fine-tune a transformer model.')
    parser.add_argument('--dataset_handle', type=str, default='anakib1/mango-truth', help='The dataset handle to use.')
    parser.add_argument('--dataset_config', type=str, default='xlsum', help='The dataset configuration.')
    parser.add_argument('--model_handle', type=str, default='FacebookAI/roberta-base',
                        help='The model handle to use.')
    parser.add_argument('--train_size', type=int, default=5000, help='Number of training examples.')
    parser.add_argument('--test_size', type=int, default=1000, help='Number of test examples.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for both training and evaluation.')

    args = parser.parse_args()

    model_handle = args.model_handle
    train_size = args.train_size
    test_size = args.test_size
    dataset_handle = args.dataset_handle
    dataset_config = args.dataset_config
    bs = args.batch_size

    data = load_dataset(dataset_handle, dataset_config)

    tokenizer = AutoTokenizer.from_pretrained(model_handle)

    finetune_data = process_data_split(data['train'], train_size, tokenizer).train_test_split(test_size=0.3)
    test_data = process_data_split(data['test'], test_size, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "HUMAN", 1: "AI"}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_handle,
        num_labels=2,
        id2label=id2label,
        label2id=label2id
    )

    run_surname = f"tune-{model_handle}-{train_size}"

    training_args = TrainingArguments(
        output_dir=run_surname,
        save_total_limit=3,
        learning_rate=2e-5,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        report_to="neptune",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=finetune_data["train"],
        eval_dataset=finetune_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer.train()

    nexus = NeptuneNexus()

    ret = Conclusion("empty".encode("UTF-8"), run_surname, [f"xlsum[{train_size}]"],
                     perform_evaluation(finetune_data["train"], model, data_collator),
                     perform_evaluation(test_data, model, data_collator))

    nexus.conclude_run(uuid4(), conclusion=ret, extra_data={"INFO": "DEBUG"})


if __name__ == '__main__':
    load_dotenv()

    main()
