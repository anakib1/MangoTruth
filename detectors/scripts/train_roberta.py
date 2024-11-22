from dataclasses import asdict
from uuid import uuid4

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
    training_args = TrainingArguments("test_trainer", report_to="none")

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
    MODEL = "distilbert/distilbert-base-uncased"
    SIZE = 5000

    TEST_SIZE = 1000

    data = load_dataset('anakib1/mango-truth', 'xlsum')

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    finetune_data = process_data_split(data['train'], SIZE, tokenizer).train_test_split(test_size=0.3)
    test_data = process_data_split(data['test'], TEST_SIZE, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "HUMAN", 1: "AI"}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2, id2label=id2label,
                                                               label2id=label2id)
    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=100,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        save_total_limit=2,
        report_to="neptune",
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

    ret = Conclusion("empty".encode("UTF-8"), "roberta", ["xlsum"],
                     perform_evaluation(finetune_data["train"], model, data_collator),
                     perform_evaluation(test_data, model, data_collator))

    nexus.conclude_run(uuid4(), conclusion=ret, extra_data={"INFO": "DEBUG"})


if __name__ == '__main__':
    load_dotenv()

    main()
