import argparse
import pathlib
from uuid import uuid4
import traceback

import numpy as np
import torch
from datasets import concatenate_datasets
from datasets import load_dataset
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from detectors.metrics import Conclusion
from detectors.neptune.nexus import NeptuneNexus
from detectors.perplexity.model import PerplexityModel
from detectors.utils.math import safe_sigmoid
from detectors.utils.training import calculate_classification


def batch_calculate_perplexity(dataset, tokenizer, llm, batch_size):
    results = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenizer(dataset['output'][i:i + batch_size], padding='max_length', truncation=True,
                           return_tensors='pt',
                           max_length=tokenizer.model_max_length)
        with torch.no_grad():
            logits = llm(**inputs.to(llm.device))
        raw_perplexity = torch.nn.functional.cross_entropy(logits['logits'].permute(0, 2, 1), inputs['input_ids'],
                                                           reduction='none')

        raw_perplexity[inputs['input_ids'] == tokenizer.pad_token_id] = 0
        perplexities = raw_perplexity.sum(dim=1) / inputs['attention_mask'].sum(dim=1)
        results.append(perplexities.exp().cpu())

    return torch.concatenate(results, dim=0).numpy()


def train_threshold(X, y):
    """
    Solves regression problem by finding the best threshold that separates binary data in y based by X.
    :param X: 1d array
    :param y: 1d boolean array
    :return: the best threshold C.
    """
    thresholds = np.sort(X)
    max_f1 = 0
    optimal_C = thresholds[0]

    for C in thresholds:
        predictions = (X > C).astype(int)
        f1 = f1_score(y, predictions)

        if f1 > max_f1:
            max_f1 = f1
            optimal_C = C

    return optimal_C


def train_scaling_factor(X, y, c):
    """
     Solves regression problem by finding the best scaling factor that maximises ROC curve for binary data in y based by X.
     Data is approxiumated by sigmoid distribution with argument k * (x - c)
    :param X: 1d array
    :param y: 1d boolean array
    :param c: threshold factor
    :return: the best scaling threshold k.
    """
    max_roc = 0
    optimal_K = None
    for k in np.linspace(0.0001, 1, 1000):
        probabilities = safe_sigmoid(k * (np.array(X) - c))

        roc = roc_auc_score(y, probabilities)
        if roc > max_roc:
            max_roc = roc
            optimal_K = k

    return optimal_K


def catch_and_return(callable, default):
    try:
        return callable()
    except:
        print(traceback.format_exc())
        return default


def main(args):
    load_dotenv()
    run = uuid4()

    print(f'Starting run {run} with following arguments: ', args)

    model = PerplexityModel(args.perplexity_model)

    data = load_dataset(args.dataset_handle, args.dataset_config)['train']
    labler = data.features['label']

    human_data = data.filter(lambda x: x['label'] == labler.str2int('human')).take(args.human_samples)
    llm_data = data.filter(lambda x: x['label'] == labler.str2int('gpt-4o-mini')).take(args.llm_samples)

    labels = np.concat([np.ones(len(human_data)), np.zeros(len(llm_data))])
    data = concatenate_datasets([human_data, llm_data])

    if model.model is not None:  # Can use batching
        perplexities = batch_calculate_perplexity(data, model.tokenizer, model.model, args.batch_size)
    else:
        perplexities = np.array([catch_and_return(lambda: model.predict_openai(x['output']), np.nan) for x in tqdm(data)])

    perplexities[perplexities == np.nan] = np.mean(perplexities[perplexities != np.nan])

    bins = np.linspace(np.percentile(perplexities, 5),
                       np.percentile(perplexities, 95), 50)

    print(f"Human Mean Perplexity: {perplexities[:len(human_data)].mean()}")
    print(f"AI Mean Perplexity: {perplexities[len(human_data):].mean()}")

    plt.figure(figsize=(10, 6))

    plt.hist(perplexities[:len(human_data)], bins=bins, alpha=0.7, label='Human', color='blue', edgecolor='black',
             density=True)
    plt.hist(perplexities[len(human_data):], bins=bins, alpha=0.7, label='AI', color='green', edgecolor='black',
             density=True)

    plt.xlabel('Perplexity Score')
    plt.ylabel('Frequency')
    plt.title('Histogram Comparison of Perplexity Scores')
    plt.legend()
    pathlib.Path('./out').mkdir(exist_ok=True)
    plt.savefig(f'./out/perplexity_distribution_{run}.png')
    print(f'Saved perplexity distribution to "./out/perplexity_distribution_{run}.png"')

    X_train, X_test, y_train, y_test = train_test_split(perplexities, labels, shuffle=True, test_size=0.3,
                                                        stratify=labels)
    model.perplexity_threshold = train_threshold(X_train, y_train)
    model.scaling_factor = train_scaling_factor(X_train, y_train, model.perplexity_threshold)

    y_train_hat = safe_sigmoid(model.scaling_factor * (X_train - model.perplexity_threshold))
    y_test_hat = safe_sigmoid(model.scaling_factor * (X_test - model.perplexity_threshold))

    train = calculate_classification(y_train, y_train_hat)
    valid = calculate_classification(y_test, y_test_hat)

    print(f'Training {run} finished successfully, uploading results to nexus.')

    conclusion = Conclusion(
        weights=model.store_weights(),
        detector_handle='perplexity',
        datasets=[args.dataset_config],
        train_conclusion=train,
        validation_conclusion=valid
    )

    training_nexus = NeptuneNexus()

    training_nexus.conclude_run(run, conclusion=conclusion,
                                extra_data={"info": 'CORRECT',
                                            "human_samples": args.human_samples,
                                            "llm_samples": args.llm_samples,
                                            "perplexity_model": args.perplexity_model})

    print(f'Upload of run {run} finished successfully, verifying results persistence.')

    obj = PerplexityModel()
    obj.load_weights(training_nexus.load_run_weights(run))

    print(f'Run {run} finished successfully.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train language models and evaluate using probability estimators.")

    parser.add_argument("--dataset_handle", type=str, default="anakib1/mango-truth",
                        help="Dataset handle to use for loading data.")
    parser.add_argument("--dataset_config", type=str, default="xlsum",
                        help="Dataset configuration to load.")
    parser.add_argument("--human_samples", type=int, default=1000,
                        help="Number of human samples to use for training.")
    parser.add_argument("--llm_samples", type=int, default=1000,
                        help="Number of LLM samples to use for training.")
    parser.add_argument("--perplexity_model", type=str, default='babbage-002',
                        help="LLM model for perplexity computation to use. Both HF and openai models are supported.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for batched training (HF models)")

    args = parser.parse_args()
    main(args)
