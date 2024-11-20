import argparse
import os
import pickle
import random
from typing import Any
from uuid import uuid4

from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from detectors.ghostbuster.features import extract_features
from detectors.ghostbuster.model import GhostbusterDetector
from detectors.ghostbuster.ngrams import UnigramModel, TrigramModel
from detectors.ghostbuster.openai import OpenaiProbabilityEstimator
from detectors.metrics import Conclusion
from detectors.neptune.nexus import NeptuneNexus
from detectors.utils.training import calculate_classification


def get_obj_from_persistance(filename: str) -> Any:
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None


def persist_obj(obj: Any, filename: str) -> None:
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def main(args):
    load_dotenv()
    run = uuid4()

    print(f'Starting run {run} with following arguments: ', args)
    dataset = load_dataset(args.dataset_handle, args.dataset_config)

    length_match = dataset['train'].filter(lambda x: len(x['output']) < args.max_length)
    reference = length_match.filter(lambda x: x['label'] == 0).take(args.human_samples)
    generated = length_match.filter(lambda x: x['label'] == 3)

    X = reference['output'] + generated['output']
    y = [0] * len(reference) + [1] * len(generated)

    unigram = UnigramModel(tokenizer_handle='gugarosa/cl100k_base')
    unigram.train(X)

    trigram = TrigramModel(tokenizer_handle='gugarosa/cl100k_base')
    trigram.train(X)

    estimator = OpenaiProbabilityEstimator(model_name=args.llm_handles[0])

    # IMPORTANT: Assertions of not broken tokenizer.
    for _ in range(10):
        i = random.randint(0, len(X) - 1)
        assert estimator.get_text_log_proba(X[i])[1].shape == unigram.get_text_log_proba(X[i])[1].shape

    models = [unigram, trigram, estimator]
    feats = get_obj_from_persistance('feats.pkl')
    if feats is None: feats = []

    for i in tqdm(range(len(feats), len(X))):
        feats.append(extract_features([model.get_text_log_proba(X[i])[1] for model in models]))
        persist_obj(feats, 'feats.pkl')

    print(f'Feature extraction for run {run} finished successfully.')

    X_train, X_test, y_train, y_test = train_test_split(feats, y, shuffle=True, test_size=0.3, stratify=y)

    model = make_pipeline(StandardScaler(), CalibratedClassifierCV(LogisticRegression(C=1, max_iter=1000)))
    model.fit(X_train, y_train)

    y_test_hat = model.predict_proba(X_test)
    y_train_hat = model.predict_proba(X_train)

    train = calculate_classification(y_train, y_train_hat[:, 1])
    valid = calculate_classification(y_test, y_test_hat[:, 1])

    print(f'Training {run} finished successfully, uploading results to nexus.')

    estimator = GhostbusterDetector(model, models)

    conclusion = Conclusion(
        weights=estimator.store_weights(),
        detector_handle='ghostbuster',
        datasets=[args.dataset_config],
        train_conclusion=train,
        validation_conclusion=valid
    )

    training_nexus = NeptuneNexus()

    training_nexus.conclude_run(run, conclusion=conclusion, extra_data={"info": 'CORRECT'})

    print(f'Upload of run {run} finished successfully, verifying results persistence.')

    obj = GhostbusterDetector()
    obj.load_weights(training_nexus.load_run_weights(run))

    print(f'Run {run} finished successfully.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train language models and evaluate using probability estimators.")

    parser.add_argument("--dataset_handle", type=str, default="anakib1/mango-truth",
                        help="Dataset handle to use for loading data.")
    parser.add_argument("--dataset_config", type=str, default="xlsum",
                        help="Dataset configuration to load.")
    parser.add_argument("--max_length", type=int, default=1750,
                        help="Maximum output length for filtering data.")
    parser.add_argument("--human_samples", type=int, default=1000,
                        help="Number of human samples to use for training.")
    parser.add_argument("--llm_handles", type=str, nargs='+', default=['babbage-002'],
                        help="List of LLM model handles to use.")

    args = parser.parse_args()
    main(args)
