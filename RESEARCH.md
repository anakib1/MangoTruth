# Research outline

This document provides outline of research done as part of `MANGO-TRUTH` project.

For documentation, configuration, deployment and contribution address [README](README.md).

## Dataset

We create and publish diverse dataset consisting of LLM outputs for ukrainian language for several popular open-source
and some popular enterprise models.

It's split in the following splits:

- AYA ([parent](https://huggingface.co/datasets/CohereForAI/aya_dataset)). Original dataset consists of prompts and
  human answers for them in multiple languages. We extract data in ukrainian and provide responses
  using `llama-3(2B, 8B)`,  `gemma(27B)` models.
- XLSUM. As collection of data from news article in ukrainian, we provide completions of news using `gpt-4o-mini` model.
- More to come!

### Stats

| AYA   | Models               | # Human | # Total AI |
|-------|----------------------|---------|------------|
| 3.65K | llama3(2), llama3(8) | 2.65K   | 3K         |
| 60.1K | gpt-4o-mini          | 57.5K   | 2.6K       |

Dataset can be accessed [here](https://huggingface.co/datasets/anakib1/mango-truth).

## Models

### Protocol

We evaluate scope of different ML models for their quality of separation AI/Human text and separation of different
language models.

Evaluation protocol consists of testing on separate test set taken from `AYA` and `XLSUM` with guarantees that model did
not
see test data during training time.

Training is performed on parts of specified datasets with balanced number of human/ai samples. Hyperparameter tuning is
done on separate validation dataset.

### Zoo

We evaluate following strategies:

- Entropy-based (models based on probability distribution of output)
    - [Ghostbuster](https://arxiv.org/pdf/2305.15047) - multiple distribution of probabilities is extended using set of
      algebraic functions with fitted logistic regressor on top.
    - Perplexity baseline - attempt to separate data based only on perplexity of output.
    - DetectGPT - model that attempts to separate samples based on perplexity **difference** after rephrasing.
    - GTLR - method based on dividing tokens in histograms based on per-token probability.
    - LogRank - attempt to classify text based on histogram of log ranks of tokens.
- Miscellaneous
    - DNA-GPT - attempt to separate text based on repetitive samples produced by LLM after similar context.
- Neural based
    - BERT Finetune - finetuning model for AI text detection.
    - RoBERTa Finetune - finetuning model for AI text detection.

### Results

To make claims of our contributions it's essential to review existing methods in the same conditions and evaluate
improvements. This table includes all our models and existing methods for gpt detection.

Evaluation protocol:

| Method              | Compute price (*)    | F1        | AUC       | tpr@1%fpr |
|---------------------|----------------------|-----------|-----------|-----------|
| GLTR                | TBD                  | -         | -         | -         |
| Ghostbuster         | 1 openai call + O(1) | **0.981** | 0.971     | -         |
| Perplexity baseline | -                    | 0.662     | 0.5       | -         |
| LogRank             |
| DetectGDP           |
| DNA-GPT             |
| BERT                | 67M                  | 0.916     | 0.988     | 0.81      |
| RoBERTA             | 125M                 | 0.971     | **0.997** | **0.923** |
| -------------       | -                    | -         |           |           |
| GPTZero             | TBD                  | TBD       | TBD       | TBD       |
| ZeroGPT             | TBD                  | TBD       | TBD       | TBD       |
| OpenAIDetector      | TBD                  | TBD       | TBD       | TBD       |

### Results
- 
