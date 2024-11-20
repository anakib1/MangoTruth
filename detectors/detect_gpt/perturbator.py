from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from typing import List
import re
from tqdm.auto import tqdm


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


pattern = re.compile(r"<extra_id_\d+>")

import numpy as np


def tokenize_and_mask(text, span_length, pct):
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + 2 * 2)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - 2)
        search_end = min(len(tokens), end + 2)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


class Perturbator:
    def __init__(self, model_handle: str = 'google/mt5-base'):
        self.model = MT5ForConditionalGeneration.from_pretrained(model_handle, device_map='auto')
        self.tokenizer = MT5Tokenizer.from_pretrained(model_handle, legacy=False)
        self.device = self.model.device

    def _replace_masks(self, texts):
        n_expected = count_masks(texts)
        stop_id = self.tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
        tokens = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model.generate(**tokens, max_length=250, do_sample=True, top_p=1.0,
                                      num_return_sequences=1, eos_token_id=stop_id)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=False)

    def perturb_texts_(self, texts: List[str]):
        masked_texts = [tokenize_and_mask(x, 2, 0.2) for x in texts]
        raw_fills = self._replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

        # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
        attempts = 1
        while '' in perturbed_texts:
            idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
            print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
            masked_texts = [tokenize_and_mask(x, 2, 0.2) for idx, x in enumerate(texts) if
                            idx in idxs]
            raw_fills = self._replace_masks(masked_texts)
            extracted_fills = extract_fills(raw_fills)
            new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
            for idx, x in zip(idxs, new_perturbed_texts):
                perturbed_texts[idx] = x
            attempts += 1

            if attempts > 3:
                print(f'ERROR: could not regenerate. Returning originals.')
                return texts

        return perturbed_texts

    def perturbate(self, texts: List[str]):
        chunk_size = 8
        outputs = []
        for i in tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
            outputs.extend(self.perturb_texts_(texts[i:i + chunk_size]))
        return outputs


texts = [
    "Багато людей вважає, що Карлсон - це просто поширене прізвище в Швеції, але насправді Карлсон живе на даху! Це герой повісті шведської письменниці Астрід Ліндґрен «Малий і Карлсон, що живе на даху». Дія повісті відбувається в 1950-і роки, де в одному будинку проживають два головні герої — Сванте, молодша дитина сім'ї Свантесон, на прізвисько Малий, і на даху — Карлсон.",
    "Помара́нчева революція - це кампанія протестів, мітингів, пікетів, страйків та інших актів громадянської непокори в Україні, організована і проведена прихильниками Віктора Ющенка, основного кандидата від опозиції на президентських виборах у листопаді-грудні 2004 року, після оголошення Центральною виборчою комісією попередніх результатів, згідно з якими нібито переміг його суперник — Віктор Янукович. Результат виборів визнано фальсифікацією і оголошено другий тур, в якому під оком незалежних спостерігачів переміг Віктор Ющенко"]

if __name__ == '__main__':
    perturbator = Perturbator()
    print(perturbator.perturbate(texts))
