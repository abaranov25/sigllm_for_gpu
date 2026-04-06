# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import pickle
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)

DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_UNK_TOKEN = '<unk>'
DEFAULT_PAD_TOKEN = '<pad>'

VALID_NUMBERS = list('0123456789')
VALID_MULTIVARIATE_SYMBOLS = []

DEFAULT_MODEL = 'mistralai/Mistral-7B-Instruct-v0.2'


class HF:
    """Prompt Pretrained models on HuggingFace to forecast a time series.

    Args:
        name (str):
            Model name. Default to `'mistralai/Mistral-7B-Instruct-v0.2'`.
        sep (str):
            String to separate each element in values. Default to `','`.
        steps (int):
            Number of steps ahead to forecast. Default `1`.
        temp (float):
            The value used to modulate the next token probabilities. Default to `1`.
        top_p (float):
             If set to float < 1, only the smallest set of most probable tokens with
             probabilities that add up to `top_p` or higher are kept for generation.
             Default to `1`.
        raw (bool):
            Whether to return the raw output or not. Defaults to `False`.
        samples (int):
            Number of forecasts to generate for each input message. Default to `1`.
        padding (int):
            Additional padding token to forecast to reduce short horizon predictions.
            Default to `0`.
        multivariate_allowed_symbols (list):
            List of token strings to allow in addition to digits when generating.
            Default to `[]`.
    """

    def __init__(
        self,
        name=DEFAULT_MODEL,
        sep=',',
        steps=1,
        temp=1,
        top_p=1,
        raw=False,
        samples=1,
        padding=0,
        multivariate_allowed_symbols=VALID_MULTIVARIATE_SYMBOLS,
        cache_dir=None,
    ):
        self.name = name
        self.sep = sep
        self.steps = steps
        self.temp = temp
        self.top_p = top_p
        self.raw = raw
        self.samples = samples
        self.padding = padding
        self.multivariate_allowed_symbols = multivariate_allowed_symbols

        cache_dir = cache_dir or os.getenv("SIGLLM_CACHE_DIR")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=False)

        # special tokens
        special_tokens_dict = dict()
        if self.tokenizer.eos_token is None:
            special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        if self.tokenizer.bos_token is None:
            special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        if self.tokenizer.unk_token is None:
            special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        if self.tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN

        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # indicate the end of the time series

        # invalid tokens
        valid_tokens = []
        for number in VALID_NUMBERS:
            token = self.tokenizer.convert_tokens_to_ids(number)
            valid_tokens.append(token)

        for symbol in self.multivariate_allowed_symbols:
            valid_tokens.append(self.tokenizer.convert_tokens_to_ids(symbol))

        valid_tokens.append(self.tokenizer.convert_tokens_to_ids(self.sep))
        self.invalid_tokens = [
            [i] for i in range(len(self.tokenizer) - 1) if i not in valid_tokens
        ]

        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map='auto',
            torch_dtype=torch.float16,
        )

        self.model.eval()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def _delete_window_pkls(self):
        if self.cache_dir is None:
            return

        for pkl_file in self.cache_dir.glob("window_*.pkl"):
            try:
                pkl_file.unlink()
            except Exception as e:
                print(f"failed to delete {pkl_file}: {e}")

    def forecast(self, X, **kwargs):
        """Use GPT to forecast a signal.
    
        Args:
            X (ndarray):
                Input sequences of strings containing signal values.
    
        Returns:
            list:
                List of forecasted signal values.
        """
        print("began forecasting stage")
        print("X shape:", len(X))
        
        def is_empty_response(resp):
            if resp is None:
                return True
        
            # normalize to list
            if isinstance(resp, str):
                resp = [resp]
        
            if isinstance(resp, (list, tuple)):
                if len(resp) == 0:
                    return True
        
                for r in resp:
                    if r is None:
                        return True
                    if not isinstance(r, str):
                        return True
                    if r.strip() == "":
                        return True
        
                    # check for valid d0:<int>
                    if not re.search(r'd0:(\d+)', r):
                        return True
        
                return False
        
            return True
    
        def run_prediction_for_text(text):
            print(f"running sigllm for {self.samples} samples")
            tokenized_input = self.tokenizer([text], return_tensors='pt').to('cuda')
    
            input_length = tokenized_input['input_ids'].shape[1]
            average_length = input_length / len(text.split(self.sep))
            max_tokens = int((average_length + self.padding) * self.steps)
    
            generate_ids = self.model.generate(
                **tokenized_input,
                do_sample=True,
                max_new_tokens=max_tokens,
                temperature=self.temp,
                top_p=self.top_p,
                bad_words_ids=self.invalid_tokens,
                renormalize_logits=True,
                num_return_sequences=self.samples,
            )
    
            responses = self.tokenizer.batch_decode(
                generate_ids[:, input_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
    
            return responses
    
        combined_file = None
        if self.cache_dir is not None:
            combined_file = self.cache_dir / "all_responses.pkl"
    
            if combined_file.exists():
                print(f"found combined cache: {combined_file}")
                with open(combined_file, "rb") as f:
                    cached = pickle.load(f)
    
                if isinstance(cached, dict):
                    all_responses = cached["responses"]
                else:
                    all_responses = cached
    
                print("loaded combined responses:", len(all_responses))
    
                if len(all_responses) != len(X):
                    raise ValueError(
                        f"Combined cache length {len(all_responses)} does not match input length {len(X)}"
                    )
    
                bad_indices = [i for i, resp in enumerate(all_responses) if is_empty_response(resp)]
    
                print("num bad cached responses:", len(bad_indices))
                if bad_indices:
                    print("bad indices (first 20):", bad_indices[:20])
    
                    for i in tqdm(bad_indices, desc="repairing empty cached responses"):
                        print(f"rerunning window {i}")
                        responses = run_prediction_for_text(X[i])
                        print(f"new responses for window {i}: {responses}")
                        all_responses[i] = responses
    
                    with open(combined_file, "wb") as f:
                        pickle.dump({"responses": all_responses}, f)
    
                    print(f"patched and rewrote combined cache: {combined_file}")
                else:
                    print("no bad cached responses found")
    
                self._delete_window_pkls()
                print("deleted all window cache files")
                return all_responses
    
        all_responses = []
    
        for i, text in enumerate(tqdm(X)):
            cache_file = None
            if self.cache_dir is not None:
                cache_file = self.cache_dir / f"window_{i:06d}.pkl"
                if cache_file.exists():
                    with open(cache_file, "rb") as f:
                        cached = pickle.load(f)
                    responses = cached["responses"] if isinstance(cached, dict) else cached
                    all_responses.append(responses)
                    print(f"found window {i}")
                    print(f"found responses {responses}")
                    continue
    
            print(f"did not find window {i}, running sigllm")
            responses = run_prediction_for_text(text)
            all_responses.append(responses)
    
            print("responses:", responses)
    
            if cache_file is not None:
                with open(cache_file, "wb") as f:
                    pickle.dump({"responses": responses}, f)
    
        print("done with predictions!")
        print("number of responses:", len(all_responses))
    
        if self.cache_dir is not None:
            combined_file = self.cache_dir / "all_responses.pkl"
            with open(combined_file, "wb") as f:
                pickle.dump({"responses": all_responses}, f)
            print(f"wrote combined cache: {combined_file}")
    
            self._delete_window_pkls()
            print("deleted all window cache files")
    
        return all_responses