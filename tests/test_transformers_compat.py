from transformers import PreTrainedTokenizerBase

from weclone.train.transformers_compat import patch_vllm_010_transformers5_tokenizer


def test_vllm_tokenizer_compat_property(monkeypatch) -> None:
    monkeypatch.delattr(PreTrainedTokenizerBase, "all_special_tokens_extended", raising=False)

    assert patch_vllm_010_transformers5_tokenizer()

    class TokenizerStub:
        SPECIAL_TOKENS_ATTRIBUTES = ["eos_token", "pad_token"]

        def __init__(self) -> None:
            self._special_tokens_map = {"eos_token": "</s>", "pad_token": "</s>"}
            self._extra_special_tokens = ["<image>"]

    stub = TokenizerStub()
    descriptor = PreTrainedTokenizerBase.all_special_tokens_extended
    assert descriptor.__get__(stub, TokenizerStub) == ["</s>", "<image>"]
