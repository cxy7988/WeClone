"""Narrow runtime compatibility fixes for supported Transformers releases."""

import inspect
from functools import wraps
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

from weclone.utils.log import logger


class _MissingAttentionSink:
    """Preserve ``None`` through Transformers 5.6.0's unconditional ``.to`` call."""

    def to(self, *_args, **_kwargs):
        return None


def patch_vllm_010_transformers5_tokenizer() -> bool:
    """Restore the tokenizer API expected by vLLM 0.10 on Transformers 5.

    vLLM 0.10 caches ``all_special_tokens_extended`` while constructing its
    tokenizer wrapper. Transformers 5 removed that legacy property, causing
    vLLM-backed inference to fail before the model engine starts. Recreate the
    old read-only view from Transformers 5's internal special-token storage.

    Returns whether the compatibility property was installed.
    """
    import transformers
    from transformers import PreTrainedTokenizerBase

    try:
        vllm_version = Version(version("vllm"))
    except PackageNotFoundError:
        return False

    if Version(transformers.__version__).major < 5 or vllm_version.release[:2] != (0, 10):
        return False

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return False

    @property
    def all_special_tokens_extended(self):
        """Return special-token objects without converting AddedToken to str."""
        tokens = []
        seen = set()
        special_tokens_map = self.__dict__.get("_special_tokens_map", {})

        for attribute in self.SPECIAL_TOKENS_ATTRIBUTES:
            token = special_tokens_map.get(attribute)
            if token is not None and str(token) not in seen:
                tokens.append(token)
                seen.add(str(token))

        for token in self.__dict__.get("_extra_special_tokens", []):
            if str(token) not in seen:
                tokens.append(token)
                seen.add(str(token))

        return tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = all_special_tokens_extended
    logger.warning(
        "Applied vLLM 0.10 compatibility patch for Transformers 5 tokenizer "
        "all_special_tokens_extended."
    )
    return True


def patch_transformers_560_flash_attention_none_sink() -> bool:
    """Fix Qwen3.5 FA2 when Transformers 5.6.0 receives ``s_aux=None``.

    Transformers 5.6.0 unconditionally evaluates ``s_aux.to(query.dtype)`` in
    its FlashAttention integration. Qwen3.5 does not provide an attention sink,
    so the first forward pass fails before reaching FlashAttention. Registering
    this wrapper preserves the intended ``None`` value without modifying the
    installed package. Returns whether the patch was applied.
    """
    import transformers
    from transformers import modeling_utils
    from transformers.integrations import flash_attention as flash_attention_module

    if Version(transformers.__version__).release[:3] != (5, 6, 0):
        return False

    current = modeling_utils.ALL_ATTENTION_FUNCTIONS.get_interface("flash_attention_2", lambda: None)
    if getattr(current, "_weclone_none_sink_patch", False):
        return False

    original = flash_attention_module.flash_attention_forward
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError):
        source = ""

    if "s_aux=s_aux.to(query.dtype)" not in source:
        return False

    @wraps(original)
    def patched_flash_attention_forward(*args, **kwargs):
        if kwargs.get("s_aux") is None:
            kwargs["s_aux"] = _MissingAttentionSink()
        return original(*args, **kwargs)

    patched_flash_attention_forward._weclone_none_sink_patch = True
    flash_attention_module.flash_attention_forward = patched_flash_attention_forward
    modeling_utils.AttentionInterface.register("flash_attention_2", patched_flash_attention_forward)
    logger.warning(
        "Applied Transformers 5.6.0 FlashAttention compatibility patch for Qwen3.5 s_aux=None."
    )
    return True
