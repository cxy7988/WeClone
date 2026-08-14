"""vLLM compatibility model for text-only Qwen3.5 merge artifacts."""

from collections.abc import Iterable
from typing import Any, ClassVar, Literal

import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM


class WeCloneQwen3_5ForCausalLM(Qwen3_5ForCausalLM):
    """Accept weights that retained the multimodal wrapper's language-model prefix."""

    supports_mrope: ClassVar[Literal[True]] = True

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[Any],
    ) -> tuple[torch.Tensor, int]:
        if mm_features:
            raise ValueError("The text-only Qwen3.5 loader does not accept multimodal inputs")
        positions = torch.arange(len(input_tokens), dtype=torch.long).expand(3, -1)
        return positions, 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        prefix = "model.language_model."
        mapped_weights = (
            (f"model.{name.removeprefix(prefix)}", tensor)
            if name.startswith(prefix)
            else (name, tensor)
            for name, tensor in weights
        )
        return super().load_weights(mapped_weights)
