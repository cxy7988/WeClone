"""Hide upstream flash-attn from native vLLM CUDA service processes.

vLLM ships its own matching ``vllm_flash_attn`` extension on CUDA. This empty
package intentionally has no ``ops`` submodule, making vLLM's optional probe
fall back to its bundled implementation without affecting training processes.
"""
