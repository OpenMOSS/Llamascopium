"""Model definitions for Llamascopium.

This package contains the sparse dictionary base class, protocol definitions,
and all concrete model implementations.
"""
from llamascopium.models.matryoshka_sae import MatryoshkaSAEConfig, MatryoshkaSparseAutoEncoder
from llamascopium.models.protocols import (
    ActiveSubspaceInitializable,
    DatasetNormStandardizable,
    EncoderBiasInitializable,
    EncoderInitializable,
    NormComputing,
    NormConstrainable,
)
from llamascopium.models.sparse_dictionary import (
    SparseDictionary,
    SparseDictionaryConfig,
    register_sae_config,
    register_sae_model,
)

__all__ = [
    "SparseDictionary",
    "SparseDictionaryConfig",
    "MatryoshkaSAEConfig",
    "MatryoshkaSparseAutoEncoder",
    "register_sae_config",
    "register_sae_model",
    "NormComputing",
    "NormConstrainable",
    "DatasetNormStandardizable",
    "EncoderInitializable",
    "ActiveSubspaceInitializable",
    "EncoderBiasInitializable",
]
