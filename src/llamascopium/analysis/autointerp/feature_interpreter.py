"""Auto-interpretation functionality for SAE features.

This module provides tools for automatically interpreting and evaluating sparse autoencoder features
based on the EleutherAI auto-interp approach (https://blog.eleuther.ai/autointerp/).

It includes:
1. Methods for prompting LLMs to generate explanations for features
2. Methods for evaluating explanations via different techniques:
   - Detection: Having LLMs identify if examples contain a feature
   - Fuzzing: Having LLMs identify correctly marked activating tokens
"""

import asyncio
import random
import time
import traceback
from typing import Any, AsyncGenerator, Callable, Literal, Optional

import json_repair
import numpy as np
import torch
from datasets import Dataset
from pydantic import BaseModel

from llamascopium.analysis.samples import Segment, TokenizedSample
from llamascopium.backend.language_model import LanguageModel
from llamascopium.database import FeatureAnalysis, FeatureRecord, MongoClient
from llamascopium.utils.logging import get_logger

from .autointerp_base import (
    AutoInterpConfig,
    ExplainerType,
    ScorerType,
)
from .evaluation_prompts import (
    generate_detection_prompt,
    generate_fuzzing_prompt,
)
from .explanation_prompts import (
    generate_explanation_prompt,
    generate_explanation_prompt_neuronpedia,
)

logger = get_logger("analysis.feature_interpreter")


class Step(BaseModel):
    """A step in the chain-of-thought process."""

    thought: str
    """The thought of the step."""

    # output: str
    # """The output of the step."""


Step_Schema = {
    "type": "object",
    "properties": {
        "thought": {"type": "string", "description": "The thought of the step."},
    },
}


class AutoInterpExplanation(BaseModel):
    """The result of an auto-interpretation of a SAE feature."""

    steps: list[Step]
    """The steps of the chain-of-thought process."""

    final_explanation: str
    """The explanation of the feature."""

    activation_consistency: Literal[1, 2, 3, 4, 5]
    """The consistency of the feature."""

    complexity: Literal[1, 2, 3, 4, 5]
    """The complexity of the feature."""


AutoInterpExplanation_Schema = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": Step_Schema},
        "final_explanation": {
            "type": "string",
            "description": "The explanation of the feature, in the form of 'This feature activates on... '",
        },
        "activation_consistency": {
            "type": "integer",
            "description": "The consistency of the feature, on a scale of 1 to 5.",
        },
        "complexity": {"type": "integer", "description": "The complexity of the feature, on a scale of 1 to 5."},
    },
}


class AutoInterpEvaluation(BaseModel):
    """The result of an auto-interpretation of a SAE feature."""

    steps: list[Step]
    """The steps of the chain-of-thought process."""

    evaluation_results: list[bool]
    """The evaluation results for each example. Should be a list of YES/NO values."""


AutoInterpEvaluation_Schema = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": Step_Schema},
        "evaluation_results": {
            "type": "array",
            "items": {"type": "boolean"},
            "description": "The evaluation results for each example. Should be a list of True/False values.",
        },
    },
}


def generate_activating_examples(
    feature: FeatureRecord,
    model: LanguageModel,
    datasets: Callable[[str, int, int], Dataset],
    analysis: FeatureAnalysis,
    n: int = 10,
    max_length: int = 50,
) -> list[TokenizedSample]:
    """Generate examples where a feature strongly activates using database records.

    Args:
        feature: FeatureRecord to analyze
        model: Language model to use
        datasets: Callable to fetch datasets
        analysis: FeatureAnalysis to use
        n: Maximum number of examples to generate
        max_length: Maximum length of examples to generate

    Returns:
        List of TokenizedExample with high activation for the feature
    """
    samples: list[TokenizedSample] = []

    # Get examples from top activations
    sampling = analysis.samplings[0]
    feature_acts_ = torch.sparse_coo_tensor(
        torch.tensor(sampling.feature_acts_indices),
        torch.tensor(sampling.feature_acts_values),
        (int(np.max(sampling.feature_acts_indices[0])) + 1, 2048),
    )
    feature_acts_ = feature_acts_.to_dense()

    # Lorsa z pattern data so we can explain which tokens are contributing to the activation
    # We want to operate in coo format since zpattern is 3-d
    # z_pattern_indices: [n_samples, n_ctx, n_ctx]
    # z_pattern_values: [n_samples, n_ctx, n_ctx]
    if sampling.z_pattern_indices is not None:
        assert sampling.z_pattern_values is not None, "Z pattern values are not available"
        z_pattern_indices = torch.tensor(sampling.z_pattern_indices).int()
        z_pattern_values = torch.tensor(sampling.z_pattern_values)
    else:
        z_pattern_indices = None
        z_pattern_values = None

    for i, (dataset_name, shard_idx, n_shards, context_idx, feature_acts) in enumerate(
        zip(
            sampling.dataset_name,
            sampling.shard_idx if sampling.shard_idx is not None else [0] * len(sampling.dataset_name),
            sampling.n_shards if sampling.n_shards is not None else [1] * len(sampling.dataset_name),
            sampling.context_idx,
            feature_acts_,
        )
    ):
        dataset = datasets(dataset_name, shard_idx, n_shards)
        data = dataset[int(context_idx)]

        # Process the sample using model's trace method
        try:
            origins = model.trace({k: [v] for k, v in data.items()})[0]
        except Exception:
            continue

        max_act_pos = torch.argmax(feature_acts).item()

        left_end = max(0, max_act_pos - max_length // 2)
        right_end = min(len(origins), max_act_pos + max_length // 2)

        # Create TokenizedExample using the trace information
        sample = TokenizedSample.construct(
            text=data["text"],
            activations=feature_acts[left_end:right_end],
            origins=origins[left_end:right_end],
            max_activation=analysis.max_feature_acts,
        )
        # Find max contributing previous tokens for Lorsa.
        # We want to operate in coo format since zpattern is 3-d.
        if z_pattern_indices is not None and z_pattern_values is not None:
            current_sequence_mask = z_pattern_indices[0] == i
            current_z_pattern_indices = z_pattern_indices[1:, current_sequence_mask]
            current_z_pattern_values = z_pattern_values[current_sequence_mask]
            # Need to adjust indices since the text has been cropped
            # and remove negative indices
            out_of_right_end_indices = current_z_pattern_indices.lt(right_end).all(dim=0)
            current_z_pattern_indices -= left_end
            z_pattern_with_negative_indices = current_z_pattern_indices.ge(0).all(dim=0)
            mask = out_of_right_end_indices * z_pattern_with_negative_indices
            sample.add_z_pattern_data(
                current_z_pattern_indices[:, mask],
                current_z_pattern_values[mask],
                origins,
            )

        samples.append(sample)

        if len(samples) >= n:
            break

    return samples


def generate_non_activating_examples(
    feature: FeatureRecord,
    model: LanguageModel,
    datasets: Callable[[str, int, int], Dataset],
    analysis: FeatureAnalysis,
    n: int = 10,
    max_length: int = 50,
) -> list[TokenizedSample]:
    """Generate examples where a feature doesn't activate much.

    Args:
        feature: FeatureRecord to analyze
        model: Language model to use
        datasets: Callable to fetch datasets
        analysis: FeatureAnalysis to use
        n: Maximum number of examples to generate
        max_length: Maximum length of examples to generate

    Returns:
        List of TokenizedExample with low activation for the feature
    """

    samples: list[TokenizedSample] = []
    if n == 0:
        return samples
    error_prefix = f"Error processing non-activating examples of feature {feature.index}:"

    sampling_idx = -1
    for i in range(len(analysis.samplings)):
        if analysis.samplings[i].name == "non_activating":
            sampling_idx = i
            break
    if sampling_idx == -1:
        return samples
    sampling = analysis.samplings[sampling_idx]

    assert sampling.name == "non_activating", f"{error_prefix} Sampling {sampling.name} is not non_activating"
    for i, (dataset_name, shard_idx, n_shards, context_idx, feature_acts_indices, feature_acts_values) in enumerate(
        zip(
            sampling.dataset_name,
            sampling.shard_idx if sampling.shard_idx else [0] * len(sampling.dataset_name),
            sampling.n_shards if sampling.n_shards else [1] * len(sampling.dataset_name),
            sampling.context_idx,
            # sampling.feature_acts,
            sampling.feature_acts_indices,
            sampling.feature_acts_values,
        )
    ):
        try:
            feature_acts = torch.sparse_coo_tensor(
                torch.Tensor(feature_acts_indices),
                torch.Tensor(feature_acts_values),
                (1024, sampling.context_idx.shape[0]),
            )
            feature_acts = feature_acts.to_dense()

            dataset = datasets(dataset_name, shard_idx, n_shards)
            data = dataset[context_idx]

            # Process the sample using model's trace method
            # lock.acquire()
            origins = model.trace({k: [v] for k, v in data.items()})[0]
            # lock.release()

            # Create TokenizedExample using the trace information
            sample = TokenizedSample.construct(
                text=data["text"],
                activations=feature_acts[:max_length],
                origins=origins[:max_length],
                max_activation=analysis.max_feature_acts,
            )

            samples.append(sample)

        except Exception as e:
            logger.error(f"{error_prefix} {e}")
            continue

        if len(samples) >= n:
            break

    return samples


class FeatureInterpreter:
    """A class for generating and evaluating explanations for SAE features."""

    def __init__(self, cfg: AutoInterpConfig, mongo_client: MongoClient):
        """Initialize the feature interpreter.

        Args:
            cfg: Configuration for interpreter
            mongo_client: Optional MongoDB client for retrieving data
        """
        self.cfg = cfg
        self.mongo_client = mongo_client
        # Set up LLM client for explanation generation
        self._setup_llm_clients()

    def _setup_llm_clients(self):
        """Set up async OpenAI client for explanation generation and evaluation."""
        try:
            import httpx
            from openai import AsyncOpenAI

            # Set up async HTTP client with proxy if needed
            http_client = None
            if self.cfg.openai_proxy:
                http_client = httpx.AsyncClient(
                    proxy=self.cfg.openai_proxy,
                    transport=httpx.AsyncHTTPTransport(local_address="0.0.0.0"),
                )

            self.explainer_client = AsyncOpenAI(
                base_url=self.cfg.openai_base_url,
                api_key=self.cfg.openai_api_key,
                http_client=http_client,
            )
        except ImportError:
            raise ImportError("OpenAI package not installed. Please install it with `uv add openai`.")

    def get_feature_examples(
        self,
        feature: FeatureRecord,
        model: LanguageModel,
        datasets: Callable[[str, int, int], Dataset],
        analysis_name: str = "default",
        max_length: int = 50,
    ) -> tuple[list[TokenizedSample], list[TokenizedSample]]:
        """Get activating and non-activating examples for a feature."""
        analysis = next((a for a in feature.analyses if a.name == analysis_name), None)
        if not analysis:
            raise ValueError(f"Analysis {analysis_name} not found for feature {feature.index}")

        if analysis.max_feature_acts == 0:
            raise ValueError(f"Feature {feature.index} has no activation. Skipping interpretation.")

        # Get examples from each sampling
        activating_examples = generate_activating_examples(
            feature=feature,
            model=model,
            datasets=datasets,
            analysis=analysis,
            n=self.cfg.n_activating_examples,
            max_length=max_length,
        )
        non_activating_examples = generate_non_activating_examples(
            feature=feature,
            model=model,
            datasets=datasets,
            analysis=analysis,
            n=self.cfg.n_non_activating_examples,
            max_length=max_length,
        )
        return activating_examples, non_activating_examples

    async def generate_explanation(
        self, activating_examples: list[TokenizedSample], top_logits: dict[str, list[dict[str, Any]]] | None = None
    ) -> dict[str, Any]:
        """Generate an explanation for a feature based on activating examples.

        Args:
            activating_examples: List of examples where the feature activates
            top_positive_logits: Top positive logits for the feature
        Returns:
            Dictionary with explanation and metadata
        """
        if self.cfg.explainer_type is ExplainerType.OPENAI:
            system_prompt, user_prompt = generate_explanation_prompt(self.cfg, activating_examples)
        else:
            system_prompt, user_prompt = generate_explanation_prompt_neuronpedia(
                self.cfg, activating_examples, top_logits
            )
        start_time = time.time()

        if self.cfg.explainer_type is ExplainerType.OPENAI:
            response = await self.explainer_client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            assert response.choices[0].message.content is not None, (
                f"No explanation returned from OpenAI\n\nsystem_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}\n\nresponse: {response}"
            )
            explanation = json_repair.loads(response.choices[0].message.content)
        else:
            response = await self.explainer_client.chat.completions.create(
                model=self.cfg.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )

            explanation = {
                "final_explanation": response.choices[0].message.content,
                "activation_consistency": 5,
                "complexity": 5,
            }
        response_time = time.time() - start_time
        return {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
            "response": explanation,
            "time": response_time,
        }

    async def evaluate_explanation_detection(
        self,
        explanation: dict[str, Any],
        activating_examples: list[TokenizedSample],
        non_activating_examples: list[TokenizedSample],
    ) -> dict[str, Any]:
        """Evaluate an explanation using the detection method.

        Args:
            explanation: The explanation to evaluate
            activating_examples: Examples where the feature activates
            non_activating_examples: Examples where the feature doesn't activate

        Returns:
            Dictionary with evaluation results
        """
        # Select a subset of examples
        n_activating = min(self.cfg.detection_n_examples, len(activating_examples))
        n_non_activating = min(self.cfg.detection_n_examples, len(non_activating_examples))

        test_activating = random.sample(activating_examples, n_activating) if n_activating > 0 else []
        test_non_activating = random.sample(non_activating_examples, n_non_activating) if n_non_activating > 0 else []

        # Mix and shuffle examples
        all_examples = test_activating + test_non_activating
        if len(all_examples) < self.cfg.detection_n_examples:
            return {
                "method": "detection",
                "prompt": "",
                "response": "",
                "ground_truth": [],
                "predictions": [],
                "metrics": {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "balanced_accuracy": 0,
                },
                "passed": False,
                "time": 0,
            }

        random.shuffle(all_examples)

        # Ground truth for each example (1 for activating, 0 for non-activating)
        ground_truth = [1 if ex in test_activating else 0 for ex in all_examples]

        # Generate prompt
        system_prompt, user_prompt = generate_detection_prompt(self.cfg, explanation, all_examples)

        # Get response from OpenAI
        start_time = time.time()
        response = await self.explainer_client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        assert response.choices[0].message.content is not None, (
            f"No detection response returned from OpenAI\n\nsystem_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}\n\nresponse: {response}"
        )
        detection_response: dict[str, Any] = json_repair.loads(response.choices[0].message.content)  # type: ignore
        predictions: list[bool] = detection_response["evaluation_results"]
        response_time = time.time() - start_time

        # Pad predictions if needed
        predictions = predictions[: len(ground_truth)]
        if len(predictions) < len(ground_truth):
            predictions.extend([False] * (len(ground_truth) - len(predictions)))

        # Calculate metrics
        tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 1)
        tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 0)
        fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 0 and pred == 1)
        fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt == 1 and pred == 0)

        accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        balanced_accuracy = ((tp / (tp + fn) if (tp + fn) > 0 else 0) + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2

        return {
            "method": "detection",
            "prompt": system_prompt + "\n\n" + user_prompt,
            "response": detection_response,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
            },
            "passed": balanced_accuracy >= 0.7,  # Arbitrary threshold for passing
            "time": response_time,
        }

    def _create_incorrectly_marked_example(self, sample: TokenizedSample) -> TokenizedSample:
        """Create an incorrectly marked version of an example.

        Args:
            sample: The original sample

        Returns:
            A copy of the sample with incorrect highlighting
        """
        # Count how many tokens would be highlighted in the correct example
        threshold = self.cfg.activation_threshold
        n_highlighted = sum(1 for seg in sample.segments if seg.activation > threshold * sample.max_activation)

        def highlight_random_tokens(sample: TokenizedSample, n_highlighted: int) -> TokenizedSample:
            non_activating_indices = [
                i for i, seg in enumerate(sample.segments) if seg.activation < threshold * sample.max_activation
            ]
            highlight_indices = random.sample(non_activating_indices, min(n_highlighted, len(non_activating_indices)))
            segments = [
                Segment(seg.text, sample.max_activation if i in highlight_indices else 0)
                for i, seg in enumerate(sample.segments)
            ]
            return TokenizedSample(segments, sample.max_activation)

        n_to_highlight = max(3, n_highlighted)  # Highlight at least 3 tokens
        return highlight_random_tokens(sample, n_to_highlight)

    async def evaluate_explanation_fuzzing(
        self, explanation: dict[str, Any], activating_examples: list[TokenizedSample]
    ) -> dict[str, Any]:
        """Evaluate an explanation using the fuzzing method.

        Args:
            explanation: The explanation to evaluate
            activating_examples: Examples where the feature activates

        Returns:
            Dictionary with evaluation results
        """
        if len(activating_examples) < self.cfg.fuzzing_n_examples:
            # Not enough examples, return empty result
            return {
                "method": "fuzzing",
                "prompt": "",
                "response": "",
                "ground_truth": [],
                "predictions": [],
                "metrics": {
                    "accuracy": 0,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "balanced_accuracy": 0,
                },
                "passed": False,
                "time": 0,
            }

        # Prepare examples:
        # - Correctly marked examples (original)
        # - Incorrectly marked examples (with wrong parts highlighted)
        n_correct = self.cfg.fuzzing_decile_correct
        n_incorrect = self.cfg.fuzzing_decile_incorrect

        # Get a sample of activating examples
        sample_examples = random.sample(activating_examples, min(n_correct + n_incorrect, len(activating_examples)))

        # Split into correct and incorrect
        correct_examples = sample_examples[:n_correct]
        incorrect_candidates = sample_examples[n_correct:]

        # Create incorrectly marked versions
        incorrect_examples = [self._create_incorrectly_marked_example(ex) for ex in incorrect_candidates]

        # Combine and mark with ground truth
        examples_with_labels = [(ex, True) for ex in correct_examples] + [(ex, False) for ex in incorrect_examples]

        # Shuffle
        random.shuffle(examples_with_labels)

        # Generate prompt
        system_prompt, user_prompt = generate_fuzzing_prompt(self.cfg, explanation, examples_with_labels)

        # Get response from OpenAI
        start_time = time.time()
        response = await self.explainer_client.chat.completions.create(
            model=self.cfg.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        assert response.choices[0].message.content is not None, (
            f"No fuzzing response returned from OpenAI\n\nsystem_prompt: {system_prompt}\n\nuser_prompt: {user_prompt}\n\nresponse: {response}"
        )
        fuzzing_response: dict[str, Any] = json_repair.loads(response.choices[0].message.content)  # type: ignore
        predictions: list[bool] = fuzzing_response["evaluation_results"]
        response_time = time.time() - start_time
        # Pad predictions if needed
        predictions = predictions[: len(examples_with_labels)]
        if len(predictions) < len(examples_with_labels):
            predictions.extend([False] * (len(examples_with_labels) - len(predictions)))

        # Extract ground truth
        ground_truth = [is_correct for _, is_correct in examples_with_labels]

        # Calculate metrics
        tp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is True and pred is True)
        tn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is False and pred is False)
        fp = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is False and pred is True)
        fn = sum(1 for gt, pred in zip(ground_truth, predictions) if gt is True and pred is False)

        accuracy = (tp + tn) / len(ground_truth) if ground_truth else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        balanced_accuracy = ((tp / (tp + fn) if (tp + fn) > 0 else 0) + (tn / (tn + fp) if (tn + fp) > 0 else 0)) / 2

        return {
            "method": "fuzzing",
            "prompt": user_prompt,
            "response": fuzzing_response,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": balanced_accuracy,
            },
            "passed": balanced_accuracy >= 0.7,  # Arbitrary threshold for passing
            "time": response_time,
        }

    async def interpret_single_feature(
        self,
        activating_examples: list[TokenizedSample],
        non_activating_examples: list[TokenizedSample],
        top_logits: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        start_time = time.time()
        response_time = 0

        # Generate explanation for the feature
        explanation_result = await self.generate_explanation(activating_examples, top_logits)
        explanation: dict[str, Any] = explanation_result["response"]
        response_time += explanation_result["time"]
        # Evaluate explanation
        evaluation_results = []

        if ScorerType.DETECTION in self.cfg.scorer_type:
            detection_result = await self.evaluate_explanation_detection(
                explanation, activating_examples, non_activating_examples
            )
            evaluation_results.append(detection_result)
            response_time += detection_result["time"]

        if ScorerType.FUZZING in self.cfg.scorer_type:
            fuzzing_result = await self.evaluate_explanation_fuzzing(explanation, activating_examples)
            evaluation_results.append(fuzzing_result)
            response_time += fuzzing_result["time"]

        total_time = time.time() - start_time

        return {
            "explanation": explanation["final_explanation"],
            "complexity": explanation["complexity"],
            "consistency": explanation["activation_consistency"],
            "explanation_details": {
                k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in explanation_result.items()
            },
            "evaluations": [
                {k: v.model_dump() if isinstance(v, BaseModel) else v for k, v in eval_result.items()}
                for eval_result in evaluation_results
            ],
            "passed": any(eval_result["passed"] for eval_result in evaluation_results),
            "time": {
                "total": total_time,
                "response": response_time,
            },
        }

    async def interpret_features(
        self,
        sae_name: str,
        sae_series: str,
        model: LanguageModel,
        datasets: Callable[[str, int, int], Dataset],
        analysis_name: str = "default",
        feature_indices: Optional[list[int]] = None,
        max_concurrent: int = 10,
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate and evaluate explanations for multiple features with async concurrency.

        Args:
            sae_name: Name of the SAE
            sae_series: Series of the SAE
            model: Language model to use for generating activations
            datasets: Callable to fetch datasets
            analysis_name: Name of the analysis to use
            feature_indices: Optional list of specific feature indices to interpret. If None, interprets all features.
            max_concurrent: Maximum number of concurrent API requests
            progress_callback: Optional callback function(completed, total, current_feature_index) for progress updates

        Yields:
            Dictionary with interpretation results for each feature
        """
        if feature_indices is None:
            sae_record = self.mongo_client.get_sae(sae_name, sae_series)
            assert sae_record is not None, f"SAE {sae_name} {sae_series} not found"
            feature_indices = list(range(sae_record.cfg.d_sae))

        total_features = len(feature_indices)
        completed = 0
        skipped = 0
        failed = 0

        logger.info(f"Starting interpretation of {total_features} features (max concurrent: {max_concurrent})")

        # Create semaphore to limit concurrent API requests
        semaphore = asyncio.Semaphore(max_concurrent)

        async def interpret_with_semaphore(feature_index: int) -> tuple[Optional[dict[str, Any]], int, bool, bool]:
            """Interpret a single feature with semaphore control.

            Returns:
                Tuple of (result, feature_index, was_skipped, was_error)
            """
            async with semaphore:
                feature = self.mongo_client.get_feature(sae_name, sae_series, feature_index)
                try:
                    analysis = (
                        next((a for a in feature.analyses if a.name == analysis_name), None)
                        if feature is not None
                        else None
                    )
                    if (
                        feature is not None
                        and analysis is not None
                        and (
                            self.cfg.overwrite_existing
                            or feature.interpretation is None
                            or feature.interpretation.get("text") is None
                        )
                        and analysis.act_times > 0
                    ):
                        activating_examples, non_activating_examples = self.get_feature_examples(
                            feature=feature,
                            model=model,
                            datasets=datasets,
                            analysis_name=analysis_name,
                            max_length=self.cfg.max_length,
                        )
                        result = await self.interpret_single_feature(
                            activating_examples=activating_examples,
                            non_activating_examples=non_activating_examples,
                            top_logits=feature.logits,
                        )
                        return (
                            {
                                "feature_index": feature.index,
                                "sae_name": sae_name,
                                "sae_series": sae_series,
                            }
                            | result,
                            feature_index,
                            False,
                            False,
                        )
                    else:
                        # Feature already has interpretation or doesn't exist
                        return None, feature_index, True, False
                except Exception as e:
                    logger.error(f"Error interpreting feature {feature_index}:\n{e}\n{traceback.format_exc()}")
                    return None, feature_index, False, True

        # Process features concurrently
        tasks = [interpret_with_semaphore(feature_index) for feature_index in feature_indices]

        # Yield results as they complete
        for coro in asyncio.as_completed(tasks):
            result, feature_index, was_skipped, was_error = await coro

            if was_skipped:
                skipped += 1
                logger.debug(f"Feature {feature_index} skipped (already has interpretation)")
            elif was_error:
                failed += 1
            elif result is not None:
                completed += 1
                logger.info(
                    f"Completed feature {feature_index} ({completed}/{total_features} completed, "
                    f"{skipped} skipped, {failed} failed)"
                )
                yield result
            else:
                skipped += 1

            # Calculate total processed (completed + skipped + failed)
            total_processed = completed + skipped + failed

            # Call progress callback if provided
            if progress_callback is not None:
                progress_callback(total_processed, total_features, feature_index)

        logger.info(
            f"Interpretation complete: {completed} completed, {skipped} skipped, {failed} failed out of {total_features} total"
        )
