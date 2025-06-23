"""Prompt optimization utilities."""

import logging
import random
from dataclasses import dataclass
from typing import Any

from ..llm import LLMProvider


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    original_prompt: str
    optimized_prompt: str
    performance_improvement: float
    metrics: dict[str, Any]
    optimization_strategy: str


class PromptOptimizer:
    """Optimize prompts for better performance."""

    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.logger = logging.getLogger(__name__)

    async def optimize_prompt(
        self,
        prompt: str,
        test_cases: list[dict[str, str]],
        optimization_strategy: str = "iterative",
    ) -> OptimizationResult:
        """
        Optimize a prompt based on test cases.

        Args:
            prompt: Original prompt to optimize
            test_cases: List of {"input": str, "expected": str} test cases
            optimization_strategy: Strategy to use for optimization
        """
        original_performance = await self._evaluate_prompt(prompt, test_cases)

        if optimization_strategy == "iterative":
            optimized_prompt = await self._iterative_optimization(prompt, test_cases)
        elif optimization_strategy == "genetic":
            optimized_prompt = await self._genetic_optimization(prompt, test_cases)
        elif optimization_strategy == "template_based":
            optimized_prompt = await self._template_based_optimization(
                prompt, test_cases
            )
        else:
            raise ValueError(f"Unknown optimization strategy: {optimization_strategy}")

        optimized_performance = await self._evaluate_prompt(
            optimized_prompt, test_cases
        )

        improvement = optimized_performance["score"] - original_performance["score"]

        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized_prompt,
            performance_improvement=improvement,
            metrics={
                "original_performance": original_performance,
                "optimized_performance": optimized_performance,
            },
            optimization_strategy=optimization_strategy,
        )

    async def _evaluate_prompt(
        self, prompt: str, test_cases: list[dict[str, str]]
    ) -> dict[str, Any]:
        """Evaluate prompt performance on test cases."""
        scores = []

        for test_case in test_cases:
            formatted_prompt = (
                prompt.format(**test_case["input"])
                if isinstance(test_case["input"], dict)
                else prompt + "\n" + test_case["input"]
            )

            try:
                response = await self.llm_provider.generate(formatted_prompt)
                score = self._calculate_similarity(response, test_case["expected"])
                scores.append(score)
            except Exception as e:
                self.logger.warning(f"Error evaluating test case: {e}")
                scores.append(0.0)

        return {
            "score": sum(scores) / len(scores) if scores else 0.0,
            "individual_scores": scores,
            "test_case_count": len(test_cases),
        }

    def _calculate_similarity(self, response: str, expected: str) -> float:
        """Calculate similarity between response and expected output."""
        # Simple similarity based on word overlap
        # In practice, you might use more sophisticated metrics like BLEU, ROUGE, or semantic similarity

        response_words = set(response.lower().split())
        expected_words = set(expected.lower().split())

        if not expected_words:
            return 1.0 if not response_words else 0.0

        intersection = response_words & expected_words
        union = response_words | expected_words

        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0

        # Length penalty (penalize responses that are too short or too long)
        length_ratio = min(len(response), len(expected)) / max(
            len(response), len(expected), 1
        )

        return jaccard * length_ratio

    async def _iterative_optimization(
        self, prompt: str, test_cases: list[dict[str, str]]
    ) -> str:
        """Optimize prompt using iterative refinement."""
        current_prompt = prompt
        best_prompt = prompt
        best_score = (await self._evaluate_prompt(prompt, test_cases))["score"]

        optimization_strategies = [
            self._add_clarity_instructions,
            self._add_examples,
            self._refine_formatting,
            self._add_constraints,
            self._simplify_language,
        ]

        for i in range(5):  # Max 5 iterations
            for strategy in optimization_strategies:
                candidate_prompt = strategy(current_prompt)
                candidate_score = (
                    await self._evaluate_prompt(candidate_prompt, test_cases)
                )["score"]

                if candidate_score > best_score:
                    best_prompt = candidate_prompt
                    best_score = candidate_score
                    current_prompt = candidate_prompt
                    self.logger.info(
                        f"Iteration {i + 1}: Improved score to {best_score:.3f}"
                    )
                    break
            else:
                # No improvement found
                break

        return best_prompt

    async def _genetic_optimization(
        self, prompt: str, test_cases: list[dict[str, str]]
    ) -> str:
        """Optimize prompt using genetic algorithm approach."""
        # Generate initial population of prompt variations
        population = [prompt]

        # Generate variations
        for _ in range(9):  # Population size of 10
            variation = self._mutate_prompt(prompt)
            population.append(variation)

        # Evolve for several generations
        for generation in range(3):
            # Evaluate all prompts in population
            scored_population = []
            for p in population:
                score = (await self._evaluate_prompt(p, test_cases))["score"]
                scored_population.append((p, score))

            # Sort by score
            scored_population.sort(key=lambda x: x[1], reverse=True)

            # Keep top 50%
            survivors = [p for p, _ in scored_population[:5]]

            # Generate new population from survivors
            new_population = survivors.copy()
            while len(new_population) < 10:
                parent = random.choice(survivors)
                child = self._mutate_prompt(parent)
                new_population.append(child)

            population = new_population

            best_score = scored_population[0][1]
            self.logger.info(
                f"Generation {generation + 1}: Best score {best_score:.3f}"
            )

        # Return best prompt
        final_scores = []
        for p in population:
            score = (await self._evaluate_prompt(p, test_cases))["score"]
            final_scores.append((p, score))

        return max(final_scores, key=lambda x: x[1])[0]

    async def _template_based_optimization(
        self, prompt: str, test_cases: list[dict[str, str]]
    ) -> str:
        """Optimize using predefined template patterns."""
        templates = [
            "Please {task}. Be specific and detailed in your response.",
            "Complete the following task step by step: {task}",
            "You are an expert. {task}. Provide a comprehensive answer.",
            "Task: {task}\n\nRequirements:\n- Be accurate\n- Be concise\n- Be helpful",
            "{task}\n\nPlease think carefully and provide your best answer.",
        ]

        best_prompt = prompt
        best_score = (await self._evaluate_prompt(prompt, test_cases))["score"]

        # Extract task from original prompt (simplified)
        task = prompt.strip()

        for template in templates:
            candidate_prompt = template.format(task=task)
            candidate_score = (
                await self._evaluate_prompt(candidate_prompt, test_cases)
            )["score"]

            if candidate_score > best_score:
                best_prompt = candidate_prompt
                best_score = candidate_score

        return best_prompt

    def _mutate_prompt(self, prompt: str) -> str:
        """Generate a mutation of the prompt."""
        mutations = [
            lambda p: self._add_clarity_instructions(p),
            lambda p: self._refine_formatting(p),
            lambda p: self._add_constraints(p),
            lambda p: self._simplify_language(p),
            lambda p: self._add_context(p),
        ]

        mutation = random.choice(mutations)
        return mutation(prompt)

    def _add_clarity_instructions(self, prompt: str) -> str:
        """Add clarity instructions to prompt."""
        clarity_phrases = [
            "Please be specific and detailed.",
            "Provide a clear and concise answer.",
            "Think step by step.",
            "Be precise in your response.",
            "Ensure your answer is accurate and helpful.",
        ]

        addition = random.choice(clarity_phrases)
        return f"{prompt}\n\n{addition}"

    def _add_examples(self, prompt: str) -> str:
        """Add example format to prompt."""
        return f"{prompt}\n\nExample format:\nInput: [example input]\nOutput: [example output]"

    def _refine_formatting(self, prompt: str) -> str:
        """Improve prompt formatting."""
        # Add structure
        if ":" not in prompt:
            return f"Task: {prompt}\n\nResponse:"
        return prompt

    def _add_constraints(self, prompt: str) -> str:
        """Add helpful constraints to prompt."""
        constraints = [
            "Keep your response under 200 words.",
            "Use bullet points if appropriate.",
            "Provide evidence for your claims.",
            "Consider multiple perspectives.",
            "Be objective and unbiased.",
        ]

        constraint = random.choice(constraints)
        return f"{prompt}\n\nConstraint: {constraint}"

    def _simplify_language(self, prompt: str) -> str:
        """Simplify language in prompt."""
        # Basic simplification (could be much more sophisticated)
        simplified = prompt.replace("utilize", "use")
        simplified = simplified.replace("commence", "start")
        simplified = simplified.replace("demonstrate", "show")
        simplified = simplified.replace("consequently", "so")

        return simplified

    def _add_context(self, prompt: str) -> str:
        """Add helpful context to prompt."""
        contexts = [
            "You are a helpful assistant.",
            "You are an expert in this field.",
            "Please consider the user's perspective.",
            "Focus on practical applications.",
            "Think about real-world implications.",
        ]

        context = random.choice(contexts)
        return f"{context} {prompt}"
