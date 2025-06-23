"""Base classes for prompt engineering."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PromptType(Enum):
    """Types of prompts."""

    COMPLETION = "completion"
    CHAT = "chat"
    INSTRUCTION = "instruction"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    SYSTEM = "system"


@dataclass
class PromptExample:
    """Example for few-shot prompting."""

    input: str
    output: str
    explanation: str | None = None


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    name: str
    template: str
    prompt_type: PromptType
    variables: list[str] = field(default_factory=list)
    examples: list[PromptExample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Extract variables from template if not provided."""
        if not self.variables:
            self.variables = self._extract_variables()

    def _extract_variables(self) -> list[str]:
        """Extract variable names from template."""
        # Find variables in format {variable_name}
        pattern = r"\{([^}]+)\}"
        variables = re.findall(pattern, self.template)
        return list(set(variables))

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        # Validate required variables
        missing_vars = set(self.variables) - set(kwargs.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        try:
            formatted = self.template.format(**kwargs)

            # Add examples if this is a few-shot prompt
            if self.prompt_type == PromptType.FEW_SHOT and self.examples:
                examples_text = self._format_examples()
                formatted = examples_text + "\n\n" + formatted

            return formatted.strip()

        except KeyError as e:
            raise ValueError(f"Template formatting error: {e}") from e

    def _format_examples(self) -> str:
        """Format examples for few-shot prompting."""
        examples_text = []

        for i, example in enumerate(self.examples):
            example_text = (
                f"Example {i + 1}:\nInput: {example.input}\nOutput: {example.output}"
            )

            if example.explanation:
                example_text += f"\nExplanation: {example.explanation}"

            examples_text.append(example_text)

        return "\n\n".join(examples_text)

    def add_example(
        self, input_text: str, output_text: str, explanation: str = None
    ) -> None:
        """Add an example to the template."""
        example = PromptExample(
            input=input_text, output=output_text, explanation=explanation
        )
        self.examples.append(example)

    def validate(self) -> bool:
        """Validate the template structure."""
        try:
            # Check if template has valid format
            test_kwargs = {var: f"test_{var}" for var in self.variables}
            self.format(**test_kwargs)
            return True
        except Exception:
            return False


class PromptEngineering:
    """Utilities for prompt engineering and optimization."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates: dict[str, PromptTemplate] = {}

    def register_template(self, template: PromptTemplate) -> None:
        """Register a prompt template."""
        self.templates[template.name] = template
        self.logger.info(f"Registered prompt template: {template.name}")

    def get_template(self, name: str) -> PromptTemplate | None:
        """Get a registered template by name."""
        return self.templates.get(name)

    def list_templates(self) -> list[str]:
        """List all registered template names."""
        return list(self.templates.keys())

    def create_few_shot_prompt(
        self,
        task_description: str,
        examples: list[PromptExample],
        input_prompt: str,
        name: str = "few_shot_template",
    ) -> PromptTemplate:
        """Create a few-shot prompt template."""

        template = f"""{task_description}

Here are some examples:

{{examples}}

Now, please complete this task:
{input_prompt}"""

        prompt_template = PromptTemplate(
            name=name,
            template=template,
            prompt_type=PromptType.FEW_SHOT,
            examples=examples,
            variables=["examples"],
        )

        return prompt_template

    def create_chain_of_thought_prompt(
        self, problem: str, thinking_steps: list[str], name: str = "cot_template"
    ) -> PromptTemplate:
        """Create a chain-of-thought prompt template."""

        steps_text = "\n".join(
            [f"{i + 1}. {step}" for i, step in enumerate(thinking_steps)]
        )

        template = f"""{problem}

Let's think about this step by step:
{steps_text}

Based on this reasoning, the answer is: {{answer}}"""

        prompt_template = PromptTemplate(
            name=name,
            template=template,
            prompt_type=PromptType.CHAIN_OF_THOUGHT,
            variables=["answer"],
        )

        return prompt_template

    def optimize_prompt_length(self, prompt: str, max_tokens: int = 2000) -> str:
        """Optimize prompt length while preserving meaning."""
        # Simple optimization strategies
        # In practice, this could be much more sophisticated

        if len(prompt.split()) <= max_tokens:
            return prompt

        # Remove redundant whitespace
        optimized = re.sub(r"\s+", " ", prompt.strip())

        # Remove repetitive phrases (simple heuristic)
        sentences = optimized.split(".")
        unique_sentences = []
        seen = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence)

        optimized = ". ".join(unique_sentences)

        # If still too long, truncate intelligently
        if len(optimized.split()) > max_tokens:
            words = optimized.split()
            optimized = " ".join(words[:max_tokens])

            # Try to end at a complete sentence
            last_period = optimized.rfind(".")
            if last_period > len(optimized) * 0.8:  # If within last 20%
                optimized = optimized[: last_period + 1]

        return optimized

    def extract_prompt_components(self, prompt: str) -> dict[str, str]:
        """Extract components from a prompt (system, user, examples, etc.)."""
        components = {}

        # Look for common patterns
        patterns = {
            "system": r"(?i)system:?\s*(.*?)(?=user:|assistant:|$)",
            "user": r"(?i)user:?\s*(.*?)(?=system:|assistant:|$)",
            "assistant": r"(?i)assistant:?\s*(.*?)(?=system:|user:|$)",
            "examples": r"(?i)example\s*\d*:?\s*(.*?)(?=example|$)",
            "instruction": r"(?i)instruction:?\s*(.*?)(?=example|user:|$)",
        }

        for component, pattern in patterns.items():
            matches = re.findall(pattern, prompt, re.DOTALL)
            if matches:
                components[component] = [match.strip() for match in matches]

        return components

    def measure_prompt_quality(self, prompt: str) -> dict[str, Any]:
        """Measure various quality metrics for a prompt."""
        metrics = {}

        # Basic metrics
        metrics["length_chars"] = len(prompt)
        metrics["length_words"] = len(prompt.split())
        metrics["length_tokens_estimate"] = len(prompt) // 4  # Rough estimate

        # Complexity metrics
        sentences = prompt.split(".")
        metrics["sentence_count"] = len([s for s in sentences if s.strip()])
        metrics["avg_sentence_length"] = metrics["length_words"] / max(
            metrics["sentence_count"], 1
        )

        # Clarity metrics
        metrics["readability_score"] = self._calculate_readability(prompt)
        metrics["specificity_score"] = self._calculate_specificity(prompt)
        metrics["clarity_score"] = (
            metrics["readability_score"] + metrics["specificity_score"]
        ) / 2

        return metrics

    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified)."""
        words = text.split()
        sentences = [s for s in text.split(".") if s.strip()]

        if not sentences:
            return 0.0

        avg_words_per_sentence = len(words) / len(sentences)

        # Simple readability heuristic
        # Lower average words per sentence = higher readability
        readability = max(0.0, min(1.0, 1.0 - (avg_words_per_sentence - 10) / 40))

        return readability

    def _calculate_specificity(self, text: str) -> float:
        """Calculate how specific/detailed the prompt is."""
        # Heuristics for specificity
        specific_words = [
            "exactly",
            "specifically",
            "precisely",
            "detailed",
            "step-by-step",
        ]
        question_words = ["what", "how", "why", "when", "where", "which"]

        text_lower = text.lower()

        specific_count = sum(1 for word in specific_words if word in text_lower)
        question_count = sum(1 for word in question_words if word in text_lower)

        # Normalize by text length
        specificity = (specific_count + question_count) / max(len(text.split()) / 10, 1)

        return min(1.0, specificity)
