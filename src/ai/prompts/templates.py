"""Pre-built prompt templates library."""

from .base import PromptExample, PromptTemplate, PromptType


class PromptLibrary:
    """Library of pre-built prompt templates for common tasks."""

    @staticmethod
    def get_all_templates() -> dict[str, PromptTemplate]:
        """Get all available templates."""
        templates = {}

        # Add all template methods
        for method_name in dir(PromptLibrary):
            if method_name.startswith("_template_"):
                template_name = method_name[10:]  # Remove '_template_' prefix
                method = getattr(PromptLibrary, method_name)
                templates[template_name] = method()

        return templates

    @staticmethod
    def _template_summarization() -> PromptTemplate:
        """Text summarization template."""
        return PromptTemplate(
            name="summarization",
            template="""Please provide a concise summary of the following text. Focus on the main points and key information.

Text to summarize:
{text}

Summary:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["text"],
            metadata={
                "description": "Summarize text content",
                "use_case": "Document processing, content curation",
            },
        )

    @staticmethod
    def _template_question_answering() -> PromptTemplate:
        """Question answering template."""
        return PromptTemplate(
            name="question_answering",
            template="""Based on the provided context, please answer the question. If the answer cannot be found in the context, say "I cannot answer this question based on the provided context."

Context:
{context}

Question: {question}

Answer:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["context", "question"],
            metadata={
                "description": "Answer questions based on context",
                "use_case": "RAG systems, knowledge bases",
            },
        )

    @staticmethod
    def _template_code_generation() -> PromptTemplate:
        """Code generation template."""
        return PromptTemplate(
            name="code_generation",
            template="""Write {language} code that accomplishes the following task:

Task: {task}

Requirements:
{requirements}

Please provide clean, well-commented code with proper error handling.

Code:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["language", "task", "requirements"],
            metadata={
                "description": "Generate code in specified language",
                "use_case": "Code assistants, automation",
            },
        )

    @staticmethod
    def _template_code_review() -> PromptTemplate:
        """Code review template."""
        return PromptTemplate(
            name="code_review",
            template="""Please review the following {language} code and provide feedback on:
1. Code quality and style
2. Potential bugs or issues
3. Performance considerations
4. Security concerns
5. Suggestions for improvement

Code to review:
```{language}
{code}
```

Review:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["language", "code"],
            metadata={
                "description": "Review code for quality and issues",
                "use_case": "Code review automation, quality assurance",
            },
        )

    @staticmethod
    def _template_data_analysis() -> PromptTemplate:
        """Data analysis template."""
        return PromptTemplate(
            name="data_analysis",
            template="""Analyze the following dataset and provide insights:

Dataset description: {description}
Data: {data}

Please provide:
1. Key patterns or trends
2. Notable observations
3. Potential insights or actionable recommendations
4. Any data quality issues you notice

Analysis:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["description", "data"],
            metadata={
                "description": "Analyze data and provide insights",
                "use_case": "Data science, business intelligence",
            },
        )

    @staticmethod
    def _template_classification() -> PromptTemplate:
        """Classification template with examples."""
        examples = [
            PromptExample(
                input="I love this product! It works perfectly.",
                output="positive",
                explanation="Contains positive sentiment words like 'love' and 'perfectly'",
            ),
            PromptExample(
                input="This is terrible. Complete waste of money.",
                output="negative",
                explanation="Contains negative sentiment words like 'terrible' and 'waste'",
            ),
            PromptExample(
                input="The product is okay. Nothing special.",
                output="neutral",
                explanation="Neutral language without strong positive or negative indicators",
            ),
        ]

        return PromptTemplate(
            name="classification",
            template="""Classify the following text into one of these categories: {categories}

Text: {text}

Classification:""",
            prompt_type=PromptType.FEW_SHOT,
            variables=["categories", "text"],
            examples=examples,
            metadata={
                "description": "Classify text into predefined categories",
                "use_case": "Sentiment analysis, content categorization",
            },
        )

    @staticmethod
    def _template_reasoning() -> PromptTemplate:
        """Chain of thought reasoning template."""
        return PromptTemplate(
            name="reasoning",
            template="""Solve this problem step by step:

Problem: {problem}

Let me think through this systematically:

Step 1: Understand the problem
{step1}

Step 2: Identify relevant information
{step2}

Step 3: Apply logical reasoning
{step3}

Step 4: Reach conclusion
{step4}

Therefore, the answer is: {answer}""",
            prompt_type=PromptType.CHAIN_OF_THOUGHT,
            variables=["problem", "step1", "step2", "step3", "step4", "answer"],
            metadata={
                "description": "Solve problems with step-by-step reasoning",
                "use_case": "Complex problem solving, mathematical reasoning",
            },
        )

    @staticmethod
    def _template_creative_writing() -> PromptTemplate:
        """Creative writing template."""
        return PromptTemplate(
            name="creative_writing",
            template="""Write a {genre} story with the following elements:

Setting: {setting}
Main character: {character}
Conflict: {conflict}
Tone: {tone}

Requirements:
- Length: approximately {length} words
- Include dialogue
- Show, don't tell
- Create a satisfying resolution

Story:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["genre", "setting", "character", "conflict", "tone", "length"],
            metadata={
                "description": "Generate creative writing with specified parameters",
                "use_case": "Content creation, storytelling",
            },
        )

    @staticmethod
    def _template_email_draft() -> PromptTemplate:
        """Professional email drafting template."""
        return PromptTemplate(
            name="email_draft",
            template="""Draft a professional email with the following details:

To: {recipient}
Subject: {subject}
Purpose: {purpose}
Tone: {tone}
Key points to include: {key_points}

Please write a clear, professional email that:
- Has an appropriate greeting and closing
- Is well-structured and easy to read
- Includes all key points
- Maintains the specified tone

Email:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["recipient", "subject", "purpose", "tone", "key_points"],
            metadata={
                "description": "Draft professional emails",
                "use_case": "Business communication, customer service",
            },
        )

    @staticmethod
    def _template_meeting_summary() -> PromptTemplate:
        """Meeting summary template."""
        return PromptTemplate(
            name="meeting_summary",
            template="""Create a structured summary of the following meeting:

Meeting Details:
- Date: {date}
- Attendees: {attendees}
- Duration: {duration}

Meeting Notes:
{notes}

Please provide:
1. **Key Decisions Made:**
   [List important decisions]

2. **Action Items:**
   [List who needs to do what by when]

3. **Next Steps:**
   [What happens next]

4. **Follow-up Required:**
   [Any follow-up meetings or communications needed]

Summary:""",
            prompt_type=PromptType.INSTRUCTION,
            variables=["date", "attendees", "duration", "notes"],
            metadata={
                "description": "Summarize meeting content with action items",
                "use_case": "Meeting management, project coordination",
            },
        )

    @staticmethod
    def get_template(name: str) -> PromptTemplate:
        """Get a specific template by name."""
        templates = PromptLibrary.get_all_templates()
        if name not in templates:
            raise ValueError(
                f"Template '{name}' not found. Available templates: {list(templates.keys())}"
            )
        return templates[name]

    @staticmethod
    def list_templates() -> list[str]:
        """List all available template names."""
        return list(PromptLibrary.get_all_templates().keys())
