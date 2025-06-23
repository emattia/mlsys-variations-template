"""Concrete implementations of FineTuningPipeline plugins."""

import openai

from .base import ExecutionContext, FineTuningPipeline
from .registry import register_plugin


@register_plugin(
    name="openai_fine_tuning",
    category="fine_tuning_pipeline",
    description="A fine-tuning pipeline that uses the OpenAI API.",
)
class OpenAIFineTuning(FineTuningPipeline):
    """A fine-tuning pipeline that uses the OpenAI API."""

    def initialize(self, context: ExecutionContext) -> None:
        """Initializes the OpenAI client."""
        self.api_key = self.config.get("api_key")
        if not self.api_key:
            raise ValueError(
                "'api_key' not specified in the configuration for OpenAIFineTuning."
            )
        self.client = openai.OpenAI(api_key=self.api_key)
        self.logger.info("OpenAIFineTuning initialized.")

    def run(self, dataset_path: str, model: str = "gpt-3.5-turbo") -> dict:
        """Runs the fine-tuning job on the OpenAI API."""
        try:
            # 1. Upload the file
            self.logger.info(f"Uploading dataset: {dataset_path}")
            with open(dataset_path, "rb") as f:
                file_response = self.client.files.create(
                    file=f,
                    purpose="fine-tune",
                )
            file_id = file_response.id
            self.logger.info(f"Dataset uploaded successfully. File ID: {file_id}")

            # 2. Create the fine-tuning job
            self.logger.info(f"Creating fine-tuning job for model: {model}")
            job_response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model,
            )
            job_id = job_response.id
            self.logger.info(f"Fine-tuning job created successfully. Job ID: {job_id}")

            return {"job_id": job_id, "status": "created"}
        except Exception as e:
            self.logger.error(f"Error creating OpenAI fine-tuning job: {e}")
            raise

    def list_jobs(self, limit: int = 10) -> list:
        """Lists the fine-tuning jobs."""
        try:
            return self.client.fine_tuning.jobs.list(limit=limit).data
        except Exception as e:
            self.logger.error(f"Error listing OpenAI fine-tuning jobs: {e}")
            raise

    def get_status(self, job_id: str) -> dict:
        """Gets the status of a fine-tuning job."""
        try:
            return self.client.fine_tuning.jobs.retrieve(job_id).dict()
        except Exception as e:
            self.logger.error(f"Error retrieving status for job {job_id}: {e}")
            raise
