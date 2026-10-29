from pydantic import BaseModel, ConfigDict, Field


class ShellCommand(BaseModel):
    """Strict contract for a single macOS terminal command returned by the LLM."""

    model_config = ConfigDict(extra="forbid", strict=True)

    command: str = Field(description="The raw macOS terminal command to run.")
    explanation: str = Field(description="A one-sentence breakdown of what the command does.")
    is_safe: bool = Field(
        description="True if read-only (like ls, df, lsof), False if it modifies or deletes anything."
    )
