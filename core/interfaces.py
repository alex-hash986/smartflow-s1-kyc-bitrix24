from abc import ABC, abstractmethod
from dataclasses import dataclass

from pydantic import BaseModel

from core.contracts import WorkerRequestEnvelope, WorkerResponseEnvelope
from core.llm_gateway import CycleBudget


@dataclass(frozen=True)
class WorkerResult:
    envelope: WorkerResponseEnvelope
    tokens_in: int | None = None
    tokens_out: int | None = None
    prompt_versions: tuple[str, ...] = ()


class BaseSmartWorker(ABC):
    request_payload: type[BaseModel]
    response_payload: type[BaseModel]

    @abstractmethod
    async def run(self, request: WorkerRequestEnvelope, budget: CycleBudget) -> WorkerResult:
        """Compute a result without making CRM writes."""
