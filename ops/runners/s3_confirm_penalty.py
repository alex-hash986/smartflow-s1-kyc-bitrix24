"""Record already verified penalty terms for the active S3 package."""

import argparse
import asyncio
from decimal import Decimal
from uuid import UUID

from core.config import Settings
from core.database import Database
from core.idempotency import Conflict
from core.llm_gateway import LLMGateway
from modules.s3_claims.contracts import S3TenderGateRequest, as_money
from modules.s3_claims.module import S3Worker


def rate(value: str) -> Decimal:
    try:
        return as_money(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected a nonnegative decimal with up to two places"
        ) from exc


async def record(args: argparse.Namespace) -> None:
    settings = Settings()
    db = Database(settings.sqlite_path)
    gateway = LLMGateway(settings)
    try:
        worker = S3Worker(settings.profiles_path, gateway, db)
        worker.record_penalty_terms(
            S3TenderGateRequest(
                tenant_id="portal_b24_b2b",
                profile_slug="portal_b24_b2b",
                profile_revision="5",
                request_id=args.request_id,
                package_id=args.package_id,
            ),
            evidence_ref=args.evidence_ref,
            daily_penalty_pct=args.daily_pct,
            max_penalty_pct=args.max_pct,
        )
    finally:
        await gateway.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", type=UUID, required=True)
    parser.add_argument("--package-id", type=UUID, required=True)
    parser.add_argument(
        "--evidence-ref", required=True, help="Reference to verified contract evidence"
    )
    parser.add_argument("--daily-pct", type=rate, required=True)
    parser.add_argument("--max-pct", type=rate, required=True)
    args = parser.parse_args()
    try:
        asyncio.run(record(args))
    except Conflict as exc:
        parser.exit(2, f"{exc}\n")
    except ValueError:
        parser.exit(2, "Invalid settings or contract evidence\n")
    print("Verified S3 penalty terms recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
