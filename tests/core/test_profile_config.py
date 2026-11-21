from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.config import Settings
from core.profile_loader import load_profile


def test_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "FAST_LLM_MODEL=fast\nAPP_PORT=8030\nMAX_CYCLE_COST_RUB=0.125\n"
        'LLM_MODEL_RATES={"fast":{"input_usd_per_million":"0.1","output_usd_per_million":"0.2"}}\n'
        "BITRIX24_WEBHOOK_URL=https://crm.invalid/rest/1/SECRET/\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.fast_llm_model == "fast" and settings.app_port == 8030
    assert settings.max_cycle_cost_rub == Decimal(".125")
    assert settings.llm_model_rates["fast"].output_usd_per_million == Decimal(".2")
    assert "SECRET" not in repr(settings)
    with pytest.raises(ValidationError):
        Settings(bitrix24_rate_limit_rps=2.1)


def test_profile_loader(tmp_path: Path) -> None:
    profile = tmp_path / "demo" / "profile.yaml"
    profile.parent.mkdir()
    profile.write_text(
        "slug: demo\nrevision: v1\ntenant_id: tenant\nportal_id: portal\ntimezone: Asia/Novosibirsk\n"
        "stages:\n  S1:\n    entity_type_id: 177\n    category_id: 0\n    stage_id: NEW\n"
        "    external_key_field: ufExternal\n    responsible_id: 7\n",
        encoding="utf-8",
    )
    assert load_profile(tmp_path, "demo").stages["S1"].entity_type_id == 177
    with pytest.raises(ValueError, match="slug"):
        load_profile(tmp_path, "../demo")
    fixture = profile.read_text(encoding="utf-8")
    profile.write_text(fixture.replace("slug: demo", "slug: wrong"), encoding="utf-8")
    with pytest.raises(ValueError, match="slug"):
        load_profile(tmp_path, "demo")
    profile.write_text(fixture.replace("Asia/Novosibirsk", "Invalid/Timezone"), encoding="utf-8")
    with pytest.raises(ValidationError, match="timezone"):
        load_profile(tmp_path, "demo")
