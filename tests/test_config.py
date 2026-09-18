"""Config 单元测试。"""

from __future__ import annotations

import pytest

from codeagentx.config import Config, get_config, load_env_file, set_config
from codeagentx.core.exceptions import ConfigError


def test_defaults_without_env(config: Config) -> None:
    assert config.llm_model_id
    assert config.llm_api_key == ""
    assert config.is_llm_configured is False
    assert config.llm_max_retries >= 0


def test_env_value_mapping(monkeypatch) -> None:
    """字段名大写即环境变量名。"""
    monkeypatch.setenv("LLM_MODEL_ID", "my-model")
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")

    cfg = Config.from_env(load_dotenv_file=False)

    assert cfg.llm_model_id == "my-model"
    assert cfg.llm_max_tokens == 512
    assert cfg.llm_temperature == 0.7


def test_overrides_win_over_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL_ID", "from-env")
    cfg = Config.from_env(load_dotenv_file=False, overrides={"llm_model_id": "from-override"})
    assert cfg.llm_model_id == "from-override"


def test_invalid_value_raises_config_error(monkeypatch) -> None:
    monkeypatch.setenv("LLM_TEMPERATURE", "not-a-number")
    with pytest.raises(ConfigError) as excinfo:
        Config.from_env(load_dotenv_file=False)
    # pydantic 报错信息里是字段名（小写），只要定位到出错字段即可
    assert "llm_temperature" in (excinfo.value.detail or "")


def test_unknown_env_ignored(monkeypatch) -> None:
    monkeypatch.setenv("LLM_UNKNOWN_FIELD", "whatever")
    cfg = Config.from_env(load_dotenv_file=False)
    assert not hasattr(cfg, "llm_unknown_field")


def test_require_llm_credentials(config: Config) -> None:
    with pytest.raises(ConfigError):
        config.require_llm_credentials()

    config.llm_api_key = "sk-test"
    assert config.is_llm_configured is True
    config.require_llm_credentials()  # 不应抛异常


def test_masked_summary_hides_secret() -> None:
    cfg = Config.from_env(
        load_dotenv_file=False,
        overrides={"llm_api_key": "sk-1234567890abcdef", "github_token": "ghp_abcdefghijklmn"},
    )
    summary = cfg.masked_summary()

    assert "1234567890abcdef" not in summary["llm_api_key"]
    assert summary["llm_api_key"].startswith("sk-")
    assert "abcdefghijklmn" not in summary["github_token"]
    assert summary["github_token"].startswith("ghp")


def test_short_secret_fully_masked() -> None:
    cfg = Config.from_env(load_dotenv_file=False, overrides={"llm_api_key": "short"})
    assert cfg.masked_summary()["llm_api_key"] == "***"


def test_path_resolution_is_absolute(config: Config) -> None:
    assert config.log_dir_path.is_absolute()
    assert config.workspace_path.is_absolute()
    assert config.workspace_path.name == "outputs"  # 来自 isolated_env 的 WORKSPACE_DIR


def test_load_env_file(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# 注释行\nLLM_API_KEY=sk-from-file\nLLM_MODEL_ID=file-model\n\nINVALID_LINE\n",
        encoding="utf-8",
    )

    load_env_file(env_file)
    cfg = Config.from_env(env_file, load_dotenv_file=False)

    assert cfg.llm_api_key == "sk-from-file"
    assert cfg.llm_model_id == "file-model"


def test_load_env_file_missing_is_noop(tmp_path) -> None:
    missing = tmp_path / "not-exists.env"
    assert load_env_file(missing) == missing

    with pytest.raises(ConfigError):
        load_env_file(missing, required=True)


def test_get_config_singleton_and_reload() -> None:
    set_config(None)
    first = get_config()
    assert get_config() is first

    second = get_config(reload=True)
    assert second is not first

    custom = Config(llm_model_id="injected")
    set_config(custom)
    assert get_config() is custom
