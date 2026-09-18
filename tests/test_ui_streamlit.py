"""网页界面的离线冒烟测试（Streamlit AppTest）。

只验证"外壳"：页面能起来、控件在、空输入点提交会报错而不是去调模型。
**不点"开始审查"**——那会真跑一次审查（花钱），端到端由真实联调覆盖。

需要 `.[serve]` 依赖（streamlit）；没装时整组跳过，而不是让套件变红。
"""

from __future__ import annotations

from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "src" / "codeagentx" / "ui" / "streamlit_app.py"


@pytest.fixture
def app_test() -> object:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=60)


def test_page_renders_without_exception(app_test: object) -> None:
    app_test.run()  # type: ignore[attr-defined]

    assert not app_test.exception  # type: ignore[attr-defined]
    assert app_test.title[0].value.startswith("CodeAgentX")  # type: ignore[attr-defined]
    assert [button.label for button in app_test.button] == ["开始审查"]  # type: ignore[attr-defined]
    assert "data/sample_repo" in app_test.info[0].value  # type: ignore[attr-defined]


def test_empty_target_reports_error_without_running_review(app_test: object) -> None:
    """空目标必须立刻报错——绝不能顺手去审"当前目录"这种东西。"""
    app_test.run()  # type: ignore[attr-defined]
    app_test.button[0].click().run()  # type: ignore[attr-defined]

    assert not app_test.exception  # type: ignore[attr-defined]
    assert "请填写审查目标" in app_test.sidebar.error[0].value  # type: ignore[attr-defined]
