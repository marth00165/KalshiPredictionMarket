import json
from types import SimpleNamespace

from app.analysis.claude_analyzer import ClaudeAnalyzer
from app.analysis.openai_analyzer import OpenAIAnalyzer
from app.models import MarketData


def _market() -> MarketData:
    return MarketData(
        platform="kalshi",
        market_id="KXNBAGAME-26FEB19DETNYK-DET",
        title="Detroit at New York Winner?",
        description="",
        yes_price=0.40,
        no_price=0.60,
        volume=15000,
        liquidity=7000,
        end_date="2026-02-19T00:00:00Z",
        category="sports",
        event_ticker="KXNBAGAME-26FEB19DETNYK",
        series_ticker="KXNBAGAME",
        yes_option="Detroit",
        no_option="New York",
    )


def _analysis_config(context_path: str):
    return SimpleNamespace(
        context_json_path=context_path,
        context_max_chars=12000,
    )


def test_openai_prompt_includes_context_json(tmp_path):
    context_path = tmp_path / "llm_context.json"
    context_path.write_text(json.dumps({"angle": "injury watch", "priority": "pace"}))

    cfg = SimpleNamespace(
        openai_api_key="test",
        openai=SimpleNamespace(
            base_url="https://api.openai.com/v1",
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
        ),
        api=SimpleNamespace(openai_api_key="test"),
        analysis=_analysis_config(str(context_path)),
    )

    analyzer = OpenAIAnalyzer(cfg)
    prompt = analyzer._build_analysis_prompt(_market())
    assert "ADDITIONAL USER CONTEXT" in prompt
    assert "injury watch" in prompt


def test_openai_prompt_includes_runtime_reasoning_context(tmp_path):
    context_path = tmp_path / "llm_context.json"
    context_path.write_text(json.dumps({"angle": "pace"}))

    cfg = SimpleNamespace(
        openai_api_key="test",
        openai=SimpleNamespace(
            base_url="https://api.openai.com/v1",
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000,
        ),
        api=SimpleNamespace(openai_api_key="test"),
        analysis=_analysis_config(str(context_path)),
    )

    analyzer = OpenAIAnalyzer(cfg)
    analyzer.set_runtime_context_block("Past note: market overreacted to injury headline.")
    prompt = analyzer._build_analysis_prompt(_market())
    assert "RECENT INTERNAL ANALYSIS CONTEXT" in prompt
    assert "Past note: market overreacted to injury headline." in prompt


def test_claude_prompt_includes_context_json(tmp_path):
    context_path = tmp_path / "llm_context.json"
    context_path.write_text(json.dumps({"rule": "prefer underdogs only with strong edge"}))

    cfg = SimpleNamespace(
        claude_api_key="test",
        claude=SimpleNamespace(
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
            model="claude-sonnet",
            temperature=0.3,
            max_tokens=1000,
        ),
        analysis=_analysis_config(str(context_path)),
    )

    analyzer = ClaudeAnalyzer(cfg)
    prompt = analyzer._build_analysis_prompt(_market())
    assert "ADDITIONAL USER CONTEXT" in prompt
    assert "prefer underdogs" in prompt


def test_claude_prompt_includes_runtime_reasoning_context(tmp_path):
    context_path = tmp_path / "llm_context.json"
    context_path.write_text(json.dumps({"rule": "small sample"}))

    cfg = SimpleNamespace(
        claude_api_key="test",
        claude=SimpleNamespace(
            input_cost_per_mtok=0.0,
            output_cost_per_mtok=0.0,
            model="claude-sonnet",
            temperature=0.3,
            max_tokens=1000,
        ),
        analysis=_analysis_config(str(context_path)),
    )

    analyzer = ClaudeAnalyzer(cfg)
    analyzer.set_runtime_context_block("Past note: close games were consistently underpriced.")
    prompt = analyzer._build_analysis_prompt(_market())
    assert "RECENT INTERNAL ANALYSIS CONTEXT" in prompt
    assert "Past note: close games were consistently underpriced." in prompt
