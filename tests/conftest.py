import pytest


@pytest.fixture(autouse=True)
def fake_env(monkeypatch):
    """Aísla los tests de las variables reales del .env."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("LLM_TIMEOUT", "5")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("llm_client.models._load_env", lambda: None)


@pytest.fixture
def fake_chat_openai(mocker):
    """Mockea la clase ChatOpenAI para no golpear la API real."""
    mock_class = mocker.patch("llm_client.models.ChatOpenAI")
    instance = mock_class.return_value
    instance.invoke.return_value = "ok"
    return mock_class


@pytest.fixture(autouse=True)
def fake_chat_models(mocker):
    mocker.patch("llm_client.models.ChatOpenAI")
    mocker.patch("llm_client.models.ChatAnthropic")
    mocker.patch("llm_client.models.ChatGoogleGenerativeAI")
