import asyncio

from src.server import handle_ml_dashboard_page, handle_ml_training_page


def test_signal_dashboard_page_uses_action_queue_copy_and_no_tailwind_cdn():
    response = asyncio.run(handle_ml_dashboard_page(None))
    html = response.text

    assert response.status == 200
    assert "Fila de ação do LSTM" in html
    assert "Curadoria de treino" in html
    assert "cdn.tailwindcss.com" not in html
    assert "/api/v1/ml/dashboard" in html


def test_training_dashboard_page_exposes_exception_inbox_copy_and_training_apis():
    response = asyncio.run(handle_ml_training_page(None))
    html = response.text

    assert response.status == 200
    assert "Sessões contínuas do LSTM" in html
    assert "Inbox de exceções" in html
    assert "/api/v1/ml/training/sessions" in html
    assert "/api/v1/ml/training/cohorts/preview" in html
    assert "Aprovar após revisão" in html
