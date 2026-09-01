from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_main_app_starts_without_exceptions():
    app = AppTest.from_file(ROOT / "main.py", default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "Product Review Sentiment Analysis"


def test_reliable_inference_page_starts_without_exceptions():
    app = AppTest.from_file(ROOT / "main.py", default_timeout=10).run()
    app.switch_page("pages/1_Reliable_Inference.py").run(timeout=10)
    assert not app.exception
    assert app.title[0].value == "Reliable Sentiment Inference"
