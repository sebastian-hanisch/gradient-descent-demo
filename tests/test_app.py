from streamlit.testing.v1 import AppTest

_APP_TIMEOUT = 60


def _fresh():
    at = AppTest.from_file("../app.py", default_timeout=_APP_TIMEOUT)
    at.run()
    return at


def test_app_runs_without_exception():
    at = _fresh()
    assert not at.exception


def test_footer_is_present():
    at = _fresh()
    captions = [c.value for c in at.caption]
    assert any("Sebastian Hanisch" in c and "Über mich" in c for c in captions)


def test_preset_gut_konditioniert_converges():
    at = _fresh()
    btn = [b for b in at.button if b.label == "Gut konditioniert — schnelle Konvergenz"][0]
    btn.click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Konvergiert?"] == "Ja"


def test_kappa_slider_extreme_values_do_not_crash():
    at = _fresh()
    k_slider = [s for s in at.slider if s.label == "Konditionszahl κ"][0]
    k_slider.set_value(k_slider.min).run()
    assert not at.exception
    k_slider = [s for s in at.slider if s.label == "Konditionszahl κ"][0]
    k_slider.set_value(k_slider.max).run()
    assert not at.exception


def test_switching_to_rosenbrock_does_not_crash():
    at = _fresh()
    radio = [r for r in at.radio if r.label == "Funktion"][0]
    radio.set_value("rosenbrock").run()
    assert not at.exception
