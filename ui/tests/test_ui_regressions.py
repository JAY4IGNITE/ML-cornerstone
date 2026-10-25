"""Regression tests for interactive state and evidence presentation."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest


def test_empty_importance_is_a_supported_state():
    app = AppTest.from_string(
        "from ui.pages.explainability import _importance_block\n"
        "_importance_block({'method': 'unavailable', 'features': []}, kind='tree')\n"
    ).run(timeout=30)
    assert not app.exception
    assert any("No importance" in item.value for item in app.info)


def test_preset_replaces_keyed_widget_values():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from ui.pages.assessment import apply_preset\n"
        "st.button('Preset', on_click=apply_preset, args=('prime',))\n"
        "st.number_input('Income', value=1.0, key='inp_AMT_INCOME_TOTAL')\n"
        "st.selectbox('Occupation', ['Managers', 'Sales staff'], key='inp_OCCUPATION_TYPE')\n"
    ).run(timeout=30)
    app.number_input[0].set_value(42.0).run()
    app.selectbox[0].select('Sales staff').run()
    app.button[0].click().run()
    assert not app.exception
    assert app.number_input[0].value == 250000.0
    assert app.selectbox[0].value == 'Managers'


def test_reset_clears_prediction_and_form_state(monkeypatch):
    from ui.pages import assessment

    state = {'last_prediction': {'result': {}}, 'form_data': {'x': 1},
             'form_preset': 'prime', 'inp_AMT_CREDIT': 400, 'unrelated': 7}
    monkeypatch.setattr(assessment.st, 'session_state', state)
    assessment.reset_assessment()
    assert state == {'unrelated': 7}


def test_analytics_respects_configured_source(monkeypatch, tmp_path):
    from ui.pages import analytics
    raw, synth = tmp_path / 'raw', tmp_path / 'synthetic'
    raw.mkdir()
    synth.mkdir()
    (raw / 'real.csv').touch()
    (synth / 'synth.csv').touch()

    class Config(dict):
        def path(self, name):
            return {'data_raw': raw, 'data_synthetic': synth}[name]

    cfg = Config(dataset={'source': 'synthetic', 'primary_file': 'real.csv',
                          'synthetic_file': 'synth.csv'})
    monkeypatch.setattr(analytics, 'get_service', lambda: SimpleNamespace(cfg=cfg))
    assert analytics._resolve() == (str(synth / 'synth.csv'), True)
    (synth / 'synth.csv').unlink()
    assert analytics._resolve() == (None, False)


@pytest.mark.parametrize('module', ['result', 'explainability'])
def test_no_prediction_actions_without_loaded_model(module, monkeypatch):
    import importlib
    page = importlib.import_module(f'ui.pages.{module}')
    monkeypatch.setattr(page, 'get_service', lambda: SimpleNamespace(available=False))
    app = AppTest.from_string(f'from ui.pages.{module} import render\nrender()').run(timeout=30)
    assert not app.exception
    assert not app.button


def test_unsigned_importance_not_presented_as_signed_shap():
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from ui.pages.explainability import _applicant_explanation\n"
        "st.session_state['last_prediction'] = {'result': {"
        "'default_probability': .2, 'risk_score': 20, 'explanation_available': True,"
        "'explanation': {'method': 'tree_importance_fallback', 'contributions': ["
        "{'feature': 'AGE_YEARS', 'contribution': .6, 'direction': 'n/a'}]}}}\n"
        "_applicant_explanation()"
    ).run(timeout=30)
    assert not app.exception
    text = ' '.join(item.value for item in app.markdown)
    assert 'Game-theoretic exact' not in text
    assert any('global' in item.value.lower() for item in app.info)


def test_service_cache_reloads_when_artifacts_appear(monkeypatch, tmp_path):
    from ui import helpers
    from loan_risk.config import Config
    cfg = Config(raw={'paths': {'artifacts': str(tmp_path)}})
    monkeypatch.setattr(helpers, 'load_config', lambda: cfg)
    monkeypatch.setattr(helpers, 'ModelService', lambda cfg: object())
    helpers.get_service.clear()
    first = helpers.get_service()
    assert helpers.get_service() is first
    (tmp_path / 'metadata.json').write_text('{}')
    second = helpers.get_service()
    assert second is not first
    assert helpers.get_service() is second
    helpers.get_service.clear()


def test_dataset_cache_reloads_replaced_file(tmp_path):
    from ui.pages.analytics import _load
    path = tmp_path / 'data.csv'
    path.write_text('TARGET\n0\n')
    before = path.stat()
    assert len(_load(str(path), before.st_mtime_ns, before.st_size)) == 1
    path.write_text('TARGET\n0\n1\n')
    after = path.stat()
    assert len(_load(str(path), after.st_mtime_ns, after.st_size)) == 2


def test_worsening_calibration_does_not_claim_improvement():
    app = AppTest.from_string(
        "from ui.pages.performance import _calibration\n"
        "_calibration({'calibration': {'brier_before': .1, 'brier_after': .2}})"
    ).run(timeout=30)
    assert not app.exception
    captions = ' '.join(item.value for item in app.caption)
    assert 'reduction' not in captions
    assert 'Isotonic' not in captions
    assert '0.2000' in captions


def test_model_comparison_does_not_clip_scores(monkeypatch):
    from ui.pages import model_selection
    charts = []
    monkeypatch.setattr(model_selection.charts, 'show', lambda chart, **kw: charts.append(chart.to_dict()))
    model_selection._cv_comparison({
        'logistic_regression': {'display_name': 'Logistic Regression', 'mean': .91, 'std': .02},
        'decision_tree': {'display_name': 'Decision Tree', 'mean': .61, 'std': .03},
    }, 'Logistic Regression')
    scale = charts[0]['layer'][0]['encoding']['x']['scale']
    assert 'domain' not in scale


def test_assessment_submit_and_result_reset(service_with_model):
    app = AppTest.from_string(
        "from ui.pages.assessment import render\nrender()"
    ).run(timeout=60)
    app.button(key='btn_preset_standard').click().run(timeout=60)
    submit = next(b for b in app.button if 'Run Risk Assessment' in b.label)
    submit.click().run(timeout=60)
    assert not app.exception
    assert app.session_state['last_prediction']['result']['default_probability'] >= 0
    app.number_input(key='inp_CNT_CHILDREN').set_value(10.0)
    next(b for b in app.button if 'Run Risk Assessment' in b.label).click().run(timeout=60)
    assert not app.exception
    assert app.error
    with pytest.raises(KeyError):
        app.session_state['last_prediction']


def test_result_reset_navigates_to_assessment(service_with_model):
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from ui.helpers import get_service\n"
        "from ui.pages import assessment, result\n"
        "from ui.state import save_prediction\n"
        "if 'initialized' not in st.session_state:\n"
        "    p = assessment.PRESETS['standard']['data']\n"
        "    save_prediction(p, get_service().predict(p, with_explanation=False))\n"
        "    st.session_state.initialized = True\n"
        "st.navigation([st.Page(result.render, title='Prediction Result', url_path='result', default=True),"
        "st.Page(assessment.render, title='Risk Assessment', url_path='assessment')]).run()"
    ).run(timeout=60)
    next(b for b in app.button if 'Assess Another Applicant' in b.label).click().run(timeout=60)
    assert not app.exception
    with pytest.raises(KeyError):
        app.session_state['last_prediction']
    assert any('Loan Risk Assessment' in t.value for t in app.title)
