"""Testes unitários da camada de monitoramento."""

from src.monitoring.tracking import _escape, build_dashboard, evaluate_alerts, log_batch_run


def test_alert_rules_use_configured_psi_thresholds() -> None:
    """PSI em atenção/crítico e score geram alertas com severidade distinta."""
    alerts = evaluate_alerts({"psi_limit_balance": 0.12, "psi_payment": 0.26, "psi_score": 0.11})
    assert [(item["name"], item["severity"]) for item in alerts] == [
        ("psi_limit_balance", "warning"),
        ("psi_payment", "critical"),
        ("psi_score", "warning"),
    ]


def test_stage2_performance_and_drift_alerts() -> None:
    """Alertas correspondem aos deltas e à fração calculados pela Etapa 2."""
    metrics = {
        "auc_drop": 0.03,
        "recall_drop": 0.05,
        "share_features_drifted": 0.20,
    }
    alerts = evaluate_alerts(metrics)
    assert [(alert["name"], alert["severity"]) for alert in alerts] == [
        ("auc_drop", "warning"),
        ("recall_drop", "warning"),
        ("share_features_drifted", "warning"),
    ]
    severe = evaluate_alerts(
        {"auc_drop": 0.05, "recall_drop": 0.10, "share_features_drifted": 0.50}
    )
    assert all(alert["severity"] == "critical" for alert in severe)


def test_alert_rules_ignore_healthy_metrics() -> None:
    """Métricas abaixo dos limites não devem gerar alertas."""
    assert evaluate_alerts({"psi_feature": 0.04, "auc_drop": 0.01}) == []


def test_mlflow_run_and_static_dashboard(tmp_path, monkeypatch) -> None:
    """Uma execução batch local aparece no HTML consolidado."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'tracking.db'}")
    log_batch_run(
        "batch-test",
        scenario="unit_test",
        status="blocked",
        metrics={"rows_in": 10, "rows_out": 0},
    )
    dashboard = build_dashboard(tmp_path / "dashboard.html")
    html = dashboard.read_text(encoding="utf-8")
    assert "batch-test" in html
    assert "blocked" in html
    assert "rows_out" in html


def test_html_values_are_escaped() -> None:
    """O dashboard não interpreta valores de tags como markup HTML."""
    assert _escape("<script>alert('x')</script>") == (
        "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;"
    )
