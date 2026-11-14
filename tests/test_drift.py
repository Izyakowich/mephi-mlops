import numpy as np

from src.monitoring.drift import (
    calculate_psi,
    get_drift_status,
)


def test_psi_identical_distributions():
    """PSI одинаковых распределений равен нулю."""

    data = np.arange(1000, dtype=float)

    psi = calculate_psi(data, data)

    assert np.isclose(psi, 0.0)


def test_psi_detects_shift():
    """PSI обнаруживает сильное изменение данных."""

    reference = np.arange(1000, dtype=float)
    current = reference + 10000

    psi = calculate_psi(reference, current)

    assert psi > 0.25


def test_drift_status():
    """Проверка порогов PSI."""

    assert get_drift_status(0.05) == "stable"
    assert get_drift_status(0.15) == "warning"
    assert get_drift_status(0.30) == "significant_drift"
