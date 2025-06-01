import pytest
import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression
from utils import (
    remove_outliers,
    scale_columns,
    evaluate_model,
    train_final_model
)


@pytest.fixture
def sample_data():
    data = {
        "TV": [100, 150, 200, 300, 1000],  # 1000 es un outlier
        "Radio": [20, 25, 30, 35, 40],
        "Newspaper": [10, 15, 10, 15, 500],  # 500 es un outlier
        "Sales": [10, 15, 20, 25, 30]
    }
    return pl.DataFrame(data)


def test_remove_outliers(sample_data):
    df_filtered = remove_outliers(sample_data, ["TV", "Newspaper"])

    # Verifica que los outliers han sido eliminados
    assert df_filtered.shape[0] < sample_data.shape[0]
    assert df_filtered["TV"].max() < 1000
    assert df_filtered["Newspaper"].max() < 500


def test_scale_columns(sample_data):
    df_scaled = scale_columns(sample_data, ["TV", "Radio"])

    # Verifica que se han añadido las columnas escaladas
    assert "TV_scaled" in df_scaled.columns
    assert "Radio_scaled" in df_scaled.columns

    # Verifica media ~0 y std ~1
    tv_scaled = df_scaled["TV_scaled"].to_numpy()
    assert np.isclose(tv_scaled.mean(), 0, atol=1e-7)
    assert np.isclose(tv_scaled.std(ddof=0), 1, atol=1e-7)


def test_evaluate_model_normal_case():
    X = np.array([[i] for i in range(10)])
    y = np.array([2 * i for i in range(10)])

    score = evaluate_model(X, y)

    assert isinstance(score, float)
    assert 0.9 <= score <= 1.0



def test_evaluate_model_empty():
    # Edge case: vacío
    X = np.array([]).reshape(0, 1)
    y = np.array([])

    with pytest.raises(ValueError):
        evaluate_model(X, y)


def test_train_final_model_output():
    X = np.array([[i] for i in range(10)])
    y = np.array([2 * i for i in range(10)])

    model, score = train_final_model(X, y)

    assert isinstance(score, float)
    assert isinstance(model, LinearRegression)
    assert 0.9 <= score <= 1.0


def test_train_final_model_mismatch_shape():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])  # Mal dimensionado

    with pytest.raises(ValueError):
        train_final_model(X, y)
