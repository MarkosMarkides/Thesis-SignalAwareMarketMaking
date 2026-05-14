import json

import pytest


@pytest.mark.artifact
def test_main_model_artifacts_can_be_loaded(artifact_dir):
    joblib = pytest.importorskip("joblib")

    direction_model = joblib.load(artifact_dir / "direction_xgb.joblib")
    volatility_model = joblib.load(artifact_dir / "vol_xgb.joblib")
    metadata = json.loads((artifact_dir / "model_metadata.json").read_text())

    assert not (artifact_dir / "direction_rf.joblib").exists()
    assert not (artifact_dir / "vol_lstm_state.pt").exists()
    assert not (artifact_dir / "vol_feature_scaler.joblib").exists()
    assert not (artifact_dir / "vol_target_scaler.joblib").exists()
    assert hasattr(direction_model, "predict")
    assert hasattr(volatility_model, "predict")
    assert metadata["symbol"] == "AAPL"
    assert metadata["horizon"] == 20
    assert isinstance(metadata["selected_features"], list)
    assert len(metadata["selected_features"]) > 10
    assert "train_sessions" in metadata
    assert "test_sessions" in metadata
    assert metadata["direction_model"]["model"] == "XGBClassifier"
    assert metadata["volatility_model"]["model"] == "XGBRegressor"
