import pytest


@pytest.mark.slow
def test_small_aapl_parquet_smoke(project_root):
    pd = pytest.importorskip("pandas")

    book_path = project_root / "data" / "AAPL_1s.parquet"
    event_path = project_root / "data" / "AAPL_evt.parquet"
    if not book_path.exists() or not event_path.exists():
        pytest.skip("AAPL parquet files are not available in this checkout")

    book = pd.read_parquet(book_path, columns=["time_abs_s", "mid"])
    events = pd.read_parquet(event_path, columns=["time", "event_type", "price", "size", "direction"])

    assert not book.head(5).empty
    assert not events.head(5).empty
    assert {"time_abs_s", "mid"} <= set(book.columns)
    assert {"time", "event_type", "price", "size", "direction"} <= set(events.columns)

