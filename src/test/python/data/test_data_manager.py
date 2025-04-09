from data.data_manager import download_price_data
from utils.tickers import tickers


def test_data_manager():
    df = download_price_data(
        tickers=tickers,
        start="2018-01-01",
        end="2024-12-31",
    )

    assert df.size > 0
