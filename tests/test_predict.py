from src.betting.edge import expected_value


def test_predict_response_shape():
    ev = expected_value(0.55, -110)
    assert isinstance(ev, float)
