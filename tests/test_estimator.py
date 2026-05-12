import numpy as np
import pytest
from hybrid_forecaster import HybridForecaster

def test_runs():
    # dummy data
    X = np.arange(20).reshape(-1, 1)
    y = 2 * X.flatten() + 5
    
    model = HybridForecaster()
    model.fit(X, y)
    
    # kaj korche?
    X_new = np.array([[21], [22]])
    res = model.predict(X_new)
    
    assert len(res) == 2