# tests/test_metrics.py
import numpy as np
from dec.metrics import accuracy, ece, brier, qwk, auroc, flip_rate, realized_cost
from dec.schema import CostMatrix

P = np.array([[0.9, 0.1], [0.4, 0.6], [0.8, 0.2], [0.3, 0.7]])
G = np.array([0, 1, 1, 1])

def test_accuracy_brier():
    assert accuracy(P, G) == 0.75
    assert abs(brier(P, G) - np.mean([(0.1)**2*2, (0.4)**2*2, (0.8)**2+(0.8)**2, (0.3)**2*2])) < 1e-9

def test_ece_perfect_and_bad():
    perfect = np.array([[1.0, 0.0]] * 10); g = np.zeros(10, int)
    assert ece(perfect, g) == 0.0
    bad = np.array([[0.9, 0.1]] * 10); g2 = np.ones(10, int)
    assert abs(ece(bad, g2) - 0.9) < 1e-9

def test_qwk_identity_is_one():
    assert qwk(np.array([0, 1, 2, 3]), np.array([0, 1, 2, 3]), 4) == 1.0

def test_auroc():
    assert auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75

def test_flip_rate_and_cost():
    assert flip_rate(np.array([0, 1, 1]), np.array([0, 0, 1])) == 1 / 3
    cm = CostMatrix(loss={("no", "yes"): 10.0})
    assert realized_cost(["no", "yes"], ["yes", "yes"], cm) == 10.0
