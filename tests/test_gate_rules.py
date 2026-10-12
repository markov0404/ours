from scripts.gates.rules import g1_rule, g2_rule, g3_rule, g4_rule, g5_rule, g6_rule, g8_rule

def _curve(acc_by, ece_by):
    return {"agg": {k: {"acc": (v, 0.0), "ece": (ece_by[k], 0.0), "brier": (0.0, 0.0)} for k, v in acc_by.items()}}

def test_g1_pass_when_two_of_three_domains_gain_3pp_and_lower_ece():
    good = _curve({"B1@1000": 0.70, "ours1@1000": 0.75}, {"B1@1000": 0.08, "ours1@1000": 0.05})
    bad = _curve({"B1@1000": 0.70, "ours1@1000": 0.71}, {"B1@1000": 0.08, "ours1@1000": 0.09})
    assert g1_rule({"plumb": good, "massive": good, "clinc": bad})["zone"] == "A"
    assert g1_rule({"plumb": good, "massive": bad, "clinc": bad})["zone"] == "C"

def test_g2_requires_gain_in_multiquestion_and_no_regression():
    mq = _curve({"ours1@1000": 0.70, "ours2@1000": 0.72}, {"ours1@1000": 0.05, "ours2@1000": 0.05})
    ok = _curve({"ours1@1000": 0.80, "ours2@1000": 0.80}, {"ours1@1000": 0.05, "ours2@1000": 0.05})
    assert g2_rule({"plumb": mq, "civil": mq, "massive": ok, "clinc": ok})["zone"] == "A"
    reg = _curve({"ours1@1000": 0.80, "ours2@1000": 0.77}, {"ours1@1000": 0.05, "ours2@1000": 0.05})
    assert g2_rule({"plumb": mq, "civil": mq, "massive": reg, "clinc": ok})["zone"] == "C"

def test_g3_g4_g6():
    assert g3_rule({"massive": {"coherence": {"incoherence_before": 0.10, "incoherence_after": 0.04, "acc_after": 0.81}, "acc": 0.80},
                    "civil": {"coherence": {"incoherence_before": 0.06, "incoherence_after": 0.02, "acc_after": 0.90}, "acc": 0.90}})["zone"] == "A"
    assert g4_rule({0.02: {"risk": {"realized_risk": 0.015, "automation": 0.55}}, 0.05: {"risk": {"realized_risk": 0.04, "automation": 0.7}}},
                   {"auroc_conf": 0.85, "auroc_conf_mass": 0.88})["zone"] == "A"
    assert g6_rule({"acc": 0.78, "ece": 0.06}, {"acc": 0.78, "ece": 0.06})["zone"] == "A"

def test_g4_g6_fail_cases():
    ok = {"risk": {"realized_risk": 0.015, "automation": 0.55}}
    bad = {"risk": {"realized_risk": 0.06, "automation": 0.7}}
    assert g4_rule({0.02: ok, 0.05: bad}, {"auroc_conf": 0.8, "auroc_conf_mass": 0.9})["zone"] == "C"
    assert g4_rule({0.02: ok, 0.05: ok}, {"auroc_conf": 0.9, "auroc_conf_mass": 0.8})["zone"] == "C"
    assert g6_rule({"acc": 0.70, "ece": 0.06}, {"acc": 0.78, "ece": 0.06})["zone"] == "C"

def test_g5_g8():
    assert g5_rule({"A": {"bayes": 1.0, "argmax": 2.0, "threshold": 3.0}})["zone"] == "A"
    assert g5_rule({"A": {"bayes": 1.0, "argmax": 2.0, "threshold": 0.5}})["zone"] == "C"
    assert g8_rule({"acc_answerable": 0.8, "abstain_unanswerable": 0.0}, {"acc_answerable": 0.8, "abstain_unanswerable": 0.15})["zone"] == "A"
    assert g8_rule({"acc_answerable": 0.8, "abstain_unanswerable": 0.0}, {"acc_answerable": 0.8, "abstain_unanswerable": 0.05})["zone"] == "C"

def test_g5_empty_is_C():
    assert g5_rule({})["zone"] == "C"


def test_g3_without_constraints_is_C():
    from scripts.gates.rules import g3_rule
    r = g3_rule({"massive": {"coherence": {"incoherence_before": 0.1, "incoherence_after": 0.02, "acc_after": 0.9}, "acc": 0.9},
                 "civil": {"coherence": {}, "acc": 0.9}})
    assert r["zone"] == "C" and r["detail"]["civil"] is False


def test_g9_weak_backbone_gains_more():
    from scripts.gates.rules import g9_rule
    weak = _curve({"B0@0": 0.50, "B2@1000": 0.65}, {"B0@0": 0.1, "B2@1000": 0.1})
    strong = _curve({"B0@0": 0.80, "B2@1000": 0.84}, {"B0@0": 0.1, "B2@1000": 0.1})
    assert g9_rule({"plumb": weak, "massive": weak, "clinc": weak}, {"plumb": strong, "massive": strong, "clinc": strong})["zone"] == "A"
    assert g9_rule({"plumb": weak, "massive": strong, "clinc": strong}, {"plumb": strong, "massive": weak, "clinc": weak})["zone"] == "C"
    assert g9_rule({"plumb": weak}, {"plumb": strong})["zone"] == "C"   # < 3 dominios: no evaluable
