import pytest
from dec.schema import DecisionItem, Option
from dec.eval.unanswerable import make_unanswerable, evidence_sentence, build_g8_items, NoLexicalEvidence


def mk(state, instructions="When will the 44.39 credit be applied?", gold="april_bill", family="trap", id="p"):
    return DecisionItem(id=id, dataset="plumb", state_id="s", state=state, qtype="choice", instructions=instructions,
                        options=[Option("april_bill", "On the April bill"), Option("may_bill", "On the May bill")],
                        gold=gold, question_key="q", family=family)


def test_removes_the_evidence_sentence():
    it = mk("Account opened in 2019. The credit of 44.39 will be applied on the April bill. Contact support for more.")
    idx, sent = evidence_sentence(it)
    assert "44.39" in sent
    un = make_unanswerable(it)
    assert "44.39" not in un.state and un.meta["unanswerable"] and un.id.endswith(":noev") and un.meta["strict"]


def mk_bill(state):
    """Opción correcta con un único término "bill" (sin descripción)."""
    return DecisionItem(id="b", dataset="plumb", state_id="s", state=state, qtype="choice",
                        instructions="When will the credit be applied?",
                        options=[Option("bill", ""), Option("other", "Something else")],
                        gold="bill", question_key="q", family="trap")


def test_strict_true_when_gold_term_only_sentence_final_in_removed():
    # pre-fix: "bill." != "bill" -> strict False (bug); post-fix: True
    it = mk_bill("Account opened in 2019. The credit will be applied on the April bill. Contact support for more.")
    un = make_unanswerable(it)
    assert "bill." in un.meta["removed"]
    assert un.meta["strict"] is True


def test_strict_false_when_gold_term_appears_in_kept_text_with_period():
    # pre-fix: "bill" (removed, mid-sentence) vs "bill." (kept, sentence-final) -> strict True (bug); post-fix: False
    it = mk_bill("Account opened in 2019. The credit will be applied to the bill next month. Contact support about the May bill.")
    un = make_unanswerable(it)
    assert "bill next" in un.meta["removed"]
    assert un.meta["strict"] is False


def test_no_lexical_evidence_raises():
    it = mk("Foo bar baz. Qux quux corge. Grault garply waldo.")
    with pytest.raises(NoLexicalEvidence, match="sin evidencia léxica"):
        evidence_sentence(it)


def test_builder_skips_items_without_lexical_evidence():
    good = mk("Account opened in 2019. The credit of 44.39 will be applied on the April bill. Contact support for more.", id="good")
    bad = mk("Foo bar baz. Qux quux corge. Grault garply waldo.", id="bad")
    out = build_g8_items([bad, good])
    assert [o.meta["source_id"] for o in out] == ["good"]


def test_single_sentence_state_raises():
    it = mk("The credit of 44.39 will be applied on the April bill.")
    with pytest.raises(ValueError):
        make_unanswerable(it)
