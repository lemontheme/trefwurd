import pytest

from trefwurd import cst


@pytest.mark.parametrize("lhs_pair, l_subsumes_r",
                         [(("*ge*apt", "*ge*t"), True),
                          (("*ge*aptfr", "*ge*t"), False),
                          (("*dden", "*den"), True),
                          (("*ten", "*den"), False),
                          (("be*aaid", "*aaid"), True),
                          (("*ge*t", "*"), True),
                          (("*eid", "b*eid"), False),
                          (("*", "*ge*t"), False),
                          (("ge*d", "*ge*d"), False),
                          (("*t", "*t"), False)])
def test_rule_subsumption(lhs_pair, l_subsumes_r):
    w1, w2 = lhs_pair
    assert cst.subsumes(w1, w2) == l_subsumes_r


def test_derive_new_rule():

    from difflib import SequenceMatcher
    from trefwurd.cst import Rule, RuleTemplate, regexify_pattern_to_match
    import re

    template = RuleTemplate("ge*d", "*en")
    rule = Rule.from_template(template)
    result = rule.apply("gelegd")
    print(result.match)
    print(result.rule)
    print(result.result)
    print(result.rule.rhs)

    lemma = "uitleggen"
    incorrect_lemma = "uitlegen"

    m_lemma = re.match(regexify_pattern_to_match(rule.rhs), lemma)
    print(m_lemma.span(1))

    diff = SequenceMatcher(None, incorrect_lemma, lemma)
    for tag, s1, e1, s2, e2 in diff.get_opcodes():
        if tag == "insert":
            print(tag, incorrect_lemma[s1:e1], "->", lemma[s2:e2])
