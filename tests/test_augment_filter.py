from augment import generate_scenarios, filter


def test_mock_generate(tmp_path):
    ex = generate_scenarios.mock_generate(5)
    assert len(ex) == 5
    for e in ex:
        assert 'prompt' in e and 'response' in e


def test_semantic_dedupe():
    records = [ {'response':'Hello there'}, {'response':'Hello there!'}, {'response':'Different'} ]
    filtered = filter.semantic_dedupe(records, threshold=0.5)
    # expect at least 2 unique
    assert len(filtered) >= 2
    for r in filtered:
        assert 'response' in r


def test_persona_check():
    assert filter.persona_check({'response':'Hi'})
    assert not filter.persona_check({'response':''})
