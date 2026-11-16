from core.generator import AnswerGenerator, build_prompt, extractive_answer


def test_extractive_answer_quotes_the_relevant_sentence(mini_corpus, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    passages = [(mini_corpus[4], 3.1), (mini_corpus[3], 1.2)]
    generated = AnswerGenerator().generate("Does clarithromycin inhibit CYP3A4?", passages)
    assert generated.mode == "extractive"
    assert "Clarithromycin inhibits CYP3A4." in generated.text
    assert "[1]" in generated.text and "[2]" in generated.text


def test_extractive_answer_contains_only_label_text(mini_corpus):
    passages = [(mini_corpus[0], 1.0)]
    answer = extractive_answer("nitrates with sildenafil", passages)
    assert answer.replace(" [1]", "") in mini_corpus[0].content


def test_prompt_numbers_passages_and_includes_graph_context(mini_corpus):
    prompt = build_prompt("q?", [(mini_corpus[0], 1.0), (mini_corpus[1], 0.5)], ["(a) -[INHIBITS]-> (b)"])
    assert "[1] Sildenafil | CONTRAINDICATIONS" in prompt
    assert "[2] Sildenafil | PHARMACOKINETICS" in prompt
    assert "(a) -[INHIBITS]-> (b)" in prompt
