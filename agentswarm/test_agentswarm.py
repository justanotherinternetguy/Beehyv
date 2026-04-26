from pathlib import Path

from agentswarm import KeywordRetriever, PaperExpertAgent, SwarmOrchestrator, load_paper


PAPER_PATH = Path(__file__).parent.parent / "data" / "cleaned_json" / "bert_cleaned.json"


class StubLLM:
    def complete(self, messages):
        return f"Stubbed paper expert answer: {messages[-1]['content'][:80]}"


def test_load_paper_extracts_sections():
    paper = load_paper(PAPER_PATH)

    assert paper.paper_id == "bert"
    assert paper.title.startswith("BERT:")
    assert len(paper.chunks) > 100
    assert any(chunk.section == "Pre-training BERT" for chunk in paper.chunks)


def test_retriever_finds_masked_language_modeling():
    paper = load_paper(PAPER_PATH)
    retriever = KeywordRetriever([paper])

    results = retriever.search("masked language model pre-training", paper_id="bert", top_k=3)

    assert results
    assert any("mask" in result.chunk.text.lower() for result in results)


def test_orchestrator_returns_grounded_synthesis():
    paper = load_paper(PAPER_PATH)
    retriever = KeywordRetriever([paper])
    agent = PaperExpertAgent(paper, retriever, llm=StubLLM())
    orchestrator = SwarmOrchestrator([agent])

    blackboard = orchestrator.run("What is next sentence prediction?")

    assert blackboard.claims
    assert blackboard.synthesis is not None
    assert "bert" in blackboard.synthesis.answer
    assert blackboard.synthesis.citations
