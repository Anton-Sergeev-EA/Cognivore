"""Answers Cognivore gives itself, without asking the model, in the nine UI
languages."""

from __future__ import annotations

from cognivore.ml.lang import detect_language

# When the knowledge base was searched for a question and has nothing on it.
NOT_IN_KNOWLEDGE_BASE = {
    "ru": "В базе знаний нет информации об этом.",
    "en": "The knowledge base has no information about this.",
    "zh": "知识库中没有这方面的信息。",
    "es": "La base de conocimiento no tiene información sobre esto.",
    "hi": "नॉलेज बेस में इसकी कोई जानकारी नहीं है।",
    "fr": "La base de connaissances ne contient aucune information à ce sujet.",
    "de": "Die Wissensbasis enthält dazu keine Informationen.",
    "ja": "ナレッジベースにはこれに関する情報がありません。",
    "it": "La base di conoscenza non contiene informazioni su questo.",
}


def not_in_knowledge_base(question: str) -> str:
    """The "no information" answer in the language of ``question``."""
    return NOT_IN_KNOWLEDGE_BASE.get(detect_language(question), NOT_IN_KNOWLEDGE_BASE["en"])
