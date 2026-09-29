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


# When a question is asked before any document has been added.
EMPTY_KNOWLEDGE_BASE = {
    "ru": "База знаний пока пуста. Загрузите документы, и я отвечу по ним.",
    "en": "The knowledge base is empty. Add some documents and I will answer from them.",
    "zh": "知识库目前是空的。请先上传文档，我会根据文档作答。",
    "es": "La base de conocimiento está vacía. Añade documentos y responderé a partir de ellos.",
    "hi": "नॉलेज बेस अभी खाली है। दस्तावेज़ जोड़ें, मैं उन्हीं के आधार पर जवाब दूँगा।",
    "fr": "La base de connaissances est vide. Ajoutez des documents et je répondrai à partir de ceux-ci.",
    "de": "Die Wissensbasis ist leer. Fügen Sie Dokumente hinzu, dann antworte ich auf deren Grundlage.",
    "ja": "ナレッジベースはまだ空です。ドキュメントを追加していただければ、その内容に基づいて回答します。",
    "it": "La base di conoscenza è vuota. Aggiungi dei documenti e risponderò in base a quelli.",
}


def empty_knowledge_base(question: str) -> str:
    """The "nothing has been added yet" answer in the language of ``question``."""
    return EMPTY_KNOWLEDGE_BASE.get(detect_language(question), EMPTY_KNOWLEDGE_BASE["en"])
