"""Language-aware tokenization shared by BM25, the hashing embedder and the
ML insight layer.

A plain ``str.split()`` is fine for English and Russian, but it silently
breaks two things that matter for a product aimed at more than one market:

* **Chinese/Japanese text has no spaces**, so ``split()`` turns a whole
  sentence into one "word". BM25 and the hashing embedder then only match a
  query that is character-for-character identical to an entire chunk --
  in practice every Chinese query scored exactly 0 against every chunk.
* **Punctuation sticks to words** ("policy," vs "policy"), which quietly
  lowers lexical recall in every language.

The fix is the same one Lucene's ``CJKAnalyzer`` uses: runs of CJK
characters become overlapping character bigrams (a single character is
kept as a unigram), everything else becomes lowercase alphanumeric words.
"""

from __future__ import annotations

import re

# CJK Unified Ideographs (+ extension A, compatibility ideographs) and
# Japanese kana. Kept as one class so mixed zh/ja text tokenizes uniformly.
_CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿"
# Combining marks that belong inside a word but are not "word" characters
# to Python's regex engine: Latin diacritics and the vowel signs/viramas of
# Indic scripts (Devanagari..Sinhala, minus the danda punctuation marks
# U+0964/U+0965). Without them "रिफ़ंड" fell apart into bare consonants.
_MARKS = r"\u0300-\u036f\u0900-\u0963\u0966-\u0dff"
# The second alternative excludes CJK explicitly: ideographs are "word"
# characters too, so "API限额" would otherwise come out as a single token.
_TOKEN_RE = re.compile(rf"[{_CJK}]+|(?:[^\W_{_CJK}]|[{_MARKS}])+", re.UNICODE)
_CJK_RUN_RE = re.compile(rf"^[{_CJK}]+$")


def is_cjk_run(token: str) -> bool:
    return bool(_CJK_RUN_RE.match(token))


def tokenize(text: str) -> list[str]:
    """Splits ``text`` into lowercase lexical tokens (see module docstring).

    >>> tokenize("Refund policy, 14 days!")
    ['refund', 'policy', '14', 'days']
    >>> tokenize("退款政策")
    ['退款', '款政', '政策']
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text.lower()):
        piece = match.group(0)
        if is_cjk_run(piece):
            if len(piece) == 1:
                tokens.append(piece)
            else:
                tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece)
    return tokens


# Very small, high-frequency function-word lists. Only used where a token
# carrying no topical meaning would actively mislead (cluster labels, query
# coverage in the insight layer) -- BM25 already down-weights them via IDF.
STOPWORDS: frozenset[str] = frozenset(
    # English
    "a an the and or but if of to in on at by for with from as is are was were be been "
    "being it its this that these those i you he she we they me my your our their what "
    "which who whom how when where why can could should would will do does did not no "
    "yes so than then there here about into over also any all per up out s t".split()
    # Russian
    + "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по "
    "только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если "
    "уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей "
    "может они тут где есть надо ней для мы тебя их чем была сам чтобы без будто чего раз "
    "тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом "
    "один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец "
    "два об другой хоть после над больше тот через эти нас про всего них какая много разве "
    "три эту моя впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им "
    "более всегда конечно всю между это сколько какие каков какая".split()
    # Spanish
    + "el la los las un una unos unas de del al y o que en es son por para con "
    "sin se su sus lo como más qué cuál cuáles cuánto cuánta cómo dónde cuándo "
    "hay tiene tienen este esta estos estas ese esa usted ustedes le les mi".split()
    # French
    + "le la les un une des du de et ou que qui en est sont par pour avec sans se "
    "sa son ses ce cette ces au aux ne pas plus quel quelle quels quelles combien "
    "comment où quand il elle ils elles vous nous on y a font fait".split()
    # German
    + "der die das den dem des ein eine einer eines einem und oder ist sind von "
    "zu mit für auf im in am an aus bei nicht wie was welche welcher welches "
    "wann wo viel viele es sie ich wir ihr hat haben wird werden gibt lautet".split()
    # Italian
    + "il lo la i gli le un uno una di del della dei delle e o che è sono per con "
    "su tra fra non più come quale quali quanto quanta dove quando ci si "
    "ha hanno fa al alla".split()
    # Hindi
    + "का के की को में से पर और या है हैं था थे क्या कैसे कब कहाँ कौन "
    "कितना कितनी कितने यह वह ये वे इस उस एक भी तो ही लिए आप हम".split()
    # Japanese particles and copulas (kana bigrams/unigrams from tokenize())
    + "は が を に の で と も へ か な ね よ です ます まし から まで ですか "
    "どの なに なん いく くら".split()
    # Chinese: common function-character bigrams/unigrams produced by tokenize()
    + "的 了 是 在 和 有 我 你 他 她 它 们 这 那 吗 呢 吧 啊 也 都 就 与 及 或 "
    "什么 多少 怎么 如何 哪些 是否 可以 我们 你们 他们 一个 这个 那个 没有 以及".split()
)


def content_tokens(text: str) -> list[str]:
    """``tokenize`` minus stopwords and 1-char non-CJK noise."""
    return [t for t in tokenize(text) if t not in STOPWORDS and (len(t) > 1 or is_cjk_run(t))]
