"""Explainability layer: classic, dependency-light ML (NumPy only, no torch)
that turns the retrieval pipeline from a black box into something a user
can *see*:

* :mod:`~cognivore.ml.projection` / :mod:`~cognivore.ml.clustering` /
  :mod:`~cognivore.ml.topics` -- a 2-D "knowledge map" of every chunk
  (PCA), grouped into topics (k-means++ with the number of topics chosen by
  silhouette score) and named by class-based TF-IDF keywords.
* :mod:`~cognivore.ml.grounding` -- how well each sentence of an answer is
  supported by the passages it was retrieved from.
* :mod:`~cognivore.ml.gaps` -- questions the knowledge base could not
  answer, remembered and grouped so the owner knows what to add.
* :mod:`~cognivore.ml.insight` -- assembles all of the above for one chat
  turn.
"""

from cognivore.ml.gaps import GapRecord, KnowledgeGapTracker
from cognivore.ml.grounding import GroundingReport, SentenceSupport, assess_grounding
from cognivore.ml.insight import TurnInsight, build_turn_insight, retrieval_confidence
from cognivore.ml.knowledge_map import KnowledgeMap, KnowledgeMapBuilder, MapCluster, MapPoint
from cognivore.ml.lang import detect_language

__all__ = [
    "GapRecord",
    "GroundingReport",
    "KnowledgeGapTracker",
    "KnowledgeMap",
    "KnowledgeMapBuilder",
    "MapCluster",
    "MapPoint",
    "SentenceSupport",
    "TurnInsight",
    "assess_grounding",
    "build_turn_insight",
    "detect_language",
    "retrieval_confidence",
]
