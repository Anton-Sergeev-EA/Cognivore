# Specification review evaluation

Run from the repository root in the Cognivore development environment.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_review.py --repeat 100
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_review.py --repeat 100 --cases benchmarks/review_public_cases.json
```

The [synthetic corpus](review_cases.json) contains ten annotated pairs covering
reordering, duplicates, edited numbers, additions, removals, ambiguous
matches, renumbering and requirements in German and Russian. It serves as
a regression set. The [published cases](review_public_cases.json) are reported separately.
They consist of a NIST excerpt and a WCAG example with paraphrased requirements.

A finding is correct when its kind and one-based source lines match the
annotation. For an ambiguous finding, this metric checks the original
source line and whether the system abstained from asserting a match. The
alternative lines are checked separately by a regression test. Precision
and recall are micro-averaged across the corpus.

The line baseline uses Python's `SequenceMatcher` and pairs replaced
lines in order. The vector baseline greedily pairs lines by cosine
similarity at a threshold of 0.50. Both the vector baseline and Cognivore
use the same offline hashing embedder. Each method prints extra and
missed findings by case. Reported timing is the median per-case time
across repeated corpus runs, excluding file loading, model setup and
browser rendering. Embedding computation is included.

## Recorded results

Intel Core i5-1235U, Python 3.12.3, NumPy 2.5.3, one OpenMP and OpenBLAS
thread, 100 repeats.

| Method | Correct | Extra | Missed | Precision | Recall | Median ms/case |
|---|---:|---:|---:|---:|---:|---:|
| Line diff | 3 | 12 | 10 | 20.0% | 23.1% | 0.004 |
| Vector only | 8 | 6 | 5 | 57.1% | 61.5% | 0.188 |
| Cognivore | 13 | 0 | 0 | 100% | 100% | 0.131 |

Before the ambiguity and renumbering fixes, Cognivore returned 12 correct,
three extra and one missed finding on these same synthetic cases.

On the published excerpt, the line baseline found the single change by
position. Both vector-only and Cognivore returned an addition and a
removal, missing the intended alignment. The excerpt pairs a minimum
length requirement for subscriber-chosen memorized secrets in the older
revision with the revised requirement scoped to single-factor passwords.
This case illustrates the limits of offline hashing when wording and
scope both change. It is one excerpt pair and cannot establish accuracy
on complete specifications. [Earlier NIST revision](https://pages.nist.gov/800-63-3/sp800-63b.html#sec5) · [Later NIST revision](https://pages.nist.gov/800-63-4/sp800-63b.html#passwordver)

## Section context on published cases

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmarks/bench_review.py \
  --repeat 30 --cases benchmarks/review_public_cases.json \
  --semantic-model sentence-transformers/all-MiniLM-L6-v2
```

The benchmark loads the FastEmbed model once before timing. It compares
the same model with and without the nearest Markdown heading. Section
context can boost a candidate only when the requirement text itself
already has sufficient semantic similarity. Both baselines omit headings
from the comparison and preserve the original source line numbers.

| Method | Correct | Extra | Missed | Median ms/case |
|---|---:|---:|---:|---:|
| Line diff | 2 | 0 | 0 | 0.011 |
| Vector only | 1 | 2 | 1 | 0.236 |
| Cognivore with hashing | 1 | 2 | 1 | 0.150 |
| Cognivore with MiniLM | 1 | 1 | 1 | 3.332 |
| Cognivore with MiniLM and section context | 2 | 0 | 0 | 8.084 |

Without context, MiniLM surfaced the NIST pair as ambiguous. With
context, it matched the two lines. All methods detected that WCAG 2.2
added the minimum target-size rule while retaining the enhanced rule.
The WCAG texts in the fixture are short paraphrases of the cited
criteria. On the ten synthetic cases, each Cognivore configuration
returned all 13 annotations with zero extra findings. The published
set contains only two cases and the NIST miss was known during
development. These measurements are a regression check. A larger held-out
corpus with full sections is needed to estimate accuracy and throughput.
[WCAG 2.1 target size](https://www.w3.org/TR/WCAG21/#target-size) ·
[WCAG 2.2 minimum target size](https://www.w3.org/TR/WCAG22/#target-size-minimum)
