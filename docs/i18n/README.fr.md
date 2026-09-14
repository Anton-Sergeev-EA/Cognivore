# Cognivore

**Un framework d'agent multimodal local-first, sans PyTorch.** Raisonnement
LLM + RAG, avec un index vectoriel écrit en C++ et une interface de chat
web -- tout fonctionne sur un simple ordinateur portable équipé d'un CPU,
rien n'a besoin de quitter votre machine.

[![CI](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anton-Sergeev-EA/cognivore/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](../../pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-black)](https://github.com/astral-sh/ruff)

🌐 **Lisez ceci dans une autre langue :** [English](../../README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Français](README.fr.md) | [Italiano](README.it.md) | [Español](README.es.md) | [中文](README.zh.md) | [日本語](README.ja.md) | [हिन्दी](README.hi.md)

Cognivore est un framework d'agent de style ReAct associant
retrieval-augmented generation et prise en charge d'outils multimodaux
(transcription audio, analyse de scènes vidéo), conçu spécifiquement pour
fonctionner entièrement sur une machine équipée d'un simple CPU, sans la
moindre dépendance à PyTorch dans toute la pile logicielle. Son index de
recherche est un cœur écrit à la main en C++ (SIMD AVX2, recherche exacte
parallélisée via OpenMP, et un graphe NSW approximatif construit de zéro),
exposé à Python via pybind11, avec un repli en NumPy pur afin que
`pip install` n'échoue jamais faute de compilateur C++.

```
you> What does the ingested spec say about the retry policy, and what's 15% of that timeout in ms?

  [tool] search_knowledge_base({"query": "retry policy timeout"})
  [tool] calculator({"expression": "5000 * 0.15"})

cognivore> The spec (spec.md) sets a 5000ms timeout with exponential
backoff. 15% of that is 750ms.
```

## Pourquoi ce projet existe

La plupart des projets de portfolio « agent IA » ne sont qu'une fine
couche autour d'un appel d'API. Ce projet montre l'inverse : un framework
où les parties intéressantes (une véritable structure de données ANN, un
protocole d'appel d'outils indépendant du fournisseur, un pipeline de
recherche hybride, une dégradation propre en l'absence de dépendances
optionnelles) sont implémentées plutôt qu'importées.

## Fonctionnalités

- **Boucle d'agent ReAct** (Thought → Action → Observation) qui
  fonctionne avec *n'importe quel* modèle local instruction-tuned, et pas
  seulement avec des modèles fine-tunés pour un format spécifique
  d'appel de fonctions -- voir
  [docs/architecture.md](../architecture.md#why-a-react-loop-instead-of-native-function-calling).
- **Index vectoriel natif en C++** (`native/vector_index.cpp`) : un
  `FlatIndex` exact (produit scalaire AVX2/FMA, scan parallélisé via
  OpenMP) et un `NSWIndex` approximatif (un graphe Navigable Small World
  monocouche construit de zéro -- insertion, élagage des voisins,
  recherche par faisceau gloutonne), tous deux exposés à Python via une
  extension pybind11 écrite à la main, accompagnée d'un stub de type
  `.pyi`. Bascule automatiquement vers un index en NumPy pur si aucun
  compilateur C++ n'est présent au moment de l'installation.
- **RAG hybride** : découpage en chunks via un splitter récursif,
  embeddings sémantiques `fastembed` (ONNX, sans PyTorch) avec un repli
  sans téléchargement basé sur le hashing-trick, recherche hybride
  vecteur + BM25.
- **Outils multimodaux** : calculatrice sûre (basée sur l'AST, sans
  `eval`), recherche dans la base de connaissances, transcription audio +
  segmentation approximative par locuteur (`faster-whisper`), et
  détection de scènes vidéo + OCR (OpenCV) -- tout cela sur CPU
  uniquement, sans PyTorch.
- **Backend LLM interchangeable** : inférence locale au format GGUF via
  `llama-cpp-python`, ou un `FakeLLMBackend` déterministe et sans
  dépendance qui exerce exactement le même chemin de code d'appel
  d'outils sans aucun téléchargement -- c'est sur lui que s'appuient la
  suite de tests et la CI.
- **Interface web** : FastAPI + streaming SSE + une interface de chat en
  JS/HTML/CSS pur (sans étape de build, sans framework) avec
  glisser-déposer de fichiers/audio/vidéo, indicateurs en direct de
  « réflexion »/connexion, thèmes sombre/clair/aurora, et localisation en
  anglais, russe, allemand, français, italien, espagnol, chinois
  simplifié, japonais et hindi.
- Suite de tests complète, lint+formatage ruff, mypy (en mode
  quasi-strict, incluant un stub pour l'extension native), et une matrice
  de CI multi-OS/multi-Python.

## Démarrage rapide

```bash
git clone https://github.com/Anton-Sergeev-EA/cognivore.git
cd cognivore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .                 # compile l'extension native si un compilateur C++17 est disponible

cognivore chat                   # chat interactif, mode démo hors ligne par défaut
cognivore ingest ./docs          # indexe un dossier de fichiers markdown/texte
cognivore serve                  # interface de chat web sur http://127.0.0.1:8420
```

Par défaut, aucune LLM n'est téléchargée et aucun accès réseau n'est
requis : l'agent fonctionne avec `FakeLLMBackend`, un petit backend
déterministe qui exerce néanmoins la véritable boucle d'appel d'outils
(essayez `What is 12 * 7?` ou `search the knowledge base for ...` après
avoir fait un `ingest`). Pour utiliser une véritable LLM locale :

```bash
pip install -e ".[llm]"
# téléchargez par exemple https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
cp .env.example .env
echo 'COGNIVORE_LLM_MODEL_PATH=/path/to/model.gguf' >> .env
cognivore chat
```

Ou, plus simple et sans toolchain C++ du tout, pointez plutôt vers
[Ollama](https://ollama.com) -- `COGNIVORE_LLM_PROVIDER=auto` (la valeur
par défaut) essaie d'abord un serveur Ollama tournant localement avant de
basculer vers un chemin GGUF ou vers `FakeLLMBackend` :

```bash
ollama pull qwen2.5:3b   # n'importe quel modèle instruction-tuned fonctionne
cognivore chat           # détecte automatiquement le serveur Ollama en cours d'exécution
```

Les outils audio/vidéo nécessitent leurs propres extras :
`pip install -e ".[audio,video]"` (ou `.[all]` pour tout avoir, GGUF
compris). Voir `.env.example` pour la liste complète des réglages.

### Docker

L'image est construite en plusieurs étapes (compile l'extension native en
C++, puis se débarrasse du compilateur pour obtenir une image d'exécution
allégée), s'exécute avec un utilisateur non root, et embarque un
`HEALTHCHECK` -- construite une seule fois grâce à la mise en cache des
couches Docker de GitHub Actions et vérifiée de bout en bout (le serveur
répond réellement à `/api/health`, le conteneur se signale `healthy`, et
s'exécute bien en non-root) à chaque push, donc elle ne se contente pas
« de compiler », elle est vérifiée.

**Le chemin le plus simple -- sans Python, sans installation d'Ollama,
fonctionne de la même façon sur Windows, macOS et Linux :**

```bash
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:3b   # une seule fois, ~2 Go
```

Ouvrez ensuite <http://127.0.0.1:8420>. `docker-compose.yml` fait tourner
Ollama *lui aussi* dans son propre conteneur, si bien qu'il n'y a rien à
installer sur l'hôte à part Docker lui-même ; Cognivore l'atteint sur le
réseau du compose par nom de service (`http://ollama:11434`), ce qui
évite entièrement les différences habituelles de mise en réseau hôte
entre Windows/macOS/Linux. Sans ce `ollama pull` réalisé une seule fois,
Cognivore démarre et fonctionne quand même très bien -- il retombe
simplement dans le mode démo hors ligne `FakeLLMBackend` jusqu'à ce qu'un
modèle soit disponible.

**Vous avez déjà Ollama qui tourne sur l'hôte, ou vous voulez un seul
conteneur ?**

```bash
docker build -t cognivore .
docker run -d -p 8420:8420 -v cognivore-data:/data \
  -e COGNIVORE_OLLAMA_HOST=http://host.docker.internal:11434 \
  --add-host=host.docker.internal:host-gateway \
  cognivore
```

`host.docker.internal` est fourni automatiquement par Docker Desktop
(Windows/macOS) ; le `--add-host` explicite ci-dessus est ce qui permet à
la même commande de fonctionner aussi sur Linux classique, où ce nom ne
se résout sinon pas.

L'image embarque les outils audio/vidéo (`faster-whisper`, OpenCV) mais
*pas* `llama-cpp-python` -- elle communique avec Ollama en HTTP simple
pour la LLM plutôt que de charger un fichier GGUF directement dans le
processus, et ce délibérément, car llama-cpp-python n'a pas de wheel
précompilée pour chaque plateforme et nécessite un compilateur que
l'étape d'exécution n'embarque pas. Vous voulez malgré tout une inférence
GGUF in-process dans le conteneur ? Ajoutez `build-essential` à l'étape
finale et faites repointer son `pip install` vers l'extra `[all]`.

**Image préconstruite (aucune étape de build du tout) :** les versions
taguées sont publiées en multi-architecture (amd64 + arm64 -- Apple
Silicon et Raspberry Pi compris) sur GHCR par
[`.github/workflows/docker-publish.yml`](../../.github/workflows/docker-publish.yml) :

```bash
docker pull ghcr.io/anton-sergeev-ea/cognivore:latest
```

## Utilisation

**CLI** (`cognivore --help` pour la liste complète) :

```bash
cognivore chat                              # REPL interactif
cognivore ingest ./docs                     # indexe récursivement les .txt/.md/.markdown/.rst
cognivore serve --host 0.0.0.0 --port 8420  # interface web + API REST/SSE
cognivore bench                             # benchmarks de construction/recherche d'index (voir plus bas)
```

**Interface web** (`cognivore serve`, puis ouvrez l'URL affichée) : une
interface de chat avec streaming de tokens en direct, une zone de
glisser-déposer pour l'ingestion de fichiers `.txt`/`.md` ainsi que de
fichiers audio/vidéo, un sélecteur de langue (9 langues), trois thèmes
(sombre/clair/aurora), et une vue de trace de chaque appel d'outil
effectué par l'agent pour une réponse donnée (cliquez sur une étape pour
voir le résultat complet, non tronqué, de l'outil).

**API REST/SSE**, une fois `cognivore serve` lancé :

```bash
curl http://127.0.0.1:8420/api/health

curl -X POST http://127.0.0.1:8420/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is 12 * 7?"}'

curl -N "http://127.0.0.1:8420/api/chat/stream?message=Summarize+the+ingested+docs"

curl -X POST http://127.0.0.1:8420/api/ingest/file -F "file=@./notes.md"
```

**En tant que bibliothèque**, plutôt que via la CLI ou l'API (voir
`examples/`) :

```python
from cognivore.bootstrap import build_agent
from cognivore.config import get_settings

agent = build_agent(get_settings())
result = agent.run("What's 15% of 5000?")
print(result.answer)
```

## Benchmarks

Chiffres issus de `python benchmarks/bench_index.py` sur une machine à 2
cœurs de classe CI (l'image du `Dockerfile` est portable ; votre
ordinateur portable réel dispose presque certainement de plus de cœurs,
ce qui a son importance -- voir plus bas). Des vecteurs à 384 dimensions,
aléatoires et uniformément distribués, ce qui est proche du *pire cas*
pour la recherche approximative (aucune structure de clusters réelle,
donc les voisins « les plus proches » ne le sont que marginalement plus
que des voisins pris au hasard) ; les embeddings réels obtiennent un
recall nettement meilleur pour le même `ef`.

**Temps de construction** (3 000 vecteurs) -- c'est ici que l'avantage de
l'extension native est sans ambiguïté :

| Index | Temps de construction | vs. repli NumPy |
|---|---|---|
| `FlatIndexPy` (NumPy, `.add()` en boucle) | 0.615s | 1x |
| `FlatIndex` (C++) | 0.027s | **~23x plus rapide** |

**Le recall face à la vitesse est un curseur réglable, pas un chiffre
unique** (`NSWIndex`, n=3 000) :

| `ef` | ms/requête | Recall@10 |
|---|---|---|
| 50  | 0.24 | 0.64 |
| 150 (par défaut) | 0.46 | 0.93 |
| 400 | 0.78 | 1.00 |

**Latence de recherche en fonction de la taille de la collection** -- un
scan linéaire exact (même accéléré par AVX2+OpenMP) reste *tout à fait
correct* jusqu'à ce que la collection devienne suffisamment grande pour
que le scan lui-même devienne le goulot d'étranglement ; c'est à ce point
de croisement qu'un index en graphe approximatif est censé commencer à
l'emporter :

| n | `FlatIndex` ms/requête | `NSWIndex` ms/requête (ef=150) | Accélération |
|---|---|---|---|
| 1 000 | 0.036 | 0.240 | 0.2x (la force brute gagne -- collection trop petite) |
| 10 000 | 0.340 | 0.798 | 0.4x |
| 50 000 | 1.561 | 1.291 | 1.2x |

Lisez ce tableau pour ce qu'il dit réellement, pas pour ce qui ferait une
plus belle histoire : à ces tailles, sur 2 cœurs, un scan brute-force
SIMD+parallèle est *compétitif, voire plus rapide,* que l'index
approximatif construit de zéro. C'est un comportement ANN réel et bien
documenté -- un graphe NSW monocouche écrit à la main a des frais
généraux par étape sensiblement plus élevés (opérations de tas, accès
mémoire aléatoires à travers le graphe, tenue à jour de l'ensemble des
nœuds visités) qu'un scan linéaire favorable au cache, et le HNSW
multicouche (l'approche de Faiss/hnswlib, pas encore implémentée ici --
voir la roadmap) existe précisément pour élargir cet écart à grande
échelle. Le constat honnête : l'index ANN de ce projet démontre
correctement *la structure de données et l'algorithme* (voir le test de
régression de recall dans `tests/test_index.py`), et le point de
croisement où il devient rentable évolue selon le nombre de cœurs, `ef`,
et le degré de clustering de vos embeddings réels -- ce n'est pas une
affirmation universelle du type « toujours plus rapide », et ce README ne
prétendra pas le contraire.

## Tests

```bash
pip install -e ".[dev]"
pytest --cov                 # 71 tests : sécurité de la calculatrice, chunking,
                              # parité entre l'index natif et l'index Python, recall NSW,
                              # boucle de l'agent, stockage RAG, endpoints FastAPI
ruff check . && ruff format --check .
mypy -p cognivore
```

Chacune des vérifications ci-dessus est exactement ce que lance la CI
(`.github/workflows/ci.yml`), sur Ubuntu/macOS/Windows et Python
3.10-3.12 ; les tests dépendant uniquement de l'extension native se
sautent eux-mêmes (plutôt que d'échouer) sur une configuration de la
matrice où la toolchain C++ n'est pas disponible, reproduisant ainsi le
même comportement de repli que le paquet lui-même.

## Structure du projet

Voir [docs/architecture.md](../architecture.md) pour un schéma et le
raisonnement derrière les deux décisions de conception les plus
importantes (une boucle ReAct textuelle plutôt qu'un function calling
spécifique à un fournisseur, et un index écrit à la main plutôt que
Faiss/hnswlib).

```
native/            C++ vector index core + pybind11 bindings
src/cognivore/
  index/            native-vs-fallback index selection
  rag/              chunking, embeddings, hybrid document store
  agent/            ReAct loop, memory, prompt/parsing
  tools/            calculator, RAG search, audio, video
  llm/              llama.cpp backend + FakeLLMBackend
  media/            faster-whisper / OpenCV wrappers
  web/              FastAPI app + static chat UI
  cli.py            cognivore chat|ingest|serve|bench
benchmarks/         standalone scripts for the numbers above
examples/           minimal library-usage scripts
tests/              pytest suite (71 tests)
```

## Licence

MIT – voir [LICENSE](../../LICENSE).

## Contact

Créé par **Sergeev Anton** ([GitHub](https://github.com/Anton-Sergeev-EA), [avsergeev1981@gmail.com](mailto:avsergeev1981@gmail.com)). Les contributions sont les bienvenues – voir [CONTRIBUTING.md](../../CONTRIBUTING.md).
