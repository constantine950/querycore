# QueryCore — Product Requirements Document

**Author:** Majorga  
**Date:** 2026-06-08  
**Status:** Draft

---

## 1. Overview

QueryCore is a text indexing and retrieval engine built from scratch. It accepts a corpus of documents, indexes them using an inverted index structure, and retrieves ranked results for user queries. The system supports relevance scoring (TF-IDF), fuzzy matching, phrase search, autocomplete, and filtering.

This is a portfolio project intended to demonstrate algorithm design and systems engineering skills.

---

## 2. Problem Statement

General-purpose search (e.g. Elasticsearch, Solr) is a black box. QueryCore exists to show that a working search engine — with real ranking, real fuzzy matching, and a real API — can be built from first principles in Python, without external search libraries.

---

## 3. Goals

- Index a corpus of text documents into an inverted index
- Retrieve and rank results using TF-IDF scoring
- Support phrase search, fuzzy search (typo tolerance), and autocomplete
- Expose all functionality via a REST API
- Visualize search analytics in a React frontend
- Containerize the full stack with Docker

---

## 4. Non-Goals

- Not a replacement for production search infrastructure
- No distributed/sharded indexing
- No machine learning re-ranking (e.g. BERT, BM25+ neural)
- No user authentication or multi-tenancy

---

## 5. Core Concepts

### 5.1 Indexing

Indexing is the process of transforming raw documents into a data structure optimized for fast retrieval.

**Pipeline:**

```
Raw Text → Tokenization → Stop Word Removal → Stemming → Inverted Index
```

Each document is broken into tokens. Tokens are normalized (lowercased, stemmed), filtered (stop words removed), and then stored in an inverted index mapping each token to the list of documents it appears in, along with positional and frequency data.

### 5.2 Inverted Index

The core data structure of any search engine. A dictionary where:

- **Key:** a token (e.g. `"search"`)
- **Value:** a posting list — the ordered list of document IDs that contain that token, with metadata (term frequency, positions)

```
"engine" → [(doc_1, tf=3, pos=[4,11,22]), (doc_5, tf=1, pos=[7])]
"search" → [(doc_1, tf=2, pos=[1,9]),    (doc_3, tf=4, pos=[0,3,5,12])]
```

Lookup is O(1) for the token. Intersection of posting lists gives AND queries.

### 5.3 Tokenization

Splitting raw text into individual searchable units (tokens). QueryCore uses whitespace + punctuation splitting, lowercasing, and Unicode normalization.

```
"Search Engines are Fast!" → ["search", "engines", "are", "fast"]
```

### 5.4 Stop Words

Common words (the, is, are, a, of...) that appear in almost every document and contribute no discriminating signal to retrieval. Removing them reduces index size and improves ranking quality.

### 5.5 Stemming

Reducing words to their root form so that "running", "runs", and "ran" all map to "run". QueryCore uses Porter stemming. This improves recall at the cost of some precision.

```
"running" → "run"
"searches" → "search"
"indexed"  → "index"
```

### 5.6 TF-IDF Ranking

Term Frequency–Inverse Document Frequency. A numerical statistic reflecting how important a word is to a document relative to the entire corpus.

- **TF (Term Frequency):** how often the term appears in the document  
  `TF(t, d) = count(t in d) / total_tokens(d)`

- **IDF (Inverse Document Frequency):** how rare the term is across all documents  
  `IDF(t) = log(N / df(t))`  
  where N = total documents, df(t) = documents containing the term

- **TF-IDF score:**  
  `score(t, d) = TF(t, d) × IDF(t)`

For a multi-term query, scores are summed across all query terms.

### 5.7 Fuzzy Search

Typo-tolerant search using edit distance (Levenshtein distance). A query token matches an index token if the edit distance between them is within a threshold (default: 2).

```
"serach" → matches "search" (edit distance = 1)
"engne"  → matches "engine" (edit distance = 2)
```

### 5.8 Phrase Matching

Exact multi-word phrase search. Requires matched tokens to appear consecutively in the document, enforced using positional data stored in the posting list.

```
"search engine" → tokens ["search", "engine"] must appear at positions n and n+1
```

### 5.9 Autocomplete

Prefix-based query suggestion using a trie data structure built from all indexed terms.

```
"sea" → ["search", "searcher", "season"]
```

---

## 6. System Components

| Component              | Responsibility                               |
| ---------------------- | -------------------------------------------- |
| `tokenizer.py`         | Split text into tokens                       |
| `preprocessor.py`      | Stop words, stemming, normalization          |
| `inverted_index.py`    | Build and query the index                    |
| `query_parser.py`      | Parse raw user query into structured form    |
| `retrieval.py`         | Fetch candidate documents from index         |
| `ranking.py`           | Score and sort candidates via TF-IDF         |
| `phrase_match.py`      | Enforce positional constraints for phrases   |
| `fuzzy_search.py`      | Edit distance matching                       |
| `autocomplete.py`      | Trie-based prefix suggestions                |
| `filters.py`           | Category/date/metadata filtering             |
| `highlighter.py`       | Mark matching terms in result snippets       |
| `paginator.py`         | Slice results into pages                     |
| `tracker.py`           | Log query frequency and latency              |
| `app.py` / `routes.py` | REST API (FastAPI)                           |
| React frontend         | Search UI, results page, analytics dashboard |

---

## 7. API Surface (High-Level)

| Method | Endpoint                    | Description              |
| ------ | --------------------------- | ------------------------ |
| GET    | `/search?q=&page=&filters=` | Full-text search         |
| GET    | `/autocomplete?q=`          | Query suggestions        |
| POST   | `/index`                    | Add document(s) to index |
| GET    | `/analytics/top`            | Top queries              |
| GET    | `/health`                   | Health check             |

---

## 8. Dataset

A corpus of plain-text documents. Options: Wikipedia article dumps, news headlines, or a custom JSON dataset. Documents will be stored in `data/sample_documents.json` as:

```json
[
  {
    "id": "doc_1",
    "title": "...",
    "body": "...",
    "category": "...",
    "date": "..."
  }
]
```

---

## 9. Deliverables

- [ ] Working inverted index with positional data
- [ ] TF-IDF ranking
- [ ] Fuzzy search (Levenshtein)
- [ ] Autocomplete (trie)
- [ ] REST API (FastAPI)
- [ ] React frontend (search + analytics)
- [ ] Dockerized full stack
- [ ] Documentation (`indexing-strategy.md`, `ranking-explanation.md`)

---

## 10. Success Criteria

1. Index 1,000+ documents in under 5 seconds
2. Return ranked results for a query in under 100ms
3. Correctly handle typos (edit distance ≤ 2)
4. Return autocomplete suggestions for partial queries
5. Display highlighted snippets in the UI
6. Run end-to-end via `docker-compose up`
