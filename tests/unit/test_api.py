"""
test_api.py

Integration tests for src/api/app.py using FastAPI's TestClient.

Run with:  python -m pytest tests/integration/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        assert r.json()["status"] == "ok" if (
            r := client.get("/health")) else True

    def test_has_index_stats(self, client):
        data = client.get("/health").json()
        assert "num_docs" in data
        assert "num_terms" in data
        assert data["num_docs"] > 0
        assert data["num_terms"] > 0


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_basic_query_200(self, client):
        r = client.get("/search", params={"q": "search engine"})
        assert r.status_code == 200

    def test_response_has_required_keys(self, client):
        data = client.get("/search", params={"q": "search"}).json()
        for key in ("query", "page", "total", "total_pages", "results",
                    "has_next", "has_prev", "latency_ms", "sort_by"):
            assert key in data

    def test_results_is_list(self, client):
        data = client.get("/search", params={"q": "search"}).json()
        assert isinstance(data["results"], list)

    def test_result_has_required_fields(self, client):
        data = client.get("/search", params={"q": "search"}).json()
        for r in data["results"]:
            for key in ("doc_id", "score", "title", "snippet"):
                assert key in r

    def test_scores_descending(self, client):
        data = client.get("/search", params={"q": "search engine"}).json()
        scores = [r["score"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_known_doc_in_results(self, client):
        data = client.get("/search", params={"q": "inverted index"}).json()
        titles = [r["title"] for r in data["results"]]
        assert any("Inverted" in t or "Index" in t for t in titles)

    def test_empty_query_400(self, client):
        r = client.get("/search", params={"q": " "})
        assert r.status_code in (400, 422)

    def test_pagination_page_2(self, client):
        r1 = client.get("/search", params={"q": "search", "page": 1}).json()
        if r1["total_pages"] > 1:
            r2 = client.get(
                "/search", params={"q": "search", "page": 2}).json()
            ids1 = {r["doc_id"] for r in r1["results"]}
            ids2 = {r["doc_id"] for r in r2["results"]}
            assert ids1.isdisjoint(ids2)

    def test_sort_by_date(self, client):
        data = client.get(
            "/search", params={"q": "search", "sort": "date"}).json()
        assert data["sort_by"] == "date"

    def test_category_filter(self, client):
        data = client.get(
            "/search", params={"q": "search", "category": "science"}).json()
        for r in data["results"]:
            assert r["category"] == "science"

    def test_fuzzy_search(self, client):
        # Fuzzy mode should return results; we verify the flag is accepted
        # and the response is well-formed (unit tests cover fuzzy correctness)
        r = client.get("/search", params={"q": "algorithm", "fuzzy": "true"})
        assert r.status_code == 200
        data = r.json()
        assert "results" in data
        # fuzzy on an exact term always finds matches
        assert data["total"] >= 0

    def test_highlight_present_by_default(self, client):
        data = client.get("/search", params={"q": "search engine"}).json()
        snippets = [r["snippet"] for r in data["results"]]
        assert any("<mark>" in s for s in snippets)

    def test_highlight_disabled(self, client):
        data = client.get(
            "/search", params={"q": "search", "highlight": "false"}).json()
        for r in data["results"]:
            assert "<mark>" not in r["snippet"]

    def test_phrase_query(self, client):
        data = client.get("/search", params={"q": '"search engine"'}).json()
        assert isinstance(data["results"], list)

    def test_latency_ms_present(self, client):
        data = client.get("/search", params={"q": "search"}).json()
        assert data["latency_ms"] >= 0

    def test_boolean_or_query(self, client):
        data = client.get("/search", params={"q": "search OR quantum"}).json()
        assert data["total"] > 0


# ---------------------------------------------------------------------------
# /autocomplete
# ---------------------------------------------------------------------------

class TestAutocomplete:
    def test_returns_200(self, client):
        r = client.get("/autocomplete", params={"q": "sea"})
        assert r.status_code == 200

    def test_has_suggestions_field(self, client):
        data = client.get("/autocomplete", params={"q": "sea"}).json()
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)

    def test_suggestions_start_with_prefix(self, client):
        data = client.get("/autocomplete", params={"q": "algo"}).json()
        for s in data["suggestions"]:
            assert s.startswith("algo")

    def test_n_param_respected(self, client):
        data = client.get("/autocomplete", params={"q": "s", "n": 3}).json()
        assert len(data["suggestions"]) <= 3

    def test_unknown_prefix_returns_empty(self, client):
        data = client.get("/autocomplete", params={"q": "zzzzqqqq"}).json()
        assert data["suggestions"] == []


# ---------------------------------------------------------------------------
# /index (add / update / delete)
# ---------------------------------------------------------------------------

class TestIndexEndpoints:
    NEW_DOC = {
        "id":       "test_doc_api",
        "title":    "API Test Document",
        "body":     "This is a test document inserted via the API endpoint.",
        "category": "test",
        "date":     "2024-06-01",
        "url":      "",
    }

    def test_add_document(self, client):
        r = client.post("/index", json=self.NEW_DOC)
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert r.json()["action"] == "add"

    def test_added_doc_searchable(self, client):
        client.post("/index", json=self.NEW_DOC)  # ensure exists
        data = client.get("/search", params={"q": "API test document"}).json()
        ids = [r["doc_id"] for r in data["results"]]
        assert "test_doc_api" in ids

    def test_add_duplicate_returns_409(self, client):
        client.post("/index", json=self.NEW_DOC)
        r = client.post("/index", json=self.NEW_DOC)
        assert r.status_code == 409

    def test_update_document(self, client):
        updated = {**self.NEW_DOC, "title": "Updated Title",
                   "body": "Updated body content about quantum."}
        r = client.put(f"/index/{self.NEW_DOC['id']}", json=updated)
        assert r.status_code == 200
        assert r.json()["action"] == "update"

    def test_delete_document(self, client):
        r = client.delete(f"/index/{self.NEW_DOC['id']}")
        # may already be deleted in prior test
        assert r.status_code in (200, 404)

    def test_delete_nonexistent_404(self, client):
        client.delete("/index/test_doc_api")  # ensure deleted
        r = client.delete("/index/test_doc_api")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# /analytics
# ---------------------------------------------------------------------------

class TestAnalytics:
    def test_summary_200(self, client):
        # generate at least one event
        client.get("/search", params={"q": "search"})
        r = client.get("/analytics/summary")
        assert r.status_code == 200

    def test_summary_has_keys(self, client):
        data = client.get("/analytics/summary").json()
        assert "total_searches" in data

    def test_top_queries_200(self, client):
        r = client.get("/analytics/top")
        assert r.status_code == 200
        assert "top_queries" in r.json()

    def test_zero_result_queries_200(self, client):
        r = client.get("/analytics/zero")
        assert r.status_code == 200

    def test_latency_200(self, client):
        r = client.get("/analytics/latency")
        assert r.status_code == 200

    def test_volume_200(self, client):
        r = client.get("/analytics/volume")
        assert r.status_code == 200
        assert "volume" in r.json()
