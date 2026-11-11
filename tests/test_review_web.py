from pathlib import Path

from fastapi.testclient import TestClient

from cognivore.config import Settings
from cognivore.web.app import create_app


def test_review_resumes_and_publishes_selected_findings(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path, prefer_semantic_embedder=False, llm_provider="fake", seed_demo_kb=False
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/review.html").status_code == 200
        assert client.get("/demo-before.md").status_code == 200
        response = client.post(
            "/api/reviews",
            files={
                "before": ("old.md", "PWR-1: Keep reserve power for 30 minutes."),
                "after": (
                    "new.md",
                    "PWR-1: Keep reserve power for 20 minutes.\nNET-1: Encrypt logs.",
                ),
            },
        )
        assert response.status_code == 201
        url = f"/api/reviews/{response.json()['id']}"
        assert client.post(f"{url}/publish").status_code == 409
        assert client.get(f"{url}/report").status_code == 409
        response = client.put(f"{url}/findings/0", json={"decision": "accepted"})
        assert response.json()["decisions"] == ["accepted", "pending"]

    with TestClient(create_app(settings)) as client:
        assert client.get(url).json()["decisions"] == ["accepted", "pending"]
        assert client.put(f"{url}/findings/1", json={"decision": "rejected"}).status_code == 200
        assert client.post(f"{url}/publish").json()["published"]
        report = client.get(f"{url}/report")
        assert report.status_code == 200
        assert "old.md:1" in report.text and "new.md:1" in report.text
        assert "30 minutes" in report.text and "20 minutes" in report.text
        assert "Encrypt logs" not in report.text
        assert client.post(f"{url}/publish").status_code == 200
        assert client.get(f"{url}/report").content == report.content
        assert client.put(f"{url}/findings/1", json={"decision": "accepted"}).status_code == 409
