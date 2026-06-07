from __future__ import annotations

import unittest
from unittest.mock import patch

from analysis_pipeline.stages.discovery.runner import check_pdf_downloadable, pdf_candidate_urls


class FakeHeadResponse:
    def __init__(self, status_code: int, content_type: str, url: str = "https://example.org/file.pdf"):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.url = url


class FakeRaw:
    def __init__(self, data: bytes):
        self.data = data

    def read(self, _max_bytes: int, decode_content: bool = True) -> bytes:
        return self.data


class FakeGetResponse:
    def __init__(self, status_code: int, content_type: str, data: bytes, url: str = "https://example.org/file.pdf"):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.raw = FakeRaw(data)
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class DiscoveryDownloadContractTests(unittest.TestCase):
    def test_arxiv_url_is_promoted_to_stable_pdf_url(self) -> None:
        urls = pdf_candidate_urls({"source": "arxiv", "arxiv_id": "2106.14834v1"})
        self.assertEqual(urls[0], "https://arxiv.org/pdf/2106.14834v1.pdf")

    def test_openalex_locations_are_extracted(self) -> None:
        urls = pdf_candidate_urls(
            {
                "source": "openalex",
                "open_access": {"oa_url": "https://publisher.example/landing"},
                "primary_location": {"pdf_url": "https://publisher.example/a.pdf"},
                "best_oa_location": {"landing_page_url": "https://repo.example/page"},
                "locations": [{"pdf_url": "https://repo.example/b.pdf"}],
            }
        )
        self.assertIn("https://publisher.example/a.pdf", urls)
        self.assertIn("https://repo.example/b.pdf", urls)
        self.assertIn("https://repo.example/page", urls)

    @patch("analysis_pipeline.stages.discovery.runner.requests.head")
    def test_head_pdf_content_type_passes(self, mock_head) -> None:
        mock_head.return_value = FakeHeadResponse(200, "application/pdf")
        result = check_pdf_downloadable(["https://example.org/file.pdf"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["reason"], "HEAD content-type is PDF")

    @patch("analysis_pipeline.stages.discovery.runner.requests.get")
    @patch("analysis_pipeline.stages.discovery.runner.requests.head")
    def test_html_landing_page_fails(self, mock_head, mock_get) -> None:
        mock_head.return_value = FakeHeadResponse(405, "text/html")
        mock_get.return_value = FakeGetResponse(200, "text/html", b"<html>not pdf</html>")
        result = check_pdf_downloadable(["https://example.org/landing"])
        self.assertFalse(result["ok"])
        self.assertIn("not a PDF", result["reason"])


if __name__ == "__main__":
    unittest.main()
