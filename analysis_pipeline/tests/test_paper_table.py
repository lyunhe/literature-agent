from __future__ import annotations

import unittest
from unittest.mock import patch

from analysis_pipeline.stages.discovery.candidate_links import get_doi_link, pdf_candidate_urls
from analysis_pipeline.stages.discovery.paper_table import (
    PAPER_TABLE_COLUMNS,
    build_paper_table,
    summarize_abstracts_cn,
)


class PaperTableTests(unittest.TestCase):
    def test_build_paper_table_columns(self) -> None:
        papers = [
            {
                "candidate_id": "c1",
                "title": "Energy Storage in Electricity Markets",
                "title_cn": "电力市场中的储能",
                "abstract": "This paper studies battery participation in day-ahead markets.",
                "venue": "IEEE Transactions on Power Systems",
                "year": 2024,
                "source": "openalex",
                "doi": "10.1000/example.1",
                "arxiv_id": "",
                "oa_url": "https://publisher.example/oa",
                "rank": 1,
                "direction_id": "D1",
                "final_score": 8.5,
                "relevance_score": 7.2,
                "journal_level_score": 4.0,
                "journal_level": "Q1",
                "pdf_url": "https://publisher.example/paper.pdf",
            }
        ]

        with patch("analysis_pipeline.stages.discovery.paper_table.summarize_abstracts_cn") as mock_summary:
            mock_summary.return_value = ["研究电池参与日前市场。"]
            rows = build_paper_table(papers)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        for column in PAPER_TABLE_COLUMNS:
            self.assertIn(column, row)
        self.assertEqual(row["title_cn"], "电力市场中的储能")
        self.assertEqual(row["abstract_summary_cn"], "研究电池参与日前市场。")
        self.assertEqual(row["venue"], "IEEE Transactions on Power Systems")
        self.assertEqual(row["doi"], "10.1000/example.1")
        self.assertEqual(row["doi_link"], "https://doi.org/10.1000/example.1")
        self.assertEqual(row["pdf_url"], "https://publisher.example/paper.pdf")
        self.assertIn("https://publisher.example/paper.pdf", row["pdf_url_candidates"])
        self.assertEqual(row["download_status"], "not_attempted")
        self.assertFalse(row["downloaded"])

    def test_pdf_candidate_urls_arxiv_and_openalex(self) -> None:
        arxiv_urls = pdf_candidate_urls({"source": "arxiv", "arxiv_id": "2106.14834v1"})
        self.assertEqual(arxiv_urls[0], "https://arxiv.org/pdf/2106.14834v1.pdf")

        openalex_urls = pdf_candidate_urls(
            {
                "source": "openalex",
                "doi": "10.1000/example.2",
                "open_access": {"oa_url": "https://publisher.example/landing"},
                "primary_location": {"pdf_url": "https://publisher.example/a.pdf"},
                "best_oa_location": {"landing_page_url": "https://repo.example/page"},
                "locations": [{"pdf_url": "https://repo.example/b.pdf"}],
            }
        )
        self.assertIn("https://publisher.example/a.pdf", openalex_urls)
        self.assertIn("https://repo.example/b.pdf", openalex_urls)
        self.assertEqual(get_doi_link({"doi": "10.1000/example.2"}), "https://doi.org/10.1000/example.2")

    def test_abstract_fallback_without_llm(self) -> None:
        long_abstract = "word " * 200
        with patch("analysis_pipeline.stages.discovery.paper_table.llm_request", side_effect=RuntimeError("no api")):
            summaries = summarize_abstracts_cn([long_abstract, ""])
        self.assertTrue(all(isinstance(item, str) for item in summaries))
        self.assertTrue(len(summaries[0]) <= 280)
        self.assertEqual(summaries[1], "")


if __name__ == "__main__":
    unittest.main()
