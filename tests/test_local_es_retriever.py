import asyncio

from app.retrieval.local_es_retriever import LocalElasticsearchRetriever


class FakeLocalElasticsearchRetriever(LocalElasticsearchRetriever):
    def __init__(self, response: dict) -> None:
        super().__init__(base_url="http://localhost:9200", index_name="test")
        self.response = response
        self.request_body = None

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        del method, path
        self.request_body = body
        return self.response


def test_local_es_retriever_uses_rrf_query_shape_and_highlights() -> None:
    retriever = FakeLocalElasticsearchRetriever(
        {
            "hits": {
                "hits": [
                    {
                        "_id": "doc-1",
                        "_score": 3.2,
                        "_source": {
                            "doc_id": "doc-1",
                            "system_id": "notepad",
                            "summary": "记录了用户对海鲜过敏",
                            "keywords": ["过敏", "海鲜过敏"],
                            "metadata": {"source": "fixture"},
                        },
                        "highlight": {
                            "keywords": ["<em>海鲜过敏</em>"],
                            "summary": ["记录了用户对<em>海鲜</em>过敏"],
                        },
                    }
                ]
            }
        }
    )

    hits = asyncio.run(retriever.search("海鲜 过敏", top_k=5))

    assert retriever.request_body is not None
    query = retriever.request_body["query"]["multi_match"]
    assert query["type"] == "cross_fields"
    assert query["operator"] == "or"
    assert query["minimum_should_match"] == "1<2"
    assert retriever.fields == ["summary^1.0", "keywords^1.0"]
    assert retriever.request_body["highlight"]["fields"]["keywords"] == {
        "number_of_fragments": 0
    }
    assert hits[0].metadata["matched_keywords"] == ["海鲜过敏"]
    assert hits[0].metadata["highlight"]["summary"] == ["记录了用户对<em>海鲜</em>过敏"]
