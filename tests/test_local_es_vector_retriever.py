import asyncio

from app.retrieval.local_es_vector_retriever import LocalElasticsearchVectorRetriever


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def embed(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(text) for text in texts]


class FakeLocalElasticsearchVectorRetriever(LocalElasticsearchVectorRetriever):
    def __init__(self, response: dict, embedding_service: FakeEmbeddingService) -> None:
        super().__init__(
            source_id="photo",
            index_name="pcs_retrieval_vectors",
            embedding_service=embedding_service,
            base_url="http://localhost:9200",
        )
        self.response = response
        self.request_body = None

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        del method, path
        self.request_body = body
        return self.response


def test_vector_retriever_filters_system_and_uses_raw_cosine_score() -> None:
    embedding_service = FakeEmbeddingService()
    retriever = FakeLocalElasticsearchVectorRetriever(
        {
            "hits": {
                "hits": [
                    {
                        "_id": "photo:doc-1",
                        "_score": 0.73,
                        "_source": {
                            "doc_id": "doc-1",
                            "system_id": "photo",
                            "summary": "去年去日本旅行时拍的照片",
                        },
                    }
                ]
            }
        },
        embedding_service,
    )

    hits = asyncio.run(retriever.search("旅游照片", top_k=10))

    assert embedding_service.queries == ["旅游照片"]
    assert retriever.request_body["size"] == 10
    assert retriever.request_body["_source"] == ["doc_id", "system_id", "summary"]
    script_score = retriever.request_body["query"]["script_score"]
    assert script_score["query"] == {"term": {"system_id": "photo"}}
    assert script_score["script"] == {
        "source": "cosineSimilarity(params.query_vector, 'embedding')",
        "params": {"query_vector": [0.1, 0.2, 0.3]},
    }
    assert hits[0].doc_id == "doc-1"
    assert hits[0].system_id == "photo"
    assert hits[0].summary == "去年去日本旅行时拍的照片"
    assert hits[0].vector_score == 0.73
    assert hits[0].vector_rank == 1
