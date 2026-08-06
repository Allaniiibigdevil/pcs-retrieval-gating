import argparse
import asyncio

from app.config import get_settings
from app.offline.local_index_builder import LocalIndexBuilder
from app.source_registry import SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore
from app.storage.local_doc_store import LocalDocStore, load_docs_from_json_or_jsonl


async def run() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build one shared FAISS index and one Elasticsearch index per source."
    )
    parser.add_argument("--docs", default=None, help="SourceDoc JSON/JSONL path.")
    parser.add_argument("--artifact-dir", default=None, help="Output docs/manifest directory.")
    parser.add_argument(
        "--source-config",
        default=None,
        help="Source registry JSON path. Defaults to LOCAL_SOURCE_CONFIG_PATH.",
    )
    parser.add_argument(
        "--faiss-index-path",
        default=None,
        help="Shared FAISS index output path. Defaults to LOCAL_FAISS_INDEX_PATH.",
    )
    parser.add_argument(
        "--faiss-doc-ids-path",
        default=None,
        help="Shared FAISS doc-id mapping output path. Defaults to LOCAL_FAISS_DOC_IDS_PATH.",
    )
    parser.add_argument(
        "--index-es",
        action="store_true",
        help="Destructively rebuild every configured source's Elasticsearch index.",
    )
    args = parser.parse_args()

    docs = (
        load_docs_from_json_or_jsonl(args.docs)
        if args.docs
        else LocalDocStore().load_all()
    )
    source_registry = SourceRegistry.from_path(
        args.source_config or settings.LOCAL_SOURCE_CONFIG_PATH
    )
    artifact_store = LocalArtifactStore(
        artifact_dir=args.artifact_dir,
        faiss_index_path=args.faiss_index_path,
        faiss_doc_ids_path=args.faiss_doc_ids_path,
    )
    result = await LocalIndexBuilder(
        artifact_store=artifact_store,
        index_elasticsearch=args.index_es,
        source_registry=source_registry,
    ).build(docs)
    print(
        "built_local_index "
        f"doc_count={result.doc_count} "
        f"source_count={result.source_count} "
        f"embedding_dim={result.embedding_dim} "
        f"artifact_dir={result.artifact_dir} "
        f"faiss_index={artifact_store.faiss_path} "
        f"faiss_doc_ids={artifact_store.faiss_doc_ids_path}"
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
