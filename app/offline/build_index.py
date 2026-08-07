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
        description="Build one Elasticsearch index and one FAISS index per source."
    )
    parser.add_argument("--docs", default=None, help="SourceDoc JSON/JSONL path.")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Output directory for shared docs.jsonl and manifest.json.",
    )
    parser.add_argument(
        "--source-config",
        default=None,
        help="Source registry JSON path. Defaults to LOCAL_SOURCE_CONFIG_PATH.",
    )
    parser.add_argument(
        "--index-es",
        action="store_true",
        help="Destructively rebuild every configured source's Elasticsearch index.",
    )
    args = parser.parse_args()

    docs = (
        load_docs_from_json_or_jsonl(args.docs, show_progress=True)
        if args.docs
        else LocalDocStore().load_all(show_progress=True)
    )
    source_registry = SourceRegistry.from_path(
        args.source_config or settings.LOCAL_SOURCE_CONFIG_PATH
    )
    result = await LocalIndexBuilder(
        artifact_store=LocalArtifactStore(artifact_dir=args.artifact_dir),
        index_elasticsearch=args.index_es,
        source_registry=source_registry,
        show_progress=True,
    ).build(docs)
    print(
        "built_local_indexes "
        f"doc_count={result.doc_count} "
        f"source_count={result.source_count} "
        f"embedding_dim={result.embedding_dim} "
        f"artifact_dir={result.artifact_dir} "
        f"faiss_indices={result.faiss_indices}"
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
