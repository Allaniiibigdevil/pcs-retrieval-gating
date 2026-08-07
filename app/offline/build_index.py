import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.offline.local_index_builder import LocalIndexBuilder
from app.source_registry import SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore
from app.storage.local_doc_store import LocalDocStore, load_docs_from_json_or_jsonl


async def run() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build one source's Elasticsearch and FAISS indexes from a mixed docs file."
    )
    parser.add_argument("--docs", default=None, help="Mixed SourceDoc JSON/JSONL path.")
    parser.add_argument(
        "--source",
        required=True,
        help="Source id to build. The input file may contain documents from other sources.",
    )
    parser.add_argument(
        "--source-config",
        default=None,
        help="Source registry JSON path. Defaults to LOCAL_SOURCE_CONFIG_PATH.",
    )
    parser.add_argument(
        "--index-es",
        action="store_true",
        help="Destructively rebuild this source's Elasticsearch index.",
    )
    args = parser.parse_args()

    registry = SourceRegistry.from_path(
        args.source_config or settings.LOCAL_SOURCE_CONFIG_PATH
    )
    source = registry.require(args.source)

    all_docs = (
        load_docs_from_json_or_jsonl(args.docs, show_progress=True)
        if args.docs
        else LocalDocStore().load_all(show_progress=True)
    )
    docs = [doc for doc in all_docs if doc.system_id == source.source_id]
    if not docs:
        raise ValueError(f"No documents found for source_id {source.source_id!r}")

    single_source_registry = SourceRegistry(
        [source],
        config_path=registry.config_path,
    )
    artifact_dir = Path(source.faiss_index_path).parent

    result = await LocalIndexBuilder(
        artifact_store=LocalArtifactStore(artifact_dir=artifact_dir),
        index_elasticsearch=args.index_es,
        source_registry=single_source_registry,
        show_progress=True,
    ).build(docs)
    print(
        "built_local_source_index "
        f"source={source.source_id} "
        f"doc_count={result.doc_count} "
        f"embedding_dim={result.embedding_dim} "
        f"artifact_dir={result.artifact_dir} "
        f"faiss_index={source.faiss_index_path}"
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
