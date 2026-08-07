import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.offline.local_index_builder import LocalIndexBuilder
from app.source_registry import SourceRegistry
from app.storage.local_artifact_store import LocalArtifactStore
from app.storage.local_doc_store import count_docs_from_jsonl, iter_docs_from_jsonl


async def run() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Stream-build one source's Elasticsearch and FAISS indexes from a mixed JSONL file."
    )
    parser.add_argument(
        "--docs",
        default=settings.LOCAL_RAW_DOCS_PATH,
        help="Mixed SourceDoc JSONL path.",
    )
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
        "--batch-size",
        type=int,
        default=None,
        help="Embedding stream batch size. Defaults to INDEX_BUILD_EMBEDDING_CHUNK_SIZE.",
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
    input_path = Path(args.docs)

    print(f"scanning_source source={source.source_id} docs={input_path}")
    total_docs = count_docs_from_jsonl(input_path, source_id=source.source_id)
    if total_docs <= 0:
        raise ValueError(f"No documents found for source_id {source.source_id!r}")
    print(f"source_docs_found source={source.source_id} count={total_docs}")

    single_source_registry = SourceRegistry(
        [source],
        config_path=registry.config_path,
    )
    artifact_dir = Path(source.faiss_index_path).parent
    docs = iter_docs_from_jsonl(input_path, source_id=source.source_id)

    result = await LocalIndexBuilder(
        artifact_store=LocalArtifactStore(artifact_dir=artifact_dir),
        index_elasticsearch=args.index_es,
        source_registry=single_source_registry,
        show_progress=True,
    ).build_streaming_source(
        docs,
        source_id=source.source_id,
        total_docs=total_docs,
        batch_size=args.batch_size,
    )
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
