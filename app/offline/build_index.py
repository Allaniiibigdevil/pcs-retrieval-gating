import argparse
import asyncio

from app.config import get_settings
from app.offline.local_index_builder import LocalIndexBuilder
from app.source_registry import SourceRegistry
from app.storage.local_doc_store import load_docs_from_json_or_jsonl


async def run() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Build one unified keyword ES index and one unified vector ES index."
    )
    parser.add_argument(
        "--docs",
        default=settings.LOCAL_RAW_DOCS_PATH,
        help="Mixed SourceDoc JSON/JSONL path containing all sources.",
    )
    parser.add_argument(
        "--source-config",
        default=None,
        help="Source registry JSON path. Defaults to LOCAL_SOURCE_CONFIG_PATH.",
    )
    args = parser.parse_args()

    docs = load_docs_from_json_or_jsonl(args.docs)
    if not docs:
        raise ValueError("No documents found in the input")

    registry = SourceRegistry.from_path(
        args.source_config or settings.LOCAL_SOURCE_CONFIG_PATH
    )
    result = await LocalIndexBuilder(source_registry=registry).build(docs)
    print(
        "built_local_index "
        f"doc_count={result.doc_count} "
        f"source_count={result.source_count} "
        f"embedding_dim={result.embedding_dim} "
        f"keyword_index={result.keyword_index} "
        f"vector_index={result.vector_index}"
    )


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
