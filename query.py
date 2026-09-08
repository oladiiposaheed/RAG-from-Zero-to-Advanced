"""
Query script for the RAG system.

Usage:
    python query.py                            # simple retrieval
    python query.py --multi-query              # multi-query retrieval
    python query.py --compression              # contextual compression
    python query.py --self-query               # self-query retrieval
    python query.py --hyde                     # HyDE retrieval
    python query.py --decompose                # query decomposition
    python query.py --all                      # enable all techniques
"""

import argparse
from dotenv import load_dotenv
from app.services.rag_service import RAGService
from app.logger.custom_logger import setup_logging, get_logger

load_dotenv()
setup_logging(log_file='logs/query.log')
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi-query', action='store_true', help='Use multi-query retrieval')
    parser.add_argument('--compression', action='store_true', help='Use contextual compression')
    parser.add_argument('--self-query', action='store_true', help='Use self-query retrieval')
    parser.add_argument('--hyde', action='store_true', help='Use HyDE retrieval')
    parser.add_argument('--decompose', action='store_true', help='Use query decomposition')
    parser.add_argument('--all', action='store_true', help='Enable all retrieval techniques')
    args = parser.parse_args()

    # Determine flags based on --all or individual arguments
    if args.all:
        use_multi_query = True
        use_compression = True
        use_self_query = True
        use_hyde = True
        use_decomposition = True
    else:
        use_multi_query = args.multi_query
        use_compression = args.compression
        use_self_query = args.self_query
        use_hyde = args.hyde
        use_decomposition = args.decompose

    logger.info(
        f'Starting RAG query (multi-query={use_multi_query}, '
        f'compression={use_compression}, self-query={use_self_query}, '
        f'hyde={use_hyde}, decompose={use_decomposition})...'
    )

    # Create RAG service with selected options
    rag = RAGService(
        use_multi_query=use_multi_query,
        use_compression=use_compression,
        use_self_query=use_self_query,
        use_hyde=use_hyde,
        use_decomposition=use_decomposition
    )

    question = input('Your question: ')
    answer = rag.answer(question)
    sources = rag.retrieve_sources(question)

    print(f'\nAnswer: {answer}')
    if sources:
        print('\nSources used:')
        for i, doc in enumerate(sources, start=1):
            print(f'\nSource {i}:')
            print(doc.page_content[:300])
            print(f'Metadata: {doc.metadata}')
    else:
        print('\nNo sources retrieved.')


if __name__ == '__main__':
    main()