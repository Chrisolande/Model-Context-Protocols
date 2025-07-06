import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from flashrank import Ranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_compressors import FlashrankRerank
from langchain_community.document_loaders import TextLoader
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from mcp.server.fastmcp.server import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration for search tools."""

    file_path: str = "romeo.txt"
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.4
    redundancy_threshold: float = 0.95
    retrieval_k: int = 25
    tavily_max_results: int = 5


class SearchToolsManager:
    """Manage search tools and vector store."""

    def __init__(self, config: SearchConfig | None = None):
        if config is None:
            self.config = SearchConfig()
        else:
            self.config = config

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )
        self.tavily_tool = TavilySearch(
            max_results=self.config.tavily_max_results, topic="general"
        )
        self.persist_directory = Path("faiss_index")
        self.vector_store = None
        self.compression_retriever = None
        logger.info("SearchToolsManager initialized")

    def initialize_vector_store(self):
        """Initialize vector store with document processing."""
        try:
            if not os.path.exists(self.config.file_path):
                logger.warning(
                    f"File {self.config.file_path} not found. Vector search disabled."
                )
                return

            if self.persist_directory.exists():
                logger.info(
                    f"Loading existing vector store from {self.persist_directory}"
                )
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
            else:
                logger.info("Creating new vector store from documents")
                docs = self._process_documents()
                self.vector_store = FAISS.from_documents(docs, self.embedding_model)

                os.makedirs(self.persist_directory, exist_ok=True)
                self.vector_store.save_local(self.persist_directory)
                logger.info(f"Saved new vector store to {self.persist_directory}")

            self.compression_retriever = self._create_compression_retriever()
            logger.info("Vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")

    async def initialize_vector_store_async(self):
        """Initialize vector store with document processing (async)"""
        try:
            if not os.path.exists(self.config.file_path):
                logger.warning(
                    f"File {self.config.file_path} not found. Vector search disabled."
                )
                return

            # Run CPU-intensive tasks in thread pool
            loop = asyncio.get_event_loop()

            if self.persist_directory.exists():
                logger.info(
                    f"Loading existing vector store from {self.persist_directory}"
                )
                self.vector_store = await loop.run_in_executor(
                    None,
                    lambda: FAISS.load_local(
                        self.persist_directory,
                        self.embedding_model,
                        allow_dangerous_deserialization=True,
                    ),
                )
            else:
                logger.info("Creating new vector store from documents")
                docs = await loop.run_in_executor(None, self._process_documents)
                self.vector_store = await loop.run_in_executor(
                    None, lambda: FAISS.from_documents(docs, self.embedding_model)
                )

                os.makedirs(self.persist_directory, exist_ok=True)
                await loop.run_in_executor(
                    None, lambda: self.vector_store.save_local(self.persist_directory)
                )
                logger.info(f"Saved new vector store to {self.persist_directory}")

            self.compression_retriever = self._create_compression_retriever()
            logger.info("Vector store initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")

    def _process_documents(self):
        """Process documents for vector store."""
        loader = TextLoader(file_path=self.config.file_path)
        doc = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )

        docs = splitter.split_documents(doc)
        logger.info(f"Processed {len(docs)} document chunks")
        return docs

    def _create_compression_retriever(self):
        """Create compression retriever with ranking."""
        if not self.vector_store:
            return None

        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )

        ranker = Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir=os.path.expanduser("~/.cache/flashrank"),
        )

        compressor = DocumentCompressorPipeline(
            transformers=[
                EmbeddingsFilter(
                    embeddings=self.embedding_model,
                    similarity_threshold=self.config.similarity_threshold,
                ),
                EmbeddingsRedundantFilter(
                    embeddings=self.embedding_model,
                    similarity_threshold=self.config.redundancy_threshold,
                ),
                FlashrankRerank(client=ranker),
            ]
        )

        return ContextualCompressionRetriever(
            base_retriever=retriever, base_compressor=compressor
        )

    def search_web(self, query: str) -> str:
        """Search web using Tavily."""
        try:
            logger.info(f"Web search for: {query}")
            results = self.tavily_tool.invoke({"query": query})
            return str(results)
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return f"Web search failed: {str(e)}"

    def search_documents(self, query: str) -> str:
        """Search local documents using vector retrieval."""
        try:
            if not self.compression_retriever:
                return "Document search not available. Vector store not initialized."

            logger.info(f"Document search for: {query}")
            results = self.compression_retriever.invoke(query)

            if not results:
                return "No relevant documents found."

            formatted_results = "\n\n".join(
                [
                    f"Document {i+1}:\n{doc.page_content}"
                    for i, doc in enumerate(results)
                ]
            )

            return formatted_results

        except Exception as e:
            logger.error(f"Document search error: {str(e)}")
            return f"Document search failed: {str(e)}"


# Initialize components
config = SearchConfig()
search_manager = SearchToolsManager(config)

mcp = FastMCP("Enhanced Search Tools")


@mcp.tool()
async def tavily_search(query: str) -> str:
    """Search the web using Tavily."""
    return search_manager.search_web(query)


@mcp.tool()
async def document_search(query: str) -> str:
    """Search local documents using vector retrieval with compression and ranking."""
    return search_manager.search_documents(query)


@mcp.tool()
async def hybrid_search(query: str) -> str:
    """Perform both web and document search, return combined results."""
    try:
        logger.info(f"Hybrid search for: {query}")

        # Run searches concurrently
        web_task = asyncio.create_task(
            asyncio.to_thread(search_manager.search_web, query)
        )
        doc_task = asyncio.create_task(
            asyncio.to_thread(search_manager.search_documents, query)
        )

        web_results, doc_results = await asyncio.gather(web_task, doc_task)

        combined_results = f"=== WEB SEARCH RESULTS ===\n{web_results}\n\n"
        combined_results += f"=== DOCUMENT SEARCH RESULTS ===\n{doc_results}"

        return combined_results

    except Exception as e:
        logger.error(f"Hybrid search error: {str(e)}")
        return f"Hybrid search failed: {str(e)}"


def main():
    """Initialize and run MCP server."""
    logger.info("Initializing search tools...")

    search_manager.initialize_vector_store()

    logger.info("Starting MCP server...")
    mcp.run()


if __name__ == "__main__":
    main()
