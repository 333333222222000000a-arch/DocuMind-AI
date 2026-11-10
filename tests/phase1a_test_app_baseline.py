import ast
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_SOURCE = APP_PATH.read_text(encoding="utf-8")
APP_TREE = ast.parse(APP_SOURCE)


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def function_from_app(name, namespace):
    """Compile the unchanged function body from app.py with controlled dependencies."""
    node = next(
        item for item in APP_TREE.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name
    )
    node.decorator_list = []
    node = ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP_PATH), "exec"), namespace)
    return namespace[name]


def install_document_loader_stubs(pages, chunks=None, ocr_pages=None, ocr_error=None):
    community = sys.modules.setdefault("langchain_community", types.ModuleType("langchain_community"))
    community.__path__ = []
    loaders = types.ModuleType("langchain_community.document_loaders")

    class PyMuPDFLoader:
        def __init__(self, path):
            self.path = path

        def load(self):
            return pages

    class UnstructuredPDFLoader:
        def __init__(self, path, **kwargs):
            self.path = path
            self.kwargs = kwargs

        def load(self):
            if ocr_error:
                raise ocr_error
            return ocr_pages or []

    class Splitter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def split_documents(self, documents):
            return chunks if chunks is not None else documents

    loaders.PyMuPDFLoader = PyMuPDFLoader
    loaders.UnstructuredPDFLoader = UnstructuredPDFLoader
    community.document_loaders = loaders
    sys.modules["langchain_community.document_loaders"] = loaders

    core = sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
    core.__path__ = []
    documents_module = types.ModuleType("langchain_core.documents")
    documents_module.Document = FakeDocument
    core.documents = documents_module
    sys.modules["langchain_core.documents"] = documents_module

    langchain = sys.modules.setdefault("langchain", types.ModuleType("langchain"))
    langchain.__path__ = []
    splitter_module = types.ModuleType("langchain.text_splitter")
    splitter_module.RecursiveCharacterTextSplitter = Splitter
    langchain.text_splitter = splitter_module
    sys.modules["langchain.text_splitter"] = splitter_module
    return Splitter


class AppBaselineTests(unittest.TestCase):
    def setUp(self):
        self.logs = []
        self.errors = []
        self.namespace = {
            "os": os,
            "Path": Path,
            "List": list,
            "Document": FakeDocument,
            "CHUNK_SIZE": 800,
            "CHUNK_OVERLAP": 150,
            "log_event": lambda message, level="INFO": self.logs.append((message, level)),
            "st": types.SimpleNamespace(
                error=lambda message: self.errors.append(message),
                warning=lambda message: None,
                success=lambda message: None,
                spinner=lambda message: __import__("contextlib").nullcontext(),
            ),
        }

    def test_uploaded_file_is_saved_and_bytes_preserved(self):
        class Upload:
            name = "notes.pdf"

            def getbuffer(self):
                return b"pdf bytes"

        save = function_from_app("save_uploaded_files", self.namespace)
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = save([Upload()], temp_dir)
            self.assertEqual(len(paths), 1)
            self.assertEqual(Path(paths[0]).name, "notes.pdf")
            self.assertEqual(Path(paths[0]).read_bytes(), b"pdf bytes")

    def test_pdf_extraction_chunking_and_source_metadata(self):
        pages = [FakeDocument("Extracted document content " * 8, {"page": 2})]
        chunks = [FakeDocument("chunk one", {"page": 2}), FakeDocument("chunk two", {"page": 3})]
        self.namespace["RecursiveCharacterTextSplitter"] = install_document_loader_stubs(pages, chunks)
        load_docs = function_from_app("load_documents_langchain", self.namespace)
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "guide.pdf").write_bytes(b"placeholder")
            docs = load_docs(temp_dir)
        self.assertEqual([doc.page_content for doc in docs], ["chunk one", "chunk two"])
        self.assertEqual(docs[0].metadata, {"page": 2, "source": "guide.pdf"})
        self.assertEqual(docs[1].metadata["source"], "guide.pdf")
        self.assertTrue(any("Created 2 chunks" in message for message, _ in self.logs))

    def test_short_pdf_uses_ocr_loader(self):
        short_pages = [FakeDocument("short", {"page": 0})]
        ocr_pages = [FakeDocument("OCR extracted content", {"page": 0})]
        self.namespace["RecursiveCharacterTextSplitter"] = install_document_loader_stubs(short_pages, ocr_pages=ocr_pages)
        load_docs = function_from_app("load_documents_langchain", self.namespace)
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "scan.pdf").touch()
            docs = load_docs(temp_dir)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "OCR extracted content")
        self.assertEqual(docs[0].metadata["source"], "scan.pdf")
        self.assertTrue(any("using OCR mode" in message for message, _ in self.logs))

    def test_embedding_factory_uses_configured_model(self):
        calls = []
        self.namespace["EMBEDDING_MODEL_NAME"] = "test/model"
        self.namespace["HuggingFaceEmbeddings"] = lambda **kwargs: calls.append(kwargs) or "embedding-object"
        get_embeddings = function_from_app("get_embeddings", self.namespace)
        self.assertEqual(get_embeddings(), "embedding-object")
        self.assertEqual(calls, [{"model_name": "test/model"}])

    def test_faiss_index_build_and_save(self):
        calls = []
        db = Mock()

        class FakeFAISS:
            @staticmethod
            def load_local(*args, **kwargs):
                raise AssertionError("load should not be called for a new index")

            @staticmethod
            def from_documents(docs, embeddings):
                calls.append(("from_documents", docs, embeddings))
                return db

        self.namespace.update({
            "FAISS": FakeFAISS,
            "get_embeddings": lambda: "fake-embeddings",
            "shutil": __import__("shutil"),
        })
        build = function_from_app("build_or_load_faiss", self.namespace)
        docs = [FakeDocument("text", {"source": "a.pdf"})]
        with tempfile.TemporaryDirectory() as root:
            index_path = str(Path(root, "index"))
            result = build(docs, index_path, force_rebuild=True)
            self.assertIs(result, db)
            db.save_local.assert_called_once_with(index_path)
            self.assertTrue(Path(index_path).is_dir())
        self.assertEqual(calls, [("from_documents", docs, "fake-embeddings")])

    def test_faiss_existing_index_is_loaded_and_retriever_uses_top_k(self):
        db = Mock()
        load_calls = []

        class FakeFAISS:
            @staticmethod
            def load_local(*args, **kwargs):
                load_calls.append((args, kwargs))
                return db

        self.namespace.update({
            "FAISS": FakeFAISS,
            "get_embeddings": lambda: "fake-embeddings",
            "TOP_K": 10,
        })
        build = function_from_app("build_or_load_faiss", self.namespace)
        with tempfile.TemporaryDirectory() as root:
            index_path = Path(root, "index")
            index_path.mkdir()
            self.assertIs(build([], str(index_path)), db)
        self.assertTrue(load_calls[0][1]["allow_dangerous_deserialization"])
        self.assertIn('db.as_retriever(\n                    search_kwargs={"k": TOP_K}', APP_SOURCE)

    def test_groq_factory_passes_api_key_model_and_settings(self):
        calls = []
        self.namespace["ChatGroq"] = lambda **kwargs: calls.append(kwargs) or "llm"
        groq = function_from_app("get_groq_llm_instance", self.namespace)
        self.assertEqual(groq("mock-key", "mock-model"), "llm")
        self.assertEqual(calls, [{
            "groq_api_key": "mock-key",
            "model_name": "mock-model",
            "temperature": 0.0,
            "max_tokens": 4096,
        }])

    def test_provider_answer_and_citation_wiring_is_present(self):
        # Answer generation is currently inline in app.py's Streamlit UI flow.
        self.assertIn('models_to_try.extend([(name, "groq") for name in GROQ_MODELS])', APP_SOURCE)
        self.assertIn('models_to_try.append(("gpt-3.5-turbo", "openai"))', APP_SOURCE)
        self.assertIn('create_retrieval_chain(retriever, document_chain)', APP_SOURCE)
        self.assertIn('qa_chain.invoke({"input": latest_query})', APP_SOURCE)
        self.assertIn('result.get("answer", "No answer generated")', APP_SOURCE)
        self.assertIn('result.get("context", [])', APP_SOURCE)
        self.assertIn('doc.metadata.get("source", f"Document {i}")', APP_SOURCE)
        self.assertIn('OpenAI(temperature=0.0, openai_api_key=openai_key)', APP_SOURCE)
        self.assertIn('sources_text += f"\\n**[{i}]** {source}\\n> {snippet}\\n"', APP_SOURCE)


if __name__ == "__main__":
    unittest.main()
