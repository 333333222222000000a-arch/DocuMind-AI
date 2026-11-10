import tempfile
import types
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from rag_app.citations import format_citations, format_debug_chunks, split_assistant_message
from rag_app.config import (
    GROQ_MODELS,
    MAX_OUTPUT_TOKENS,
    PROMPT_TEMPLATE,
)
from rag_app.answering import estimate_prompt_token_upper_bound, select_context_documents
from rag_app.ingestion import (
    corpus_fingerprint,
    safe_filename,
    save_uploaded_files,
    temporary_upload_directory,
    uploads_fingerprint,
)
from rag_app.vector_store import deduplicate_ranked_documents


class FakeDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class Upload:
    def __init__(self, name, content):
        self.name = name
        self.content = content

    def getbuffer(self):
        return self.content


class IngestionTests(unittest.TestCase):
    def test_filename_is_a_safe_pdf_component(self):
        self.assertEqual(safe_filename(r"..\..\outside\report?.pdf"), "report_.pdf")
        self.assertEqual(safe_filename(""), "document.pdf")
        self.assertEqual(safe_filename("notes.txt"), "notes.txt.pdf")

    def test_duplicate_safe_names_do_not_overwrite_uploads(self):
        with tempfile.TemporaryDirectory() as root:
            paths = save_uploaded_files(
                [Upload("../report.pdf", b"first"), Upload("report.pdf", b"second")], root
            )
            self.assertEqual([Path(path).name for path in paths], ["report.pdf", "report_1.pdf"])
            self.assertEqual([Path(path).read_bytes() for path in paths], [b"first", b"second"])

    def test_upload_fingerprint_tracks_content_even_when_name_matches(self):
        first = uploads_fingerprint([Upload("guide.pdf", b"version one")])
        changed = uploads_fingerprint([Upload("guide.pdf", b"version two")])
        self.assertNotEqual(first, changed)

    def test_upload_fingerprint_tracks_all_processing_and_retrieval_settings(self):
        upload = [Upload("guide.pdf", b"same bytes")]
        base = uploads_fingerprint(upload)
        changed_settings = [
            {"embedding_model": "other/model"},
            {"chunk_size": 700},
            {"chunk_overlap": 100},
            {"chunk_separators": ("\n", " ", "")},
            {"ocr_text_threshold": 120},
            {"top_k": 5},
            {"retrieval_fetch_k": 30},
            {"mmr_lambda": 0.8},
            {"duplicate_similarity_threshold": 0.95},
            {"max_prompt_token_budget": 3500},
            {"index_schema_version": 3},
        ]
        for settings in changed_settings:
            with self.subTest(settings=settings):
                self.assertNotEqual(base, uploads_fingerprint(upload, **settings))

    def test_corpus_fingerprint_tracks_content_and_configuration(self):
        docs = [FakeDocument("body", {"source": "guide.pdf"})]
        base = corpus_fingerprint(docs, "model-a", 800, 150)
        self.assertNotEqual(base, corpus_fingerprint(docs, "model-b", 800, 150))
        self.assertNotEqual(base, corpus_fingerprint([FakeDocument("changed", {"source": "guide.pdf"})], "model-a", 800, 150))
        self.assertNotEqual(base, corpus_fingerprint(docs, "model-a", 700, 150))
        self.assertNotEqual(base, corpus_fingerprint(docs, "model-a", 800, 150, ("\n", " ", "")))

    def test_temporary_upload_directory_is_removed_after_processing(self):
        with temporary_upload_directory() as directory:
            directory_path = Path(directory)
            self.assertTrue(directory_path.is_dir())
            Path(directory, "upload.pdf").write_bytes(b"placeholder")
        self.assertFalse(directory_path.exists())

    def test_pdf_loader_uses_ocr_for_short_text_and_preserves_source(self):
        loader_module = types.ModuleType("langchain_community.document_loaders")

        class TextLoader:
            def __init__(self, path):
                self.path = path

            def load(self):
                return [FakeDocument("scan", {"page": 0})]

        class OCRLoader:
            def __init__(self, path, **kwargs):
                self.options = kwargs

            def load(self):
                return [FakeDocument("recognized words", {"page": 0})]

        class Splitter:
            def __init__(self, **kwargs):
                self.options = kwargs

            def split_documents(self, pages):
                return pages

        loader_module.PyMuPDFLoader = TextLoader
        loader_module.UnstructuredPDFLoader = OCRLoader
        community = types.ModuleType("langchain_community")
        community.__path__ = []
        langchain = types.ModuleType("langchain")
        langchain.__path__ = []
        splitter_module = types.ModuleType("langchain.text_splitter")
        splitter_module.RecursiveCharacterTextSplitter = Splitter
        core = types.ModuleType("langchain_core")
        core.__path__ = []
        document_module = types.ModuleType("langchain_core.documents")
        document_module.Document = FakeDocument
        messages = []

        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {
            "langchain_community": community,
            "langchain_community.document_loaders": loader_module,
            "langchain": langchain,
            "langchain.text_splitter": splitter_module,
            "langchain_core": core,
            "langchain_core.documents": document_module,
        }):
            Path(root, "scan.pdf").write_bytes(b"first pdf")
            Path(root, "scan_1.pdf").write_bytes(b"second pdf")
            from rag_app.ingestion import load_documents_langchain
            docs = load_documents_langchain(
                root,
                source_names={"scan.pdf": "same name.pdf", "scan_1.pdf": "same name.pdf"},
                logger=lambda message, level="INFO": messages.append(message),
            )

        self.assertEqual([doc.page_content for doc in docs], ["recognized words", "recognized words"])
        self.assertEqual([doc.metadata["source"] for doc in docs], ["same name.pdf", "same name.pdf"])
        self.assertNotEqual(docs[0].metadata["source_id"], docs[1].metadata["source_id"])
        self.assertTrue(all(doc.metadata["chunk_id"] for doc in docs))
        self.assertTrue(all("file_path" not in doc.metadata for doc in docs))
        self.assertTrue(any("OCR mode" in message for message in messages))

    def test_duplicate_filenames_remain_distinct_sources(self):
        with tempfile.TemporaryDirectory() as root:
            paths = save_uploaded_files([Upload("report.pdf", b"A"), Upload("report.pdf", b"B")], root)
            self.assertEqual([Path(path).name for path in paths], ["report.pdf", "report_1.pdf"])


class CitationTests(unittest.TestCase):
    def test_assistant_message_footer_does_not_discard_citations(self):
        document = FakeDocument("source evidence", {"source": "manual.pdf", "page": 0})
        message = (
            "Supported answer [S1].|||FOOTER:Generated with: test-model"
            + format_citations([document])
            + format_debug_chunks([document])
        )

        parsed = split_assistant_message(message)

        self.assertEqual(parsed["answer"], "Supported answer [S1].")
        self.assertEqual(parsed["footer"], "Generated with: test-model")
        self.assertIn("manual.pdf — page 1", parsed["sources"])
        self.assertIn("source evidence", parsed["sources"])
        self.assertEqual(parsed["debug_chunks"], ["**Chunk 1**: source evidence"])

    def test_citations_include_source_and_truncated_snippet(self):
        result = format_citations([FakeDocument("abcdef", {"source": "manual.pdf"})], 3)
        self.assertIn("**[S1]** manual.pdf", result)
        self.assertIn("page not available", result)
        self.assertIn("abc...", result)
        self.assertIn("|||DEBUG_CHUNKS:1", format_debug_chunks([FakeDocument("line\ntwo")]))
        self.assertEqual(format_citations([]), "")

    def test_citations_render_real_pages_and_explicitly_mark_missing_pages(self):
        docs = [
            FakeDocument("first", {"source": "manual.pdf", "page": 0}),
            FakeDocument("second", {"source": "scan.pdf", "page_number": 8}),
            FakeDocument("third", {"source": "unknown.pdf"}),
        ]
        citations = format_citations(docs)
        self.assertIn("manual.pdf — page 1", citations)
        self.assertIn("scan.pdf — page 8", citations)
        self.assertIn("unknown.pdf — page not available", citations)

    def test_arabic_excerpt_is_normalized_for_display_only(self):
        original = "ﻣﻮﺍﺻﻔﺎﺕ ﻓﻨﻴﺔ"
        document = FakeDocument(original, {"source": "مواصفات.pdf", "page": 0})
        citation = format_citations([document])
        self.assertIn("مواصفات فنية", citation)
        self.assertEqual(document.page_content, original)


class AnsweringTests(unittest.TestCase):
    def test_answer_question_retrieves_context_and_passes_it_to_llm_chain(self):
        from rag_app.answering import answer_question

        prompt_factory = Mock(return_value="prompt")
        documents = [FakeDocument("server memory is 64 GB", {"source": "spec.pdf", "page": 2})]
        chain = Mock()
        chain.invoke.return_value = "The server has 64 GB memory [S1]."
        document_chain_factory = Mock(return_value=chain)
        retriever = Mock()
        retriever.invoke.return_value = documents

        prompt_module = types.ModuleType("langchain_core.prompts")
        prompt_module.ChatPromptTemplate = types.SimpleNamespace(from_template=prompt_factory)
        combine_module = types.ModuleType("langchain.chains.combine_documents")
        combine_module.create_stuff_documents_chain = document_chain_factory
        chains_module = types.ModuleType("langchain.chains")
        chains_module.__path__ = []

        with patch.dict("sys.modules", {
            "langchain_core": types.ModuleType("langchain_core"),
            "langchain_core.prompts": prompt_module,
            "langchain": types.ModuleType("langchain"),
            "langchain.chains": chains_module,
            "langchain.chains.combine_documents": combine_module,
        }):
            from rag_app.answering import answer_question
            answer, sources = answer_question(retriever, "llm", "question", "template")

        self.assertEqual(answer, "The server has 64 GB memory [S1].")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].metadata["source"], "spec.pdf")
        self.assertEqual(sources[0].metadata["citation_id"], "S1")
        retriever.invoke.assert_called_once_with("question")
        prompt_factory.assert_called_once_with("template")
        document_chain_factory.assert_called_once_with("llm", "prompt")
        chain.invoke.assert_called_once()
        prompt_context = chain.invoke.call_args.args[0]
        self.assertEqual(prompt_context["input"], "question")
        self.assertEqual(len(prompt_context["context"]), 1)
        self.assertIn("[S1] Source: spec.pdf | Page: 3", prompt_context["context"][0].page_content)
        self.assertIn("server memory is 64 GB", prompt_context["context"][0].page_content)

    def test_answer_citations_are_validated_and_no_answer_has_no_sources(self):
        from rag_app.answering import _cited_source_documents

        documents = [
            FakeDocument("cpu evidence", {"source": "cpu.pdf", "page": 0}),
            FakeDocument("ram evidence", {"source": "ram.pdf", "page": 2}),
        ]
        answer, sources = _cited_source_documents("The documents conflict [S2].", documents)
        self.assertEqual(answer, "The documents conflict [S2].")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].metadata["citation_id"], "S2")
        self.assertEqual(sources[0].metadata["source"], "ram.pdf")

        no_answer, no_sources = _cited_source_documents("NO_ANSWER: Not specified.", documents)
        self.assertEqual(no_answer, "Not specified.")
        self.assertEqual(no_sources, [])

        invalid_answer, invalid_sources = _cited_source_documents("Unsupported [S9].", documents)
        self.assertEqual(invalid_sources, [])
        self.assertIn("could not verify", invalid_answer)

    def test_conflicting_passages_are_both_contextualized_and_cited(self):
        from rag_app.answering import answer_question

        documents = [
            FakeDocument("The server specification lists 32GB RAM.", {
                "source": "specs.pdf", "source_id": "a", "page": 4,
            }),
            FakeDocument("The revised requirement lists 64GB RAM.", {
                "source": "amendment.pdf", "source_id": "b", "page": 1,
            }),
        ]
        retriever = Mock()
        retriever.invoke.return_value = documents
        chain = Mock()
        chain.invoke.return_value = "The passages conflict: 32GB [S1] vs 64GB [S2]."
        prompt_module = types.ModuleType("langchain_core.prompts")
        prompt_module.ChatPromptTemplate = types.SimpleNamespace(from_template=Mock(return_value="prompt"))
        combine_module = types.ModuleType("langchain.chains.combine_documents")
        combine_module.create_stuff_documents_chain = Mock(return_value=chain)
        chains_module = types.ModuleType("langchain.chains")
        chains_module.__path__ = []

        with patch.dict("sys.modules", {
            "langchain_core": types.ModuleType("langchain_core"),
            "langchain_core.prompts": prompt_module,
            "langchain": types.ModuleType("langchain"),
            "langchain.chains": chains_module,
            "langchain.chains.combine_documents": combine_module,
        }):
            answer, sources = answer_question(retriever, "llm", "How much RAM?", "{context}\n{input}")

        self.assertIn("conflict", answer)
        self.assertEqual([doc.metadata["citation_id"] for doc in sources], ["S1", "S2"])
        self.assertIn("[S1] Source: specs.pdf | Page: 5", chain.invoke.call_args.args[0]["context"][0].page_content)
        self.assertIn("[S2] Source: amendment.pdf | Page: 2", chain.invoke.call_args.args[0]["context"][1].page_content)
        citations = format_citations(sources)
        self.assertIn("[S1]", citations)
        self.assertIn("[S2]", citations)

    def test_answer_question_skips_llm_for_empty_retrieval(self):
        from rag_app.answering import answer_question

        retriever = Mock()
        retriever.invoke.return_value = []
        llm = Mock()
        answer, sources = answer_question(retriever, llm, "missing question")
        self.assertIn("do not contain relevant information", answer)
        self.assertEqual(sources, [])
        llm.invoke.assert_not_called()

    def test_prompt_budget_selects_ranked_chunks_without_sending_all_ten(self):
        docs = [
            FakeDocument(f"chunk {index}: " + ("مواصفة " * 100), {"source": "report.pdf", "page": index})
            for index in range(10)
        ]
        selected = select_context_documents(
            docs, "اذكر أهم المواصفات الفنية", max_prompt_tokens=3000
        )
        self.assertGreater(len(selected), 0)
        self.assertLess(len(selected), 10)
        self.assertIs(selected[0], docs[0])
        self.assertLessEqual(
            estimate_prompt_token_upper_bound(selected, "اذكر أهم المواصفات الفنية"),
            3000,
        )

    def test_prompt_preserves_document_context_and_question_fields(self):
        self.assertIn("{context}", PROMPT_TEMPLATE)
        self.assertIn("{input}", PROMPT_TEMPLATE)
        self.assertIn("Do not add outside knowledge", PROMPT_TEMPLATE)
        self.assertIn("cite each factual claim", PROMPT_TEMPLATE.casefold())
        self.assertIn("passages conflict", PROMPT_TEMPLATE)
        self.assertIn("same language as the question", PROMPT_TEMPLATE)


class VectorStoreTests(unittest.TestCase):
    def test_index_round_trip_uses_native_faiss_and_json_without_pickle(self):
        from rag_app.vector_store import build_or_load_faiss

        class FakeIndex:
            def __init__(self, content=b"native-index"):
                self.content = content

        class FakeDocstore:
            def __init__(self, docs):
                self.docs = docs

            def search(self, doc_id):
                return self.docs[doc_id]

        created = []
        loaded = []

        class FakeFAISS:
            @staticmethod
            def from_documents(documents, embeddings):
                created.append((documents, embeddings))
                docs_by_id = {"doc-0": documents[0]}
                return types.SimpleNamespace(
                    index=FakeIndex(),
                    index_to_docstore_id={0: "doc-0"},
                    docstore=FakeDocstore(docs_by_id),
                )

            def __init__(self, **kwargs):
                loaded.append(kwargs)
                self.kwargs = kwargs

        community = types.ModuleType("langchain_community")
        community.__path__ = []
        vectorstores = types.ModuleType("langchain_community.vectorstores")
        vectorstores.FAISS = FakeFAISS
        docstore_module = types.ModuleType("langchain_community.docstore.in_memory")
        docstore_module.InMemoryDocstore = lambda docs: FakeDocstore(docs)
        core = types.ModuleType("langchain_core")
        core.__path__ = []
        documents_module = types.ModuleType("langchain_core.documents")
        documents_module.Document = FakeDocument

        fake_faiss = types.ModuleType("faiss")
        fake_faiss.write_index = lambda index, filename: Path(filename).write_bytes(index.content)
        fake_faiss.read_index = lambda filename: FakeIndex(Path(filename).read_bytes())

        with tempfile.TemporaryDirectory() as root, patch.dict("sys.modules", {
            "langchain_community": community,
            "langchain_community.vectorstores": vectorstores,
            "langchain_community.docstore": types.ModuleType("langchain_community.docstore"),
            "langchain_community.docstore.in_memory": docstore_module,
            "langchain_core": core,
            "langchain_core.documents": documents_module,
            "faiss": fake_faiss,
        }):
            docs = [FakeDocument("evidence", {"source": "guide.pdf"})]
            with self.assertRaisesRegex(ValueError, "does not match"):
                build_or_load_faiss(
                    docs, root, embeddings="embeddings", embedding_model="test/model",
                    expected_fingerprint="0" * 64,
                )
            built = build_or_load_faiss(
                docs, root, embeddings="embeddings", embedding_model="test/model"
            )
            entry = next(Path(root).iterdir())
            self.assertTrue((entry / "index.faiss").is_file())
            self.assertTrue((entry / "documents.json").is_file())
            self.assertFalse(list(Path(root).rglob("*.pkl")))
            self.assertEqual(json.loads((entry / "manifest.json").read_text())["format"], 1)

            restored = build_or_load_faiss(
                docs, root, embeddings="embeddings", embedding_model="test/model"
            )

        self.assertIsNotNone(built)
        self.assertEqual(len(created), 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["index"].content, b"native-index")
        self.assertEqual(restored.kwargs["index"].content, b"native-index")

    def test_retriever_uses_diverse_mmr_candidates_for_budget_selection(self):
        from rag_app.vector_store import make_retriever

        vector_store = Mock()
        make_retriever(vector_store, 10)
        vector_store.as_retriever.assert_called_once_with(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.5},
        )

    def test_near_duplicates_are_filtered_without_collapsing_distinct_sources(self):
        base_text = "Server requires 32 GB RAM. " + ("Detailed specifications are listed. " * 20)
        first = FakeDocument(base_text, {
            "source": "report.pdf", "source_id": "file-a", "page": 4, "chunk_id": "a1",
        })
        duplicate = FakeDocument(base_text, {
            "source": "report.pdf", "source_id": "file-a", "page": 4, "chunk_id": "a2",
        })
        near_duplicate = FakeDocument(base_text[:-1] + "!", {
            "source": "report.pdf", "source_id": "file-a", "page": 4, "chunk_id": "a3",
        })
        other_pdf = FakeDocument(base_text, {
            "source": "report.pdf", "source_id": "file-b", "page": 4, "chunk_id": "b1",
        })
        self.assertEqual(
            deduplicate_ranked_documents([first, duplicate, near_duplicate, other_pdf]),
            [first, other_pdf],
        )


class ProviderTests(unittest.TestCase):
    def test_api_keys_absent_when_no_environment_or_secrets_file(self):
        from rag_app.providers import get_api_key

        class MissingSecretsStreamlit:
            @property
            def secrets(self):
                raise FileNotFoundError("No secrets file")

        streamlit_stub = MissingSecretsStreamlit()
        with patch.dict("sys.modules", {"streamlit": streamlit_stub}), \
                patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(get_api_key("GROQ_API_KEY"))
            self.assertIsNone(get_api_key("OPENAI_API_KEY"))

    def test_groq_key_is_read_from_environment(self):
        from rag_app.providers import get_api_key
        with patch.dict("os.environ", {"GROQ_API_KEY": "groq-env-key"}, clear=True):
            self.assertEqual(get_api_key("GROQ_API_KEY"), "groq-env-key")

    def test_openai_key_is_read_from_environment(self):
        from rag_app.providers import get_api_key
        with patch.dict("os.environ", {"OPENAI_API_KEY": "openai-env-key"}, clear=True):
            self.assertEqual(get_api_key("OPENAI_API_KEY"), "openai-env-key")

    def test_api_key_is_read_from_streamlit_secrets(self):
        from rag_app.providers import get_api_key
        streamlit_stub = types.SimpleNamespace(secrets={
            "GROQ_API_KEY": "groq-secret-key",
            "OPENAI_API_KEY": "openai-secret-key",
        })
        with patch.dict("sys.modules", {"streamlit": streamlit_stub}), \
                patch.dict("os.environ", {}, clear=True):
            self.assertEqual(get_api_key("GROQ_API_KEY"), "groq-secret-key")
            self.assertEqual(get_api_key("OPENAI_API_KEY"), "openai-secret-key")

    def test_groq_models_are_current_documented_production_ids(self):
        self.assertEqual(
            GROQ_MODELS,
            ("openai/gpt-oss-120b", "openai/gpt-oss-20b"),
        )

    def test_provider_adapters_preserve_original_model_settings(self):
        from rag_app import providers

        groq_calls = []
        openai_calls = []
        mock_groq_llm = Mock()
        mock_groq_llm.invoke.return_value = "mocked Groq response"
        with patch.object(providers, "GROQ_AVAILABLE", True), \
                patch.object(providers, "OPENAI_AVAILABLE", True), \
                patch.object(providers, "ChatGroq", lambda **kwargs: groq_calls.append(kwargs) or mock_groq_llm), \
                patch.object(providers, "ChatOpenAI", lambda **kwargs: openai_calls.append(kwargs) or "openai"):
            groq_llm = providers.get_groq_llm_instance("key", "model")
            self.assertEqual(groq_llm.invoke("question"), "mocked Groq response")
            self.assertEqual(providers.get_openai_llm_instance("key", "gpt-3.5-turbo"), "openai")

        mock_groq_llm.invoke.assert_called_once_with("question")

        self.assertEqual(groq_calls[0], {
            "groq_api_key": "key", "model_name": "model", "temperature": 0.0,
            "max_tokens": MAX_OUTPUT_TOKENS, "max_retries": 2,
        })
        self.assertEqual(openai_calls[0], {
            "model_name": "gpt-3.5-turbo", "temperature": 0.0, "openai_api_key": "key",
        })

    def test_provider_factories_reject_missing_api_keys(self):
        from rag_app import providers
        with patch.object(providers, "GROQ_AVAILABLE", True), patch.object(providers, "OPENAI_AVAILABLE", True):
            with self.assertRaisesRegex(ValueError, "GROQ_API_KEY"):
                providers.get_groq_llm_instance(None, "model")
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                providers.get_openai_llm_instance(None)

    def test_provider_errors_are_helpful_and_do_not_expose_secrets(self):
        from rag_app.providers import provider_error_summary

        class FakeResponse:
            status_code = 413
            def json(self):
                return {"error": {
                    "message": "Request too large for organization org_private: tokens per minute Limit 8,000, Requested 10,125",
                    "code": "rate_limit_exceeded",
                    "api_key": "gsk_private_secret",
                }}

        class FakeProviderError(Exception):
            response = FakeResponse()

        error = FakeProviderError("request rejected; key=gsk_supersecret123")
        summary = provider_error_summary(error)
        self.assertIn("token limit", summary)
        self.assertNotIn("gsk_supersecret123", summary)
        self.assertIn("8000", summary)
        self.assertIn("10125", summary)
        self.assertNotIn("org_private", summary)
        self.assertNotIn("gsk_private_secret", summary)

    def test_unavailable_model_error_is_sanitized(self):
        from rag_app.providers import provider_error_summary

        class FakeResponse:
            status_code = 404

        class FakeProviderError(Exception):
            response = FakeResponse()

        self.assertIn("model is unavailable", provider_error_summary(FakeProviderError("private body")))

    def test_invalid_provider_key_error_is_sanitized(self):
        from rag_app.providers import provider_error_summary

        class FakeResponse:
            status_code = 401

        class FakeProviderError(Exception):
            response = FakeResponse()

        summary = provider_error_summary(FakeProviderError("invalid key gsk_sensitive_value"))
        self.assertIn("rejected the configured API key", summary)
        self.assertNotIn("gsk_sensitive_value", summary)


if __name__ == "__main__":
    unittest.main()
