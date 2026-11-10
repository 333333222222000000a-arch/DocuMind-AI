# Phase 1A baseline test notes

Run from the repository root with the Python standard library:

    python -m unittest discover -s tests -v

The tests compile selected existing helper functions directly from app.py and
provide fake document loaders, embedding constructors, FAISS, and Groq clients.
No provider API key or external service is used. PDF parsing itself is mocked;
the extraction/chunking path and source metadata behavior are exercised against
the unchanged application function.

Answer generation and the OpenAI constructor are inside Streamlit's top-level
UI flow, not callable application functions. Importing app.py executes UI setup
and requires Streamlit and LangChain packages, which are not installed in the
current Python environment. For that reason this baseline uses source contract
assertions for provider ordering, chain invocation, returned answer/context,
and citation formatting. It does not claim to execute those UI interactions.

Embedding generation, FAISS indexing/search, and actual PDF parsing are
represented with deterministic fakes because the application dependencies are
not installed. The tests verify the app's wiring and configuration, not upstream
library behavior or model quality. OpenAI's constructor is checked in the source
contract only; no callable OpenAI adapter currently exists to test independently.
