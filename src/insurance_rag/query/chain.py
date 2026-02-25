"""RAG chain with local Hugging Face LLM (Phase 4)."""
import functools
from collections.abc import Callable
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from insurance_rag.config import (
    LOCAL_LLM_DEVICE,
    LOCAL_LLM_MAX_NEW_TOKENS,
    LOCAL_LLM_MODEL,
    LOCAL_LLM_REPETITION_PENALTY,
)
from insurance_rag.query.retriever import get_retriever

_DEFAULT_SYSTEM_PROMPT = (
    "You are an insurance industry assistant. "
    "Answer the user's question using ONLY the provided context. "
    "Cite sources using [1], [2], etc. corresponding to the numbered context items. "
    "If the context is insufficient to answer, say so explicitly. "
    "This is not legal or medical advice."
)


def _resolve_system_prompt(domain_name: str | None = None) -> str:
    """Return the system prompt for the given domain (or the default domain)."""
    if domain_name is None:
        from insurance_rag.config import DEFAULT_DOMAIN

        domain_name = DEFAULT_DOMAIN
    try:
        from insurance_rag.domains import get_domain

        return get_domain(domain_name).get_system_prompt()
    except (KeyError, ImportError):
        return _DEFAULT_SYSTEM_PROMPT

USER_PROMPT = """Context:
{context}

Question: {question}"""


def _invoke_chain(prompt: ChatPromptTemplate, llm: Any, input_dict: dict) -> Any:
    """Invoke prompt | llm. Extracted for testability."""
    return (prompt | llm).invoke(input_dict)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs)
    )


@functools.lru_cache(maxsize=1)
def _create_llm() -> ChatHuggingFace:
    """Create local chat model using Hugging Face pipeline (no API key).
    Device placement via model_kwargs device_map to avoid conflicts with accelerate.
    Cached so the model is loaded once per process.
    """
    model_kwargs: dict = {}
    if LOCAL_LLM_DEVICE == "auto":
        model_kwargs["device_map"] = "auto"
    elif LOCAL_LLM_DEVICE == "cpu":
        model_kwargs["device_map"] = "cpu"
    else:
        model_kwargs["device_map"] = LOCAL_LLM_DEVICE
    llm = HuggingFacePipeline.from_model_id(
        model_id=LOCAL_LLM_MODEL,
        task="text-generation",
        model_kwargs=model_kwargs,
        pipeline_kwargs=dict(
            max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=LOCAL_LLM_REPETITION_PENALTY,
        ),
    )
    return ChatHuggingFace(llm=llm)


def build_rag_chain(
    retriever: Any = None,
    k: int = 8,
    metadata_filter: dict | None = None,
    system_prompt: str | None = None,
    domain_name: str | None = None,
    store: Any = None,
    embeddings: Any = None,
) -> Callable[[dict], dict]:
    """Build an LCEL RAG chain.

    Returns a runnable that takes ``{"question": str}`` and returns
    ``{"answer": str, "source_documents": list[Document]}``.

    *system_prompt* overrides the domain's prompt when provided.
    *domain_name* selects which domain's prompt and retriever to use.
    *store* and *embeddings* are passed to get_retriever when building
    a new retriever (for domain-specific collection).
    """
    if retriever is None:
        retriever = get_retriever(
            k=k,
            metadata_filter=metadata_filter,
            store=store,
            embeddings=embeddings,
            domain_name=domain_name,
        )
    if system_prompt is None:
        system_prompt = _resolve_system_prompt(domain_name)
    llm = _create_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", USER_PROMPT),
        ]
    )

    def runnable_invoke(input_dict: dict) -> dict:
        question = input_dict.get("question", "")
        docs = retriever.invoke(question)
        context = _format_context(docs)
        response = _invoke_chain(prompt, llm, {"context": context, "question": question})
        content = getattr(response, "content", None)
        return {
            "answer": content if content is not None else str(response),
            "source_documents": docs,
        }

    return runnable_invoke


def run_rag(
    question: str,
    retriever: Any = None,
    k: int = 8,
    metadata_filter: dict | None = None,
    domain_name: str | None = None,
    store: Any = None,
    embeddings: Any = None,
) -> tuple[str, list[Document]]:
    """Run the RAG chain for one question. Returns (answer, source_documents)."""
    invoke = build_rag_chain(
        retriever=retriever,
        k=k,
        metadata_filter=metadata_filter,
        domain_name=domain_name,
        store=store,
        embeddings=embeddings,
    )
    result = invoke({"question": question})
    return result["answer"], result["source_documents"]
