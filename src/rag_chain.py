from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from src.config import LLM_MODEL
from src.retriever import EnterpriseHybridRerankRetriever
from src.guardrails import RAGGuardrails

FALLBACK_MESSAGE = (
    "I could not find information addressing your question in the available NexaCore policy documents. "
    "Please verify with HR or IT Support directly if this covers a specialized exception."
)


class QueryIntent(BaseModel):
    intent: Literal["CHITCHAT", "POLICY_QUERY"] = Field(
        description=(
            "Classify as 'CHITCHAT' ONLY for pure greetings, introductions, thanks, or questions "
            "about what this assistant can do. Classify ANY factual question, policy inquiry, "
            "benefit question, or domain query (even if unanswerable) as 'POLICY_QUERY'."
        )
    )


class EnterpriseKnowledgeAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0)
        self.retriever = EnterpriseHybridRerankRetriever()
        self.guardrails = RAGGuardrails()

        # Intent Router: separates greetings/introductions from knowledge queries
        self.intent_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0).with_structured_output(
            QueryIntent,
            method="json_schema",
            strict=True,
        )
        self.intent_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Classify the user's message.\n"
             "- Return 'CHITCHAT' ONLY if the user is greeting, introducing themselves, saying thanks, "
             "or asking about your general capabilities.\n"
             "- Return 'POLICY_QUERY' for ANY question seeking facts, rules, limits, procedures, "
             "IT help, HR benefits, or company details—even if it sounds unrelated to NexaCore."),
            ("human", "{input}"),
        ])

        self.chitchat_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the NexaCore Enterprise Knowledge Assistant. Respond warmly and concisely to "
             "greetings or introductions. Mention that you can answer questions about NexaCore's "
             "Leave Policy, Travel & Expenses, IT Security/BYOD, Employee Benefits, and Code of Conduct. "
             "Never invent company policies or facts."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.chitchat_chain = self.chitchat_prompt | self.llm | StrOutputParser()

        # Step 1: Contextualize follow-up queries using conversation history
        self.contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Given the chat history and the latest user question, rewrite the question into a "
             "standalone search query that includes all necessary context (such as policy names or topics "
             "discussed earlier). Do NOT answer the question. If it is already standalone, return it unchanged."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.query_rewriter = self.contextualize_prompt | self.llm | StrOutputParser()

        # Step 2: Grounded QA Prompt with strict anti-hallucination rules
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are the NexaCore Enterprise Knowledge Assistant. Answer the employee's question "
             "STRICTLY using the retrieved policy excerpts provided below.\n\n"
             "CRITICAL RULES:\n"
             "1. Rely ONLY on facts explicitly stated in the CONTEXT. Do not use outside knowledge or extrapolate.\n"
             "2. Cite the exact source document name inline using brackets, e.g. [Leave_Policy.pdf].\n"
             "3. If the CONTEXT does not contain sufficient information to answer the question, respond ONLY with:\n"
             f'   "{FALLBACK_MESSAGE}"\n'
             "4. Keep answers structured, concise, and actionable.\n\n"
             "CONTEXT:\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.answer_chain = self.qa_prompt | self.llm | StrOutputParser()

    def _classify_intent(self, query: str) -> Literal["CHITCHAT", "POLICY_QUERY"]:
        raw = (self.intent_prompt | self.intent_llm).invoke({"input": query})
        parsed = raw if isinstance(raw, QueryIntent) else QueryIntent.model_validate(raw)
        return parsed.intent

    @staticmethod
    def _format_docs_with_metadata(docs) -> str:
        formatted = []
        for idx, doc in enumerate(docs, start=1):
            src = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 1)
            formatted.append(f"[Excerpt {idx} | Source: {src} | Page: {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)

    def ask(self, user_query: str, chat_history: list[dict]) -> dict:
        # 1. Input Guardrail
        input_check = self.guardrails.validate_input(user_query)
        if not input_check.allowed:
            return {
                "answer": f"⚠️ {input_check.reason}",
                "standalone_query": user_query,
                "sources": [],
                "retrieved_chunks": [],
                "grounding_passed": True,
                "blocked": True,
            }

        clean_query = input_check.sanitized_query

        lc_history = []
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                lc_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                lc_history.append(AIMessage(content=msg["content"]))

        # 2. Intent Routing: handle greetings/introductions gracefully without retrieval
        if self._classify_intent(clean_query) == "CHITCHAT":
            greeting_reply = self.chitchat_chain.invoke({
                "chat_history": lc_history,
                "input": clean_query,
            })
            return {
                "answer": greeting_reply,
                "standalone_query": "[CHITCHAT_BYPASS]",
                "sources": [],
                "retrieved_chunks": [],
                "hybrid_candidate_count": 0,
                "grounding_passed": True,
                "blocked": False,
            }

        # 3. History-Aware Query Rewriting
        standalone_query = (
            self.query_rewriter.invoke({"chat_history": lc_history, "input": clean_query})
            if lc_history
            else clean_query
        )

        # 4. Hybrid Search + OpenAI Reranking
        retrieval_output = self.retriever.retrieve_with_diagnostics(standalone_query)
        reranked_docs = retrieval_output["reranked_docs"]

        if not reranked_docs:
            return {
                "answer": FALLBACK_MESSAGE,
                "standalone_query": standalone_query,
                "sources": [],
                "retrieved_chunks": [],
                "hybrid_candidate_count": retrieval_output["hybrid_candidate_count"],
                "grounding_passed": True,
                "blocked": False,
            }

        context_str = self._format_docs_with_metadata(reranked_docs)

        # 5. Grounded Answer Generation
        draft_answer = self.answer_chain.invoke({
            "context": context_str,
            "chat_history": lc_history,
            "input": clean_query,
        })

        # 6. Output Guardrails: OpenAI Moderation + Grounding Faithfulness Audit
        output_audit = self.guardrails.validate_output(context=context_str, answer=draft_answer)
        if not output_audit.passed:
            final_answer = (
                f"{FALLBACK_MESSAGE}\n\n"
                f"*(Guardrail Notice: {output_audit.reason})*"
            )
            unique_sources = []
        else:
            final_answer = draft_answer
            if FALLBACK_MESSAGE in final_answer:
                unique_sources = []
            else:
                unique_sources = sorted({
                    f"{d.metadata.get('source')} (Page {d.metadata.get('page', 1)})"
                    for d in reranked_docs
                })

        return {
            "answer": final_answer,
            "standalone_query": standalone_query,
            "sources": unique_sources,
            "retrieved_chunks": reranked_docs,
            "hybrid_candidate_count": retrieval_output["hybrid_candidate_count"],
            "grounding_passed": output_audit.passed,
            "blocked": False,
        }