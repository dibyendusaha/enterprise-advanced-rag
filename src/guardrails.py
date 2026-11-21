import re
from openai import OpenAI
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from src.config import LLM_MODEL, MODERATION_MODEL, MODERATION_SCORE_THRESHOLD


class ModerationCheckResult(BaseModel):
    flagged: bool = False
    flagged_categories: list[str] = Field(default_factory=list)
    max_score: float = 0.0


class InputCheckResult(BaseModel):
    allowed: bool
    reason: str = ""
    sanitized_query: str = ""
    moderation: ModerationCheckResult = Field(default_factory=ModerationCheckResult)


class GroundingAuditResult(BaseModel):
    is_grounded: bool = Field(
        description="True if every factual claim in the answer is directly supported by the provided context."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="List of any claims not found in the context."
    )


class OutputCheckResult(BaseModel):
    passed: bool
    reason: str = ""
    grounding: GroundingAuditResult
    moderation: ModerationCheckResult = Field(default_factory=ModerationCheckResult)


class RAGGuardrails:
    INJECTION_PATTERNS = [
        r"ignore (all )?(previous|above|prior) instructions",
        r"disregard your system prompt",
        r"you are now (dan|developer mode|unrestricted)",
        r"reveal your (system prompt|hidden instructions)",
        r"bypass (safety|guardrails|rules)",
    ]

    PII_PATTERNS = {
        "Credit/Debit Card": r"\b(?:\d[ -]*?){13,16}\b",
        "Indian PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
        "US SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    def __init__(self, moderation_threshold: float = MODERATION_SCORE_THRESHOLD):
        self.openai_client = OpenAI()
        self.moderation_model = MODERATION_MODEL
        self.moderation_threshold = moderation_threshold

        self.judge_llm = ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(
            GroundingAuditResult
        )
        self.audit_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict compliance auditor. Verify whether the ASSISTANT ANSWER is 100% grounded "
             "in the RETRIEVED CONTEXT. If the assistant states that the information is not found in the "
             "company documents, mark `is_grounded=True`. Flag any external facts, numbers, or assumptions "
             "not explicitly present in the context."),
            ("human", "RETRIEVED CONTEXT:\n{context}\n\nASSISTANT ANSWER:\n{answer}")
        ])

    def moderate_text(self, text: str) -> ModerationCheckResult:
        """
        Calls OpenAI's official Moderation API (`omni-moderation-latest`) and checks
        both the model's `flagged` boolean and calibrated `category_scores`.
        """
        response = self.openai_client.moderations.create(
            model=self.moderation_model,
            input=text,
        )
        result = response.results[0]
        categories_dict = result.categories.model_dump()
        scores_dict = result.category_scores.model_dump()

        flagged_cats = [
            cat for cat, is_flagged in categories_dict.items()
            if is_flagged or scores_dict.get(cat, 0.0) >= self.moderation_threshold
        ]
        max_score = max(scores_dict.values()) if scores_dict else 0.0
        is_blocked = bool(result.flagged or flagged_cats)

        return ModerationCheckResult(
            flagged=is_blocked,
            flagged_categories=sorted(flagged_cats),
            max_score=round(max_score, 4),
        )

    def validate_input(self, query: str) -> InputCheckResult:
        """
        Three-layer input gate:
        1. Length & empty validation
        2. Prompt-injection / jailbreak pattern check
        3. OpenAI `omni-moderation-latest` content safety check
        4. Regex PII redaction (PAN, SSN, Credit Cards)
        """
        cleaned = query.strip()
        if not cleaned:
            return InputCheckResult(allowed=False, reason="Query cannot be empty.")

        if len(cleaned) > 1500:
            return InputCheckResult(
                allowed=False,
                reason="Query exceeds maximum allowed length of 1,500 characters."
            )

        lowered = cleaned.lower()
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, lowered):
                return InputCheckResult(
                    allowed=False,
                    reason="Security Guardrail Triggered: Potential prompt injection detected."
                )

        # OpenAI Official Moderation API check
        mod_result = self.moderate_text(cleaned)
        if mod_result.flagged:
            cats = ", ".join(mod_result.flagged_categories) or "policy_violation"
            return InputCheckResult(
                allowed=False,
                reason=(
                    f"Content Safety Guardrail Triggered (`{self.moderation_model}`): "
                    f"Flagged under [{cats}] (max confidence: {mod_result.max_score:.2f})."
                ),
                moderation=mod_result,
            )

        # Redact accidental PII before sending to retrieval/LLM stages
        sanitized = cleaned
        for pii_label, pattern in self.PII_PATTERNS.items():
            sanitized = re.sub(
                pattern,
                f"[REDACTED_{pii_label.upper().replace(' ', '_')}]",
                sanitized,
            )

        return InputCheckResult(
            allowed=True,
            sanitized_query=sanitized,
            moderation=mod_result,
        )

    def verify_grounding(self, context: str, answer: str) -> GroundingAuditResult:
        chain = self.audit_prompt | self.judge_llm
        raw_result = chain.invoke({"context": context, "answer": answer})
        return (
            raw_result
            if isinstance(raw_result, GroundingAuditResult)
            else GroundingAuditResult.model_validate(raw_result)
        )

    def validate_output(self, context: str, answer: str) -> OutputCheckResult:
        """
        Dual output guardrail:
        1. OpenAI `omni-moderation-latest` safety check on generated response
        2. LLM-as-a-Judge faithfulness / anti-hallucination audit
        """
        mod_result = self.moderate_text(answer)
        grounding_result = self.verify_grounding(context=context, answer=answer)

        if mod_result.flagged:
            cats = ", ".join(mod_result.flagged_categories) or "policy_violation"
            return OutputCheckResult(
                passed=False,
                reason=f"Output blocked by `{self.moderation_model}` [{cats}].",
                grounding=grounding_result,
                moderation=mod_result,
            )

        if not grounding_result.is_grounded:
            return OutputCheckResult(
                passed=False,
                reason="Output suppressed: contains claims not verified in retrieved documents.",
                grounding=grounding_result,
                moderation=mod_result,
            )

        return OutputCheckResult(
            passed=True,
            grounding=grounding_result,
            moderation=mod_result,
        )