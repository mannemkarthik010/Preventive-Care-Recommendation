import os
from typing import List, Type, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from rag import get_vector_store
from schemas import CarePlan, Citation, CritiqueResult, PatientProfile, RiskAssessment
from utils import calculate_ascvd_10yr_risk

load_dotenv()

MAX_REVISIONS = 2


class AgentState(TypedDict, total=False):
    patient: PatientProfile
    risk: RiskAssessment
    citations: List[Citation]
    guideline_context: str
    plan: CarePlan
    critique: CritiqueResult
    revision_count: int


def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.environ.get("HF_CHAT_MODEL"),
        base_url=os.environ.get("HF_CHAT_URL"),
        api_key=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0,
    )


def _structured_invoke(schema: Type[BaseModel], prompt: str) -> BaseModel:
    """Native tool-calling structured output, with a JSON-mode fallback for
    inference backends that don't support forced tool_choice."""
    llm = _llm()
    try:
        return llm.with_structured_output(schema, method="function_calling").invoke(prompt)
    except Exception:
        json_prompt = (
            f"{prompt}\n\nRespond with ONLY a single valid JSON object matching this schema "
            f"(no prose, no markdown fences):\n{schema.model_json_schema()}"
        )
        return llm.with_structured_output(schema, method="json_mode").invoke(json_prompt)


# --- Graph nodes ---

def assess_risk(state: AgentState) -> dict:
    p = state["patient"]
    result = calculate_ascvd_10yr_risk(
        p.age, p.sex, p.total_cholesterol, p.hdl_cholesterol, p.systolic_bp,
        p.smoker, p.diabetes, p.bp_meds, race=p.race,
    )
    return {"risk": RiskAssessment(**result)}


def retrieve_guidelines(state: AgentState) -> dict:
    p, risk = state["patient"], state["risk"]
    query = (
        f"Preventive cardiovascular care for a {p.age}-year-old {p.sex} with "
        f"{risk.risk_tier.lower()} 10-year ASCVD risk ({risk.risk_percentage}%), "
        f"smoker={p.smoker}, diabetes={p.diabetes}, on BP medication={p.bp_meds}."
    )
    db = get_vector_store()
    docs = db.similarity_search(query, k=5)

    citations = [
        Citation(source=d.metadata.get("source", "Unknown Document"), page=d.metadata.get("page", 0))
        for d in docs
    ]
    context = "\n\n".join(
        f"Source: {c.source} (Page {c.page})\nContent: {d.page_content}"
        for c, d in zip(citations, docs)
    )
    return {
        "citations": citations,
        "guideline_context": context if context else "No relevant guidelines found.",
    }


def synthesize_plan(state: AgentState) -> dict:
    p, risk = state["patient"], state["risk"]
    critique = state.get("critique")
    revision_note = (
        f"\n\nA prior draft was rejected for this reason — address it directly: {critique.feedback}"
        if critique and not critique.passes
        else ""
    )

    prompt = f"""You are a clinical recommendation synthesizer. Build a structured 10-year
preventive cardiovascular care plan for this patient, grounded ONLY in the guideline
excerpts below. Every recommendation must be traceable to the excerpts.

Patient: {p.age}-year-old {p.sex}, 10-year ASCVD risk {risk.risk_percentage}% ({risk.risk_tier} tier).
Smoker: {p.smoker}. Diabetes: {p.diabetes}. On BP medication: {p.bp_meds}.

Guideline excerpts:
{state['guideline_context']}
{revision_note}"""

    plan = _structured_invoke(CarePlan, prompt)
    return {"plan": plan}


def critique_plan(state: AgentState) -> dict:
    plan, risk = state["plan"], state["risk"]
    prompt = f"""Evaluate this draft preventive-care plan against the rubric:
1. Contains all four sections (screening, lifestyle, pharmacological, follow-up) with substantive content.
2. Is grounded in and consistent with the retrieved guideline excerpts.
3. Is tailored to the patient's {risk.risk_tier} risk tier, not generic.
4. Contains no contraindicated suggestions.

Draft plan:
{plan.model_dump_json(indent=2)}

Guideline excerpts used:
{state['guideline_context']}"""

    result = _structured_invoke(CritiqueResult, prompt)
    revision_count = state.get("revision_count", 0)
    if not result.passes:
        revision_count += 1
    return {"critique": result, "revision_count": revision_count}


def _route_after_critique(state: AgentState) -> str:
    critique = state["critique"]
    if critique.passes:
        return "end"
    if state.get("revision_count", 0) > MAX_REVISIONS:
        return "end"
    return "revise"


# --- Graph construction ---

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("assess_risk", assess_risk)
    graph.add_node("retrieve_guidelines", retrieve_guidelines)
    graph.add_node("synthesize_plan", synthesize_plan)
    graph.add_node("critique_plan", critique_plan)

    graph.set_entry_point("assess_risk")
    graph.add_edge("assess_risk", "retrieve_guidelines")
    graph.add_edge("retrieve_guidelines", "synthesize_plan")
    graph.add_edge("synthesize_plan", "critique_plan")
    graph.add_conditional_edges(
        "critique_plan", _route_after_critique, {"revise": "synthesize_plan", "end": END}
    )
    return graph.compile()


def create_agent():
    return build_graph()


def run_agent(patient: PatientProfile) -> AgentState:
    graph = create_agent()
    return graph.invoke({"patient": patient, "revision_count": 0})
