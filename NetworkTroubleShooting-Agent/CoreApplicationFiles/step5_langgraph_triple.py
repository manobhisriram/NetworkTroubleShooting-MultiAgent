"""
LangGraph Dual-Agent System (Local LLM Integrated — Gemma 2B)
-------------------------------------------------------------
Enhanced dual-agent system using Qdrant retriever + local HuggingFace Gemma-2B-Instruct reasoner.
Automatically handles retriever confidence, hybrid reasoning, and fallback logic.
"""

from dotenv import load_dotenv
load_dotenv()

import json
from langgraph.graph import StateGraph, END
from typing import Dict, Any
from step2_retriever_qdrant import retrieve_from_qdrant
from transformers import pipeline

# ========================================
# 🧠 Load Local HuggingFace LLM (Gemma 2B)
# ========================================
print("🧠 Loading local reasoning model (google/gemma-2b-it)...")
try:
    llm = pipeline("text-generation", model="google/gemma-2b-it")
    print("✅ Gemma 2B-Instruct model loaded successfully.")
except Exception as e:
    print(f"⚠️ LLM load failed: {e}")
    llm = None

# ========================================
# ⚙️ Configuration
# ========================================
HIGH_CONF = 0.5
MEDIUM_CONF = 0.35


class AgentState(dict):
    """Holds shared state between retriever and reasoner agents."""
    query: str
    results: list
    reasoner_output: dict
    confidence: float
    mode: str


# ========================================
# 🔍 Retriever Node
# ========================================
def retriever_node(state: AgentState):
    print("\n🔍 Agent 1 (Retriever) active...")
    q = state.get("query", "")
    results = retrieve_from_qdrant(q)
    state["results"] = results

    top_score = float(results[0].get("score", 0.0)) if results else 0.0
    state["confidence"] = top_score
    print(f"Retrieved {len(results)} records; top_score = {top_score:.4f}")
    return state


# ========================================
# 🧠 Reasoner Node (Gemma 2B)
# ========================================
def reason_with_local_llm(query, retrieved_docs, mode):
    """
    Uses Gemma 2B Instruct to generate structured troubleshooting steps.
    Produces technically sound, human-readable, and relevant solutions.
    """
    FALLBACK_STEPS = [
        "Check cable and power connections.",
        "Restart the affected network device.",
        "Verify IP assignment and DHCP lease.",
        "Test DNS and routing paths.",
        "Review firewall, VLAN, or VPN settings.",
        "Check logs and recent firmware updates."
    ]

    # Prepare retriever context - filter out empty docs
    valid_docs = [doc for doc in retrieved_docs[:3] if doc.get('problem', '').strip() or doc.get('solution', '').strip()]
    
    if valid_docs:
        retriever_context = "\n\n".join([
            f"Problem: {doc.get('problem','N/A')}\nSolution: {doc.get('solution','N/A')}"
            for doc in valid_docs
        ])
    else:
        retriever_context = "No historical data found."

    # Core system prompt
    bluecom_context = (
        "You are part of the BlueCom Network Troubleshooter Agent Team — "
        "an AI assistant that helps telecom engineers diagnose and fix network issues. "
        "Respond clearly, using 3–5 short, numbered steps. "
        "Focus on practical technical reasoning."
    )

    # Dynamic prompt logic
    if mode == "retriever-only":
        prompt = f"""
{bluecom_context}

Customer issue: {query}

Relevant historical match:
{retriever_context}

Provide a short, direct fix or summary based on the retrieved solution.
"""
    elif mode == "hybrid":
        prompt = f"""
{bluecom_context}

Customer issue: {query}

Similar known issues and resolutions:
{retriever_context}

Generate 3–5 concise troubleshooting steps combining past data with logical technical reasoning.
"""
    else:
        prompt = f"""
{bluecom_context}

Customer issue: {query}

No historical matches found.
Generate 3–5 helpful troubleshooting steps that are general, logical, and relevant to the issue.
"""

    # --- Call Gemma 2B ---
    if llm:
        try:
            response = llm(
                prompt.strip(),
                max_new_tokens=220,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.5
            )[0]["generated_text"]

            response = response.strip().replace("\n\n", "\n")
            if len(response) < 30:
                response = "\n".join(FALLBACK_STEPS)
            return response

        except Exception as e:
            return f"⚠️ Gemma generation failed: {e}\nFallback:\n" + "\n".join(FALLBACK_STEPS)
    else:
        return "⚠️ Gemma unavailable — fallback steps:\n" + "\n".join(FALLBACK_STEPS)


# ========================================
# 🧠 Reasoner Node Wrapper
# ========================================
def reasoner_node(state: AgentState):
    print("🧠 Agent 2 (Reasoner via local Gemma 2B) active...")
    q = state.get("query", "")
    retrieved_docs = state.get("results", [])
    mode = state.get("mode", "fallback")

    reasoning_output = reason_with_local_llm(q, retrieved_docs, mode)

    state["reasoner_output"] = {
        "query": q,
        "mode": mode,
        "answer": reasoning_output,
        "summary": "Generated locally using Google Gemma 2B-Instruct model",
        "retrieved_solutions": retrieved_docs[:3] if retrieved_docs else [],
    }
    print("✅ Local Gemma Reasoner produced output.")
    return state


# ========================================
# 🤖 Decision Logic
# ========================================
def decide_next(state: AgentState):
    results = state.get("results", [])
    top_score = float(state.get("confidence", 0.0))

    if not results:
        print("⚠️ No retriever results → Gemma fallback")
        state["mode"] = "fallback"
        state["confidence"] = 0.0
        return "reasoner"

    top_result = results[0]
    if top_result.get("source", "").lower() == "error" or top_score == 0.0:
        print("❌ Retrieval failed → Gemma fallback")
        state["mode"] = "fallback"
        return "reasoner"

    print(f"🎯 Top score = {top_score:.4f}")

    if top_score >= HIGH_CONF:
        print("✅ High confidence → retriever-only mode")
        state["mode"] = "retriever-only"
        state["reasoner_output"] = {
            "query": state["query"],
            "mode": "retriever-only",
            "best_score": top_score,
            "summary": "High confidence retriever answer used directly.",
            "answer": top_result.get("solution", "No solution found."),
            "retrieved_solutions": results[:3],
        }
        return END

    if MEDIUM_CONF <= top_score < HIGH_CONF:
        print("🟡 Medium confidence → hybrid mode (combine retriever + Gemma)")
        state["mode"] = "hybrid"
        return "reasoner"

    print("🔴 Low confidence → fallback mode (Gemma only)")
    state["mode"] = "fallback"
    return "reasoner"


# ========================================
# 🧩 Graph Definition
# ========================================
graph = StateGraph(AgentState)
graph.add_node("retriever", retriever_node)
graph.add_node("reasoner", reasoner_node)
graph.set_entry_point("retriever")
graph.add_conditional_edges("retriever", decide_next)
graph.add_edge("reasoner", END)
app = graph.compile()


# ========================================
# 🧾 CLI Test Runner (for debugging)
# ========================================
if __name__ == "__main__":
    print("\n💬 BlueCom Dual-Agent (Retriever + Gemma 2B)\n--------------------------------------------")
    q = input("Describe your network issue:\n> ")
    state = app.invoke({"query": q})

    out = state.get("reasoner_output", {})
    confidence = state.get("confidence", out.get("best_score", 0.0))
    mode = state.get("mode", out.get("mode", "unknown"))
    final_answer = out.get("answer", "No answer found.")
    summary = out.get("summary", "")
    retrieved = out.get("retrieved_solutions", [])

    print("\n--- FINAL RESULT ---")
    print(f"Mode: {mode}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Summary: {summary}\n")

    if retrieved:
        print("📄 Retrieved context (top results):")
        for r in retrieved:
            print(f" - Problem: {r.get('problem','')[:80]}...")
            print(f"   Solution: {r.get('solution','')[:100]}...\n")

    print("🧠 Gemma Response:\n")
    print(final_answer)