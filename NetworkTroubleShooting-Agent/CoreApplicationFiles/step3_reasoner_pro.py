# step3_reasoner_pro.py
"""
Reasoner Agent (Agent 2)
Return dict with keys: query, mode, best_score, summary, answer, results
"""

from step2_retriever_qdrant import retrieve_from_qdrant

# Thresholds (should match LangGraph)
HIGH_CONF = 0.5
MEDIUM_CONF = 0.35

FALLBACK_STEPS = [
    "Check physical connections (LAN cables, patch panels, RJ45 ports).",
    "Restart the affected network device (router/switch) and verify power.",
    "Verify device IP configuration (IP, subnet mask, gateway) and DHCP status.",
    "Test connectivity using ping, tracert (tracepath) to check latency and hops.",
    "Check DNS settings: try nslookup/dig and flush DNS cache on client.",
    "Review firewall, ACLs, VPN or proxy rules that could block traffic.",
    "Check recent configuration changes or firmware upgrades and roll back if needed.",
    "Collect logs from network device/system and inspect for errors (timestamps)."
]

def _call_llm_for_reasoning(prompt: str) -> str:
    """
    Placeholder for LLM-based reasoning. For now we return a synthesized text using heuristics.
    Replace this with Bedrock/GPT call when ready.
    """
    # Simple heuristic "synthesis"
    return "Synthesized reasoning:\n" + prompt[:800]

def reason(query: str):
    # Get top retrieved records to inform reasoning
    results = retrieve_from_qdrant(query, limit=5)
    best = results[0] if results else {}
    score = float(best.get("score", 0.0))

    if score >= HIGH_CONF:
        mode = "DIRECT"
        summary = f"High confidence ({score:.2f}). Direct solution found."
        answer = best.get("solution", "No solution found.")
        combined_results = results
    elif score >= MEDIUM_CONF:
        mode = "HYBRID"
        summary = f"Medium confidence ({score:.2f}). Combining retrieved insight with fallback steps."
        # combine retrieved solution + top fallback steps
        retrieved = best.get("solution", "")
        fallback = "\n".join(f"- {s}" for s in FALLBACK_STEPS[:4])
        # Optionally run a small LLM synthesis summarizing both — here lightweight
        llm_summary = _call_llm_for_reasoning(retrieved + "\n\n" + fallback)
        answer = f"Partial match found:\n\n{retrieved}\n\n{llm_summary}\n\n{fallback}"
        combined_results = results
    else:
        mode = "FALLBACK"
        summary = f"Low confidence ({score:.2f}). Using fallback troubleshooting."
        # Use heuristic fallback steps; optionally call LLM to structure them
        llm_text = _call_llm_for_reasoning("\n".join(FALLBACK_STEPS[:6]))
        answer = "No close match found.\n" + llm_text + "\n\n" + "\n".join(f"- {s}" for s in FALLBACK_STEPS)
        combined_results = results

    return {
        "query": query,
        "mode": mode,
        "best_score": score,
        "summary": summary,
        "answer": answer,
        "results": combined_results
    }

if __name__ == "__main__":
    q = input("Enter network issue:\n> ")
    o = reason(q)
    print(o["summary"])
    print("\n", o["answer"])
