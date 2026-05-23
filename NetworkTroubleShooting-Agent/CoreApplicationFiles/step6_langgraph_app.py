"""
Streamlit Interface for BlueCom Network Troubleshooter Agent
-------------------------------------------------------------
Agent 1 (Retriever) + Agent 2 (Local Reasoner using Google Gemma 2B-Instruct)
Logs feedback locally and uploads it to S3.
"""

import os
import csv
from datetime import datetime
import streamlit as st
import boto3
from dotenv import load_dotenv
from step5_langgraph_triple import app

# Load environment variables
load_dotenv()

# ==============================
# ☁️ S3 Upload Helper
# ==============================
def upload_feedback_to_s3(local_path="logs/feedback.csv"):
    """Upload feedback file to S3 bucket."""
    try:
        # Get credentials from environment variables
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
        bucket_name = os.getenv("S3_BUCKET_NAME", "netro-network-data")
        
        if not aws_access_key or not aws_secret_key:
            st.warning("⚠️ AWS credentials not configured. Feedback saved locally only.")
            return
        
        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        s3_key = f"feedback_logs/{timestamp}_feedback.csv"

        print(f"⬆️ Uploading {local_path} to s3://{bucket_name}/{s3_key}")
        s3.upload_file(local_path, bucket_name, s3_key)
        st.success(f"✅ Feedback uploaded to S3 as {s3_key}")
    except Exception as e:
        st.warning(f"⚠️ Failed to upload feedback to S3: {e}")
        print(f"❌ Upload error: {e}")


# ==============================
# ⚙️ App Initialization
# ==============================
st.set_page_config(page_title="BlueCom Troubleshooter", page_icon="⚡", layout="wide")
st.title("⚡ BlueCom Network Troubleshooter Agent")
st.markdown("🧠 Dual-Agent System — Agent 1 (Retriever) + Agent 2 (Local Gemma 2B-Instruct Reasoner)")

# Ensure log directory exists
os.makedirs("logs", exist_ok=True)
feedback_file = os.path.join("logs", "feedback.csv")

# Initialize session state
if "result" not in st.session_state:
    st.session_state.result = None
if "query" not in st.session_state:
    st.session_state.query = ""


# ==============================
# 💬 Input Section
# ==============================
query = st.text_area(
    "💬 Describe your network issue:",
    height=150,
    placeholder="e.g., router not assigning IP address or slow Wi-Fi performance...",
    value=st.session_state.query
)

if st.button("🔍 Analyze Issue"):
    if not query.strip():
        st.warning("Please enter a problem description.")
    else:
        st.session_state.query = query
        with st.spinner("🤖 Engaging Dual Agents (Retriever + Gemma Reasoner)..."):
            result = app.invoke({"query": query})
            st.session_state.result = result


# ==============================
# 📊 Display Results
# ==============================
if st.session_state.result:
    result = st.session_state.result

    confidence = result.get("confidence", 0.0)
    mode = result.get("mode", "unknown").upper()
    reasoner_output = result.get("reasoner_output", {})
    final_answer = reasoner_output.get("answer", "No answer found.")
    summary = reasoner_output.get("summary", "Generated locally using Google Gemma 2B-Instruct model")
    retriever_results = reasoner_output.get("retrieved_solutions", result.get("results", []))

    # 🎯 Confidence & Mode
    st.subheader("🎯 Confidence & Mode")
    col1, col2 = st.columns([1, 3])

    with col1:
        if confidence >= 0.5:
            st.success(f"🟢 High Confidence ({confidence:.2f}) — {mode}")
        elif confidence >= 0.4:
            st.info(f"🟡 Medium Confidence ({confidence:.2f}) — {mode}")
        else:
            st.warning(f"🔴 Low Confidence ({confidence:.2f}) — {mode}")
        st.caption(f"**Agent Mode:** {mode.title()}")

    with col2:
        st.markdown(f"**Summary:** {summary}")

    st.markdown("---")

    # 🧩 Dual-Agent Outputs
    st.markdown("## 🧩 Dual-Agent Outputs")
    left, right = st.columns(2)

    # Agent 1 — Retriever
    with left:
        st.markdown("### Agent 1 — Retriever 🔍")
        if retriever_results:
            for idx, r in enumerate(retriever_results, start=1):
                score = r.get("score", 0.0)
                src = r.get("source", "")
                problem_text = r.get('problem', '').strip()
                solution_text = r.get('solution', '').strip()
                
                # Show appropriate label based on what's available
                result_label = f"Result {idx} | Score: {score:.2f} | Source: {src}"
                
                with st.expander(result_label):
                    if problem_text:
                        st.markdown(f"**Problem:** {problem_text[:500]}")
                    else:
                        st.markdown("**Problem:** _(Not available)_")
                    
                    if solution_text:
                        st.markdown(f"**Solution:** {solution_text[:800]}")
                    else:
                        st.markdown("**Solution:** _(Not available)_")
                    
                    product_id = r.get('product_id', '').strip()
                    doc_id = r.get('doc_id', '').strip()
                    
                    if product_id or doc_id:
                        st.caption(f"**Product:** {product_id or 'N/A'} — **DocID:** {doc_id or 'N/A'}")
                    
                    # Warning if both are empty
                    if not problem_text and not solution_text:
                        st.warning("⚠️ This record appears to be empty. Data indexing may need attention.")
        else:
            st.info("No retrieved records found.")

    # Agent 2 — Reasoner (Gemma 2B)
    with right:
        st.markdown("### Agent 2 — Local Reasoner 🧠 (Google Gemma 2B-Instruct Model)")
        st.markdown("**Generated Troubleshooting Steps:**")
        st.write(final_answer)

    with st.expander("🧠 Full Reasoner Output (Raw JSON)"):
        st.json(reasoner_output)

    st.markdown("---")

    # ✅ Feedback Section
    st.markdown("## ✅ Feedback (Help Us Improve)")
    fcol1, fcol2 = st.columns([1, 1])

    def record_feedback(status: str):
        """Write feedback locally and upload to S3."""
        os.makedirs("logs", exist_ok=True)
        with open(feedback_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                st.session_state.query,
                confidence,
                mode,
                status,
                final_answer
            ])
        st.success(f"✅ Feedback recorded ({status}).")
        upload_feedback_to_s3(feedback_file)

    if fcol1.button("👍 Solution Worked"):
        record_feedback("worked")

    if fcol2.button("❗ Needs Manual Review"):
        record_feedback("review")

    st.caption("Logs are saved locally in `logs/feedback.csv` and synced to S3.")


# ==============================
# 🧾 Footer
# ==============================
st.markdown("---")
st.caption("Developed by BlueCom AI Team • Powered by LangGraph + Google Gemma 2B 🚀")