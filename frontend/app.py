"""Streamlit UI for the Knowledge Enterprise Agent (Baseline RAG / Adaptive + Corrective + Verified RAG).

Run:  streamlit run frontend/app.py      (backend must be running: uvicorn backend.main:app)
"""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
STATUS_ICON = {"indexed": "✅", "processing": "⏳", "failed": "❌"}

st.set_page_config(page_title="Knowledge Enterprise Agent", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- API helpers
class APIError(Exception):
    pass


def api(method: str, path: str, *, timeout: int = 60, **kwargs):
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise APIError(f"Cannot reach the backend at {API_URL}. Is `uvicorn backend.main:app` running?")
    except requests.Timeout:
        raise APIError("The backend took too long to respond.")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        raise APIError(str(detail)[:500])
    return resp.json()


def fetch_health():
    try:
        return api("GET", "/health", timeout=10)
    except APIError as exc:
        return {"error": str(exc)}


def fetch_documents() -> list[dict]:
    try:
        return api("GET", "/documents", timeout=10)["documents"]
    except APIError:
        return []


# --------------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex)
ss.setdefault("messages", [])
ss.setdefault("uploader_key", 0)
ss.setdefault("upload_report", [])


# --------------------------------------------------------------------------- rendering
VERDICT_ICON = {"SUPPORTED": "✅", "CONTRADICTED": "❌", "INSUFFICIENT": "⚠️"}


def render_verification(v: dict, original: str | None) -> None:
    if v.get("error"):
        st.warning(f"Verification unavailable, answer is UNVERIFIED: {v['error']}")
        return
    c = v["counts"]
    st.caption(
        f"NLI model `{v['nli_model']}` · thresholds entail ≥ {v['thresholds'].get('entail')}, contradict ≥ "
        f"{v['thresholds'].get('contradict')} · {c.get('supported', 0)} supported / {c.get('contradicted', 0)} contradicted / "
        f"{c.get('insufficient', 0)} insufficient · citations valid {c.get('citations_valid', 0)}/{c.get('citations_total', 0)}"
    )
    for vc in v["claims"]:
        cl = vc["claim"]
        st.markdown(f"{VERDICT_ICON[vc['verdict']]} **{vc['verdict']}** — {cl['text']}")
        flags = [f"entail {vc['entailment']:.2f}", f"contradict {vc['contradiction']:.2f}", f"action: {vc['action']}"]
        if vc["conflict"]:
            flags.append("⚖️ conflicting evidence")
        if vc["demoted_by_numeric_guard"]:
            flags.append("number not found in evidence")
        if vc["reattributed"]:
            flags.append("citation re-attributed")
        st.caption(" · ".join(flags))
        ev = vc.get("supporting") or vc.get("contradicting") or (vc["evidence"][0] if vc["evidence"] else None)
        if ev:
            st.text(f"[{ev['filename']}, p. {ev['page_label']}] {ev['window_text']}")
        for ck in vc["citation_checks"]:
            st.caption(f"citation {ck['source_id']}: " + ("valid" if ck["exists"] and ck["page_valid"] and ck["supports_claim"] else f"INVALID ({ck['reason']})"))
        st.divider()
    if v["unavailable_statements"]:
        st.info("Not covered by documents: " + " ".join(v["unavailable_statements"]))
    if original and original != v["final_answer"]:
        with st.expander("Original (pre-verification) answer"):
            st.markdown(original)


def render_trace(t: dict) -> None:
    if t.get("router"):
        r = t["router"]
        st.markdown(f"**Adaptive routing:** `{r['query_type']}` — {r.get('reasoning', '')}")
        for i, q in enumerate(r.get("sub_questions", []), 1):
            st.markdown(f"&nbsp;&nbsp;{i}. {q}")
        if r.get("error"):
            st.warning(f"Router fell back to SIMPLE: {r['error']}")
    for u in t.get("retrieval_units", []):
        st.markdown(f"**Search:** _{u['query']}_")
        corr = u.get("corrective")
        if corr:
            for n, a in enumerate(corr["attempts"], 1):
                ev = a.get("evaluation") or {}
                st.markdown(
                    f"- Attempt {n} (`{a['query']}`): {', '.join(a['pages']) or 'no passages'} → "
                    f"**{ev.get('verdict', '?')}** — {ev.get('reasoning', '')}"
                    + (f" _Missing: {ev['missing_information']}_" if ev.get("missing_information") else "")
                )
            if corr["retried"]:
                st.success(f"Retried because {corr['retry_reason']} → refined query: “{corr['refined_query']}”")
            else:
                st.caption("No retry needed.")
        else:
            st.caption("Plain retrieval: " + (", ".join(f"{c['filename']}:{c['page_start']}" for c in u["final_chunks"]) or "nothing"))
    st.caption(f"Retrieval operations: {t.get('retrieval_calls', 1)} · tokens in/out: {t['tokens']['input_tokens']}/{t['tokens']['output_tokens']} · "
               + " · ".join(f"{k.replace('_ms', '')} {v:.0f} ms" for k, v in t["timings_ms"].items()))
    if t.get("fallbacks"):
        st.warning("Fallbacks used: " + "; ".join(t["fallbacks"]))


def render_assistant(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["content"])
        return
    if msg["status"] == "insufficient_information":
        st.warning(msg["content"], icon="🔎")
    elif msg["status"] == "conversational":
        st.markdown(msg["content"])
    else:
        st.markdown(msg["content"])
        if msg["citations"]:
            st.markdown("**Sources**")
            for c in msg["citations"]:
                st.caption(f"📄 **{c['filename']}**, page {c['page_label']} — “{c['passage'][:220]}{'…' if len(c['passage']) > 220 else ''}”")
    ver = msg.get("verification")
    st.caption(f"Mode: **{msg.get('mode', 'baseline')}**" + ("" if ver else " · answer not claim-verified"))
    if ver:
        with st.expander(f"Verification details ({len(ver['claims'])} claims)"):
            render_verification(ver, msg.get("original_answer"))
    if msg.get("trace"):
        with st.expander("Retrieval trace"):
            render_trace(msg["trace"])
    evidence = msg.get("evidence", [])
    if evidence:
        with st.expander(f"Retrieved evidence ({len(evidence)} passages)"):
            if msg.get("standalone") and msg["standalone"] != msg.get("question"):
                st.caption(f"Search query used (follow-up resolved): _{msg['standalone']}_")
            for e in evidence:
                badge = " · **used in answer**" if e["cited"] else ""
                st.markdown(f"**#{e['rank']} · {e['filename']} · page {e['page_label']}** · score {e['similarity']:.3f}{badge}")
                st.text(e["text"])
                st.divider()
    elif msg["status"] == "insufficient_information" and msg.get("reason") not in ("no_documents",):
        st.caption("No passages passed the relevance threshold.")


def ask(question: str) -> None:
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer…"):
            payload = {"question": question, "session_id": ss.session_id, "mode": ss.get("mode_value", "baseline")}
            if payload["mode"] == "custom":
                payload["flags"] = ss.get("custom_flags", {})
            if ss.get("scope"):
                payload["document_ids"] = ss.scope
            try:
                r = api("POST", "/chat", json=payload, timeout=360)
                msg = {
                    "role": "assistant", "content": r["answer"], "status": r["status"],
                    "reason": r.get("reason"), "citations": r["citations"], "evidence": r["evidence"],
                    "question": question, "standalone": r["standalone_question"],
                    "verification": r.get("verification"), "original_answer": r.get("original_answer"),
                    "trace": r.get("trace"), "mode": r.get("mode"),
                }
            except APIError as exc:
                msg = {"role": "assistant", "content": str(exc), "error": True, "status": "error",
                       "citations": [], "evidence": []}
        render_assistant(msg)
    ss.messages.append(msg)


def process_uploads(files) -> None:
    report, bar = [], st.sidebar.progress(0.0, text="Uploading…")
    for i, f in enumerate(files):
        bar.progress(i / len(files), text=f"Processing {f.name} ({i + 1}/{len(files)})…")
        try:
            res = api("POST", "/documents/upload", timeout=600,
                      files=[("files", (f.name, f.getvalue(), "application/pdf"))])["results"][0]
            report.append((res["status"], f"{f.name}: {res['message']}"))
        except APIError as exc:
            report.append(("failed", f"{f.name}: {exc}"))
    bar.empty()
    ss.upload_report = report
    ss.uploader_key += 1  # resets the uploader widget
    st.rerun()


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📚 Documents")
    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True,
                             key=f"uploader_{ss.uploader_key}")
    if files and st.button("Process uploads", type="primary", use_container_width=True):
        process_uploads(files)

    for status, text in ss.upload_report:
        {"indexed": st.success, "duplicate": st.info}.get(status, st.error)(text)

    docs = fetch_documents()
    if docs:
        st.subheader(f"Uploaded ({len(docs)})")
    for d in docs:
        with st.container(border=True):
            st.markdown(f"{STATUS_ICON.get(d['status'], '•')} **{d['filename']}**")
            if d["status"] == "indexed":
                st.caption(f"{d['num_pages']} pages · {d['num_chunks']} chunks")
            elif d["status"] == "failed":
                st.caption(f"Failed: {d.get('error') or 'unknown error'}")
            else:
                st.caption("Processing…")
            if st.button("Delete", key=f"del_{d['document_id']}", use_container_width=True):
                try:
                    api("DELETE", f"/documents/{d['document_id']}")
                    ss.upload_report = []
                    st.rerun()
                except APIError as exc:
                    st.error(str(exc))

    indexed = [d for d in docs if d["status"] == "indexed"]
    if len(indexed) > 1:
        st.multiselect(
            "Search only in (empty = all)", options=[d["document_id"] for d in indexed],
            format_func=lambda i: next(d["filename"] for d in indexed if d["document_id"] == i),
            key="scope",
        )
    else:
        ss.scope = []

    st.divider()
    st.subheader("Pipeline mode")
    choice = st.radio("Mode", ["Baseline RAG", "Full Agentic RAG", "Custom (ablation)"], key="mode_choice")
    ss.mode_value = {"Baseline RAG": "baseline", "Full Agentic RAG": "full", "Custom (ablation)": "custom"}[choice]
    if ss.mode_value == "custom":
        ss.custom_flags = {
            "adaptive": st.checkbox("Adaptive RAG (routing + decomposition)"),
            "corrective": st.checkbox("Corrective RAG (evaluate + one retry)"),
            "verification": st.checkbox("Claim-level verification"),
        }

    st.divider()
    st.subheader("System")
    h = fetch_health()
    if "error" in h:
        st.error(h["error"])
    else:
        c1, c2 = st.columns(2)
        c1.metric("Indexed docs", h["indexed_documents"])
        c2.metric("Chunks", h["total_chunks"])
        st.caption(f"Embedding model: `{h['embedding_model']}` ({'loaded' if h['embedding_model_loaded'] else 'not loaded yet'})")
        st.caption(f"Vector DB: {'🟢' if h['vector_db'] == 'ok' else '🔴'} {h['vector_db']}")
        st.caption(f"LLM: `{h['llm_provider']}/{h['llm_model']}` — {'configured' if h['llm_configured'] else '⚠️ API key missing'}")

# --------------------------------------------------------------------------- main
head, clear = st.columns([5, 1])
head.title("Knowledge Enterprise Agent")
head.caption("Ask questions about your uploaded PDFs. Answers cite the document and page they come from.")
if clear.button("Clear chat", use_container_width=True):
    try:
        api("DELETE", f"/chat/{ss.session_id}", timeout=10)
    except APIError:
        pass
    ss.messages, ss.session_id = [], uuid.uuid4().hex
    st.rerun()

if not ss.messages:
    st.info("Upload one or more PDFs in the sidebar, then ask a question below.")
for m in ss.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "assistant":
            render_assistant(m)
        else:
            st.markdown(m["content"])

if prompt := st.chat_input("Ask a question about your documents…"):
    ask(prompt)
