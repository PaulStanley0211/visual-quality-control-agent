"""Streamlit demo: upload a part image, watch the investigation, view the three-part output.

Run:  uv run streamlit run ui/streamlit_app.py

Runs the agent loop in-process (no separate service needed). The MES is seeded on
first launch if empty.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from agent.graph import build_graph, run_inspection
from agent.state import AgentDeps
from memory import mes
from memory import seed as seed_module

st.set_page_config(page_title="Visual Quality Control Agent", layout="wide")


@st.cache_resource
def get_graph():
    seed_module.ensure_seeded()  # idempotent; guarantees the schema exists before any MES read
    return build_graph(AgentDeps())


@st.cache_data(ttl=60)
def scenario_parts() -> list[str]:
    seed_module.ensure_seeded()
    conn = mes.connect()
    try:
        rows = conn.execute(
            "SELECT part_id FROM parts WHERE part_id LIKE 'SCN-%' ORDER BY part_id"
        ).fetchall()
    finally:
        conn.close()
    return [r["part_id"] for r in rows]


def part_context_caption(part_id: str) -> str:
    ctx = mes.get_part_context(part_id)
    m, b = ctx["machine"], ctx["batch"]
    return (
        f"Machine {m['id']} ({m['name']}) — recent defect rate {m['rate']:.0%}, last maintained {m['last_maintenance']}  ·  "
        f"Batch {b['id']} ({b['material_lot']}) — {b['rate']:.0%}  ·  Operator {ctx['operator']['id']}"
    )


st.title("🔍 Visual Quality Control Agent")
st.caption(
    "Single-station inspection on MVTec AD. Detects defects, diagnoses random vs systematic from factory "
    "history, decides pass / rework / reject, and triggers corrective workflows — escalating low-confidence cases."
)

col_in, col_out = st.columns([1, 1.4], gap="large")

with col_in:
    st.subheader("Input")
    part_id = st.selectbox(
        "Part ID (resolves machine / batch / operator from the MES)", scenario_parts(),
        help="Each part maps to a machine and batch whose history drives the random-vs-systematic call.",
    )
    if part_id:
        st.caption(part_context_caption(part_id))
    uploaded = st.file_uploader("Part image", type=["png", "jpg", "jpeg"])
    if uploaded:
        st.image(uploaded, caption="Uploaded part", width="stretch")
    run = st.button("Inspect", type="primary", disabled=not uploaded)

with col_out:
    st.subheader("Result")
    if run and uploaded:
        out = None
        with st.spinner("Running detect → gather context → investigate → decide → reason → act…"):
            suffix = os.path.splitext(uploaded.name)[1] or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            try:
                out = run_inspection(get_graph(), part_id, image_path=tmp_path)
            except FileNotFoundError:
                st.error("Perception model not ready — run `uv run python -m perception.train` then `uv run python -m eval.perception_eval`.")
            except ValueError as e:
                st.error(f"Could not process this image: {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Inspection failed: {e}")
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        if out is None:
            st.stop()

        disp = out.decision.disposition.value
        badge = {"pass": "✅", "rework": "🛠️", "reject": "⛔"}.get(disp, "•")
        esc = "  ·  ⚠️ ESCALATED to human" if out.escalated else ""
        st.markdown(f"### {badge} {disp.upper()} — confidence {out.decision.confidence:.0%}{esc}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Decision**")
            st.write(f"Disposition: `{disp}`")
            st.write(f"Confidence: {out.decision.confidence:.0%}")
            st.write(f"Escalated: {out.escalated}")
        with c2:
            st.markdown("**Diagnosis**")
            if out.diagnosis:
                d = out.diagnosis
                st.write(f"Defect: {d.defect_type}")
                st.write(f"Location: {d.location}")
                st.write(f"Pattern: `{d.fault_pattern.value}`")
                st.write(f"Confidence: {d.confidence:.0%}")
            else:
                st.write("No defect detected.")
        with c3:
            st.markdown("**Actions**")
            a = out.actions
            st.write(f"NCR: {a.ncr}" + (f" ({a.ncr_id})" if a.ncr_id else ""))
            st.write(f"CAPA: {a.capa}" + (f" ({a.capa_id})" if a.capa_id else ""))
            st.write(f"Machine flag: {a.machine_flag}")

        st.info(out.summary)

        if out.diagnosis and out.diagnosis.probable_cause:
            st.markdown(f"**Probable cause:** {out.diagnosis.probable_cause}")

        # Anomaly heatmap produced by THIS inspection (path carried on the result).
        if out.heatmap_path and Path(out.heatmap_path).exists():
            st.image(out.heatmap_path, caption="Anomaly heatmap", width="stretch")

        with st.expander("Investigation trace (audit trail)"):
            for i, line in enumerate(out.reasoning_trace, 1):
                st.write(f"{i}. {line}")
    else:
        st.caption("Pick a part, upload an image, and press **Inspect**.")
