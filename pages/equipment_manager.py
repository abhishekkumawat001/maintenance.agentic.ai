"""
🔧 Equipment Manager — Add, view, and manage industrial equipment.
"""

import asyncio
import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.llm_provider import LLMProvider
from app.knowledge_base import MaintenanceKnowledgeBase

st.set_page_config(page_title="Equipment Manager", page_icon="🔧", layout="wide")


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@st.cache_resource
def init_components():
    llm = LLMProvider()
    kb = MaintenanceKnowledgeBase()
    return {'llm': llm, 'kb': kb}


components = init_components()

st.markdown("""
<style>
    .gradient-text {
        background: linear-gradient(90deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2rem;
    }
    .equipment-card {
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 10px;
        padding: 20px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="gradient-text">🔧 Equipment Manager</p>', unsafe_allow_html=True)
st.caption("Register and manage your industrial equipment inventory")

# ─── Add Equipment ───
st.subheader("➕ Add New Equipment")

with st.form("add_equipment_form", clear_on_submit=True):
    col1, col2 = st.columns(2)
    with col1:
        eq_id = st.text_input("Equipment ID *", placeholder="e.g., PUMP_001")
        eq_name = st.text_input("Equipment Name *", placeholder="e.g., Main Cooling Pump")
        eq_type = st.selectbox("Equipment Type *", [
            "pump", "motor", "compressor", "turbine", "fan",
            "heat_exchanger", "valve", "generator", "transformer",
            "chiller", "boiler", "conveyor", "other"
        ])
    with col2:
        eq_model = st.text_input("Model", placeholder="e.g., XR-5000")
        eq_manufacturer = st.text_input("Manufacturer", placeholder="e.g., Grundfos")
        eq_install_date = st.date_input("Installation Date")

    submitted = st.form_submit_button("Add Equipment", use_container_width=True, type="primary")

    if submitted:
        if not eq_id or not eq_name:
            st.error("Equipment ID and Name are required!")
        else:
            success = components['kb'].add_equipment({
                'id': eq_id.upper(),
                'name': eq_name,
                'type': eq_type,
                'model': eq_model or 'Unknown',
                'manufacturer': eq_manufacturer or 'Unknown',
                'installation_date': eq_install_date.strftime('%Y-%m-%d')
            })
            if success:
                st.success(f"✅ Equipment **{eq_id.upper()}** added successfully!")

                # Get AI insights about this equipment type
                with st.spinner("Getting AI insights..."):
                    advice = run_async(components['llm'].generate(
                        f"Provide brief maintenance recommendations for a {eq_type} "
                        f"({eq_name}): typical maintenance intervals, key parameters "
                        f"to monitor, common failure modes, safety considerations. "
                        f"Keep it concise (5-6 bullet points)."
                    ))
                    st.info(f"🤖 **AI Insights for {eq_type}:**\n\n{advice}")

                st.rerun()
            else:
                st.error("Failed to add equipment!")

st.divider()

# ─── Equipment List ───
st.subheader("📋 Equipment Inventory")

equipment_list = components['kb'].get_all_equipment()

if equipment_list:
    st.caption(f"Total: {len(equipment_list)} equipment registered")

    for eq in equipment_list:
        with st.container():
            st.markdown(
                f'<div class="equipment-card">'
                f'<strong style="color: #6C63FF; font-size: 1.1rem">{eq["id"]}</strong> — {eq["name"]}<br>'
                f'<span style="color: #4ECDC4">Type:</span> {eq["type"]} · '
                f'<span style="color: #4ECDC4">Model:</span> {eq.get("model", "N/A")} · '
                f'<span style="color: #4ECDC4">Manufacturer:</span> {eq.get("manufacturer", "N/A")}<br>'
                f'<span style="color: #888">Installed: {eq.get("installation_date", "N/A")} · '
                f'Last Maintenance: {eq.get("last_maintenance", "Never")}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        # Show maintenance history
        history = components['kb'].get_maintenance_history(eq['id'])
        if history:
            with st.expander(f"📜 Maintenance History ({len(history)} records)"):
                for record in history[:5]:
                    st.markdown(
                        f"- **{record['maintenance_date']}**: {record['description']} "
                        f"({record['type']}) — {record.get('technician', 'N/A')}"
                    )
else:
    st.info("No equipment registered yet. Use the form above to add your first equipment.")
    st.markdown("""
    **Quick Start — Add these sample equipment:**
    - `PUMP_001` — Main Cooling Pump (pump)
    - `MOTOR_001` — Drive Motor Unit A (motor)
    - `COMP_001` — Air Compressor (compressor)
    """)
