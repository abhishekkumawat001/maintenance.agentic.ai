"""
📊 Sensor Analysis — Streamlit page for equipment sensor data input and AI analysis.
"""

import asyncio
import os
import sys
import streamlit as st
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig
from app.llm_provider import LLMProvider
from app.knowledge_base import MaintenanceKnowledgeBase
from app.vector_store import VectorStore
from app.sensor_processor import SensorDataProcessor
from app.diagnostic_engine import DiagnosticEngine
from app.models import SensorData

st.set_page_config(page_title="Sensor Analysis", page_icon="📊", layout="wide")


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
    vs = VectorStore()
    sensor_proc = SensorDataProcessor(llm)
    diag_engine = DiagnosticEngine(kb, llm)
    return {'llm': llm, 'kb': kb, 'vs': vs, 'sensor_proc': sensor_proc, 'diag_engine': diag_engine}


components = init_components()

# ─── Page Header ───
st.markdown("""
<style>
    .gradient-text {
        background: linear-gradient(90deg, #6C63FF, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2rem;
    }
    .metric-card {
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .status-normal { color: #4ECDC4; font-weight: bold; font-size: 1.2rem; }
    .status-warning { color: #FFE66D; font-weight: bold; font-size: 1.2rem; }
    .status-critical { color: #FF6B6B; font-weight: bold; font-size: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="gradient-text">📊 Sensor Data Analysis</p>', unsafe_allow_html=True)
st.caption("Input equipment sensor readings for AI-powered anomaly detection and fault diagnosis")

# ─── Equipment Selection ───
equipment_list = components['kb'].get_all_equipment()

if not equipment_list:
    st.warning("⚠️ No equipment registered. Go to **Equipment Manager** to add equipment first.")
    st.stop()

equipment_options = {f"{eq['id']} — {eq['name']}": eq['id'] for eq in equipment_list}
selected = st.selectbox("Select Equipment", list(equipment_options.keys()))
equipment_id = equipment_options[selected]

st.divider()

# ─── Sensor Input Form ───
st.subheader("📡 Enter Sensor Readings")

col1, col2, col3 = st.columns(3)

with col1:
    temperature = st.number_input(
        "🌡️ Temperature (°C)", min_value=-50.0, max_value=500.0,
        value=45.0, step=0.5, help="Normal range: 0-80°C"
    )
    vibration = st.number_input(
        "📳 Vibration (mm/s)", min_value=0.0, max_value=100.0,
        value=3.5, step=0.1, help="Normal range: 0-10 mm/s"
    )

with col2:
    pressure = st.number_input(
        "🔵 Pressure (bar)", min_value=0.0, max_value=500.0,
        value=55.0, step=0.5, help="Normal range: 0-100 bar"
    )
    humidity = st.number_input(
        "💧 Humidity (%)", min_value=0.0, max_value=100.0,
        value=45.0, step=0.5, help="Normal range: 0-100%"
    )

with col3:
    sound_level = st.number_input(
        "🔊 Sound Level (dB)", min_value=0.0, max_value=200.0,
        value=65.0, step=1.0, help="Typical industrial: 60-90 dB"
    )

st.divider()

# ─── Analysis ───
if st.button("🔍 Analyze Sensor Data", use_container_width=True, type="primary"):
    sensor_data = SensorData(
        sensor_id=equipment_id,
        timestamp=datetime.now(),
        temperature=temperature,
        vibration=vibration,
        pressure=pressure,
        humidity=humidity,
        sound_level=sound_level
    )

    # Basic analysis
    processed = components['sensor_proc'].process_sensor_data(sensor_data)

    # Display status
    status = processed['status']
    status_class = f"status-{status}" if status in ['normal', 'warning', 'critical'] else 'status-normal'

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><p class="{status_class}">{status.upper()}</p>'
            f'<p style="color:#888">Overall Status</p></div>',
            unsafe_allow_html=True
        )
    with col2:
        st.metric("Anomalies", len(processed['anomalies']))
    with col3:
        st.metric("Equipment", equipment_id)
    with col4:
        st.metric("Timestamp", datetime.now().strftime("%H:%M:%S"))

    # Show anomalies
    if processed['anomalies']:
        st.subheader("⚠️ Anomalies Detected")
        for anomaly in processed['anomalies']:
            severity_color = "🔴" if anomaly['severity'] == 'high' else "🟡"
            st.warning(
                f"{severity_color} **{anomaly['parameter'].title()}**: "
                f"{anomaly['value']} (threshold: {anomaly['threshold']['min']}-{anomaly['threshold']['max']}) "
                f"— Severity: **{anomaly['severity'].upper()}**"
            )

    # LLM Analysis
    st.subheader("🤖 AI Analysis")
    with st.spinner("Gemini is analyzing sensor data..."):
        llm_analysis = run_async(components['sensor_proc'].analyze_with_llm(sensor_data))

    if isinstance(llm_analysis, dict):
        risk_score = llm_analysis.get('risk_score', 'N/A')
        st.metric("Risk Score", f"{risk_score}/10")

        if llm_analysis.get('recommendations'):
            st.markdown("**Recommendations:**")
            for rec in llm_analysis['recommendations']:
                st.markdown(f"- {rec}")

        if llm_analysis.get('failure_modes'):
            st.markdown("**Potential Failure Modes:**")
            for mode in llm_analysis['failure_modes']:
                st.markdown(f"- {mode}")
    else:
        st.markdown(str(llm_analysis))

    # Fault diagnosis if anomalies detected
    if processed['status'] in ['warning', 'critical']:
        st.subheader("🔧 Fault Diagnosis")
        with st.spinner("Running root cause analysis..."):
            diagnosis = run_async(
                components['diag_engine'].diagnose_fault(processed, {}, equipment_id)
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Severity", diagnosis.severity.upper())
        with col2:
            st.metric("Confidence", f"{diagnosis.confidence:.0%}")
        with col3:
            st.metric("Est. Downtime", f"{diagnosis.estimated_downtime}h")

        st.markdown(f"**Root Cause:** {diagnosis.root_cause}")
        st.markdown("**Recommended Actions:**")
        for action in diagnosis.recommended_actions:
            st.markdown(f"- ✅ {action}")
