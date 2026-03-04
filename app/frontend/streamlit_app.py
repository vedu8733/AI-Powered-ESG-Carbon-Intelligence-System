"""
ESG & Carbon Intelligence Platform — Enterprise Analytics Dashboard.

Production-level SaaS interface with gauge visualization, PDF export,
and company comparison.
"""

from __future__ import annotations

import io
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
import os

# Detect if running in HuggingFace Space
if os.getenv("SPACE_ID"):
    ANALYZE_ENDPOINT = "/api/v1/analyze-company"
else:
    API_BASE_URL = "http://localhost:8000"
    ANALYZE_ENDPOINT = f"{API_BASE_URL}/api/v1/analyze-company"

REQUIRED_COLUMNS: List[str] = [
    "electricity_kwh",
    "diesel_liters",
    "coal_kg",
    "waste_tons",
    "csr_spending",
    "employee_count",
]

COLORS = {
    "primary": "#1a365d",
    "secondary": "#2c5282",
    "success": "#276749",
    "warning": "#c05621",
    "danger": "#c53030",
    "neutral": "#4a5568",
    "bg_light": "#f7fafc",
}

# -----------------------------------------------------------------------------
# Page Config & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ESG & Carbon Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 600; color: #1a365d; margin-bottom: 0.25rem; }
    .sub-header { font-size: 1rem; color: #4a5568; margin-bottom: 1rem; }
    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #2d3748;
        margin: 0 0 1rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .pill-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .pill-low { background: #c6f6d5; color: #276749; }
    .pill-medium { background: #feebc8; color: #c05621; }
    .pill-high { background: #fed7d7; color: #c53030; }
    .block-container { padding-top: 1.5rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helper: Default Form Values
# -----------------------------------------------------------------------------
def get_default_form_values() -> Dict[str, float | int]:
    return {
        "electricity_kwh": 50000.0,
        "diesel_liters": 5000.0,
        "coal_kg": 10000.0,
        "waste_tons": 200.0,
        "csr_spending": 250000.0,
        "employee_count": 1000,
    }


# -----------------------------------------------------------------------------
# Helper: CSV Normalization & Validation
# -----------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming CSV column names to a consistent machine-friendly schema.

    - Strip whitespace
    - Lowercase
    - Replace spaces with underscores
    - Remove special characters (keep a–z, 0–9, and underscore)
    """
    normalized_df = df.copy()
    normalized_columns: List[str] = []
    for col in normalized_df.columns:
        normalized = col.strip().lower()
        normalized = normalized.replace(" ", "_")
        normalized = re.sub(r"[^a-z0-9_]", "", normalized)
        normalized_columns.append(normalized)
    normalized_df.columns = normalized_columns
    return normalized_df


def validate_csv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that the normalized DataFrame contains all required ESG fields.

    Returns:
        (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return (len(missing) == 0, missing)


# -----------------------------------------------------------------------------
# Helper: PDF Generation
# -----------------------------------------------------------------------------
def generate_pdf(report_text: str) -> bytes:
    """
    Convert markdown report text to PDF bytes using ReportLab.

    Args:
        report_text: Markdown-formatted report string.

    Returns:
        PDF file as bytes.
    """
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    header1 = ParagraphStyle(
        "CustomH1",
        parent=styles["Heading1"],
        fontSize=16,
        spaceAfter=12,
    )
    header2 = ParagraphStyle(
        "CustomH2",
        parent=styles["Heading2"],
        fontSize=14,
        spaceAfter=10,
    )
    header3 = ParagraphStyle(
        "CustomH3",
        parent=styles["Heading3"],
        fontSize=12,
        spaceAfter=8,
    )
    normal = ParagraphStyle(
        "CustomNormal",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
    )

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    story: List[Any] = []

    for line in report_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        text = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if line.startswith("# "):
            story.append(Paragraph(text[2:], header1))
        elif line.startswith("## "):
            story.append(Paragraph(text[3:], header2))
        elif line.startswith("### "):
            story.append(Paragraph(text[4:], header3))
        elif line.startswith("- ") or line.startswith("* "):
            story.append(Paragraph(f"• {text[2:]}", normal))
        else:
            story.append(Paragraph(text, normal))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# -----------------------------------------------------------------------------
# Helper: ESG Gauge Chart
# -----------------------------------------------------------------------------
def render_esg_gauge(score: float) -> go.Figure:
    """Create ESG score gauge with green/amber/red zones."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " ", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#c53030"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#e2e8f0",
                "steps": [
                    {"range": [0, 50], "color": "#fed7d7"},
                    {"range": [50, 75], "color": "#feebc8"},
                    {"range": [75, 100], "color": "#c6f6d5"},
                ],
                "threshold": {
                    "line": {"color": "#1a365d", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            },
            title={"text": "ESG Score", "font": {"size": 14}},
        )
    )
    fig.update_layout(
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        height=220,
    )
    return fig


# -----------------------------------------------------------------------------
# Helper: Risk Badge HTML
# -----------------------------------------------------------------------------
def render_risk_badge(risk_level: str) -> str:
    """Return HTML for pill-style risk badge."""
    level = risk_level.upper()
    if risk_level == "Low":
        cls = "pill-low"
    elif risk_level == "Medium":
        cls = "pill-medium"
    else:
        cls = "pill-high"
    return f'<span class="pill-badge {cls}">{level}</span>'


# -----------------------------------------------------------------------------
# Helper: API Client
# -----------------------------------------------------------------------------
def call_analyze_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Call FastAPI analyze-company endpoint."""
    response = requests.post(
        ANALYZE_ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


# -----------------------------------------------------------------------------
# Session State
# -----------------------------------------------------------------------------
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "form_values" not in st.session_state:
    st.session_state.form_values = get_default_form_values()
if "form_values_2" not in st.session_state:
    st.session_state.form_values_2 = get_default_form_values()
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None
if "csv_df" not in st.session_state:
    st.session_state.csv_df = None
if "csv_selected_index" not in st.session_state:
    st.session_state.csv_selected_index = 0


# -----------------------------------------------------------------------------
# Top Controls: Toggles
# -----------------------------------------------------------------------------
col_toggle1, _, _ = st.columns([1, 1, 2])
with col_toggle1:
    enable_comparison = st.checkbox("Enable Company Comparison", value=False)

st.divider()



# -----------------------------------------------------------------------------
# Sidebar — CSV Upload + Company Input Form(s)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Upload ESG Dataset (CSV)")
    sample_df = pd.DataFrame(
        [
            {
                "electricity_kwh": 50000,
                "diesel_liters": 5000,
                "coal_kg": 10000,
                "waste_tons": 200,
                "csr_spending": 250000,
                "employee_count": 1000,
            }
        ]
    )
    sample_csv_bytes = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Sample ESG Template",
        data=sample_csv_bytes,
        file_name="esg_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

    uploaded_file = st.file_uploader("Upload ESG CSV File", type=["csv"], key="esg_csv")

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            df_normalized = normalize_columns(raw_df)
            is_valid, missing_cols = validate_csv(df_normalized)
            if not is_valid:
                st.session_state.csv_df = None
                missing_list = "\n".join(f"- `{col}`" for col in missing_cols)
                st.error(
                    "Uploaded CSV is missing required columns:\n"
                    f"{missing_list}"
                )
            else:
                st.session_state.csv_df = df_normalized
                st.session_state.csv_selected_index = 0
                st.success("CSV dataset loaded successfully.")
                st.caption("Preview of the first 5 rows from the uploaded dataset:")
                st.dataframe(df_normalized.head(), use_container_width=True)
        except Exception as exc:  # pragma: no cover - defensive
            st.session_state.csv_df = None
            st.error(f"Failed to read CSV file: {exc}")

    df_state = st.session_state.csv_df
    if isinstance(df_state, pd.DataFrame) and not df_state.empty:
        row_indices = list(range(len(df_state)))
        selected_index = st.selectbox(
            "Select Company Row",
            options=row_indices,
            index=min(st.session_state.csv_selected_index, len(df_state) - 1),
            format_func=lambda i: f"Row {i + 1}",
        )
        st.session_state.csv_selected_index = int(selected_index)

        row = df_state.iloc[selected_index]
        # Auto-fill manual form defaults from selected row
        st.session_state.form_values = {
            "electricity_kwh": float(row["electricity_kwh"]),
            "diesel_liters": float(row["diesel_liters"]),
            "coal_kg": float(row["coal_kg"]),
            "waste_tons": float(row["waste_tons"]),
            "csr_spending": float(row["csr_spending"]),
            "employee_count": int(row["employee_count"]),
        }

    st.markdown("---")
    st.markdown("### Company Input")
    st.markdown("Enter operational metrics for ESG analysis.")
    st.divider()

    fv = st.session_state.form_values
    electricity_kwh = st.number_input(
        "Electricity (kWh)",
        min_value=0.0,
        value=float(fv["electricity_kwh"]),
        step=1000.0,
        format="%.0f",
        key="elec",
    )
    diesel_liters = st.number_input(
        "Diesel (liters)",
        min_value=0.0,
        value=float(fv["diesel_liters"]),
        step=100.0,
        format="%.0f",
        key="diesel",
    )
    coal_kg = st.number_input(
        "Coal (kg)",
        min_value=0.0,
        value=float(fv["coal_kg"]),
        step=100.0,
        format="%.0f",
        key="coal",
    )
    waste_tons = st.number_input(
        "Waste (tons)",
        min_value=0.0,
        value=float(fv["waste_tons"]),
        step=10.0,
        format="%.1f",
        key="waste",
    )
    csr_spending = st.number_input(
        "CSR Spending",
        min_value=0.0,
        value=float(fv["csr_spending"]),
        step=10000.0,
        format="%.0f",
        key="csr",
    )
    employee_count = st.number_input(
        "Employee Count",
        min_value=1,
        value=int(fv["employee_count"]),
        step=10,
        key="emp",
    )

    payload_1 = {
        "electricity_kwh": electricity_kwh,
        "diesel_liters": diesel_liters,
        "coal_kg": coal_kg,
        "waste_tons": waste_tons,
        "csr_spending": csr_spending,
        "employee_count": int(employee_count),
    }

    if enable_comparison:
        st.markdown("---")
        st.markdown("**Company 2**")
        fv2 = st.session_state.form_values_2
        electricity_kwh_2 = st.number_input(
            "Electricity (kWh)",
            min_value=0.0,
            value=float(fv2["electricity_kwh"]),
            step=1000.0,
            format="%.0f",
            key="elec2",
        )
        diesel_liters_2 = st.number_input(
            "Diesel (liters)",
            min_value=0.0,
            value=float(fv2["diesel_liters"]),
            step=100.0,
            format="%.0f",
            key="diesel2",
        )
        coal_kg_2 = st.number_input(
            "Coal (kg)",
            min_value=0.0,
            value=float(fv2["coal_kg"]),
            step=100.0,
            format="%.0f",
            key="coal2",
        )
        waste_tons_2 = st.number_input(
            "Waste (tons)",
            min_value=0.0,
            value=float(fv2["waste_tons"]),
            step=10.0,
            format="%.1f",
            key="waste2",
        )
        csr_spending_2 = st.number_input(
            "CSR Spending",
            min_value=0.0,
            value=float(fv2["csr_spending"]),
            step=10000.0,
            format="%.0f",
            key="csr2",
        )
        employee_count_2 = st.number_input(
            "Employee Count",
            min_value=1,
            value=int(fv2["employee_count"]),
            step=10,
            key="emp2",
        )
        payload_2 = {
            "electricity_kwh": electricity_kwh_2,
            "diesel_liters": diesel_liters_2,
            "coal_kg": coal_kg_2,
            "waste_tons": waste_tons_2,
            "csr_spending": csr_spending_2,
            "employee_count": int(employee_count_2),
        }

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    with col_btn2:
        reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        st.session_state.analysis_result = None
        st.session_state.comparison_result = None
        st.session_state.form_values = get_default_form_values()
        st.session_state.form_values_2 = get_default_form_values()
        st.session_state.csv_df = None
        st.session_state.csv_selected_index = 0
        st.rerun()


# -----------------------------------------------------------------------------
# Page Header
# -----------------------------------------------------------------------------
st.markdown('<p class="main-header">ESG & Carbon Intelligence Platform</p>', unsafe_allow_html=True)
mode_badge = (
    '<span class="pill-badge pill-medium">Mode: CSV Upload</span>'
    if isinstance(st.session_state.csv_df, pd.DataFrame)
    else '<span class="pill-badge pill-low">Mode: Manual Entry</span>'
)
st.markdown(
    f'<p class="sub-header">AI-Powered Sustainability & Compliance Analytics&nbsp;&nbsp;{mode_badge}</p>',
    unsafe_allow_html=True,
)
if isinstance(st.session_state.csv_df, pd.DataFrame) and not st.session_state.csv_df.empty:
    st.info("CSV Dataset Mode Active – Using uploaded dataset")
st.divider()


# -----------------------------------------------------------------------------
# Analyze Action
# -----------------------------------------------------------------------------
if analyze_clicked:
    # Decide data source for primary company: CSV row or manual form.
    df_state = st.session_state.csv_df
    if isinstance(df_state, pd.DataFrame) and not df_state.empty:
        idx = min(st.session_state.csv_selected_index, len(df_state) - 1)
        row = df_state.iloc[idx]
        payload_1 = {
            "electricity_kwh": float(row["electricity_kwh"]),
            "diesel_liters": float(row["diesel_liters"]),
            "coal_kg": float(row["coal_kg"]),
            "waste_tons": float(row["waste_tons"]),
            "csr_spending": float(row["csr_spending"]),
            "employee_count": int(row["employee_count"]),
        }
    else:
        st.session_state.form_values = payload_1

    if enable_comparison:
        st.session_state.form_values_2 = payload_2

    with st.spinner("Running ESG analysis… This may take a minute."):
        try:
            result_1 = call_analyze_api(payload_1)
            st.session_state.analysis_result = result_1

            if enable_comparison:
                result_2 = call_analyze_api(payload_2)
                st.session_state.comparison_result = result_2

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the backend. Ensure the FastAPI server is running at "
                f"`{API_BASE_URL}` and try again."
            )
        except requests.exceptions.Timeout:
            st.error("Request timed out. The analysis is taking longer than expected.")
        except requests.exceptions.HTTPError as e:
            detail = "Unknown error"
            if e.response is not None:
                try:
                    err_body = e.response.json()
                    detail = err_body.get("detail", str(e))
                except Exception:
                    detail = e.response.text or str(e)
            st.error(f"Backend error: {detail}")
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

    if st.session_state.analysis_result is not None:
        st.rerun()


# -----------------------------------------------------------------------------
# Main Dashboard — Render Results
# -----------------------------------------------------------------------------
result = st.session_state.analysis_result
result_2 = st.session_state.comparison_result if enable_comparison else None

if result is None:
    st.info(
        "Enter company metrics in the sidebar and click **Analyze** to run "
        "the ESG & Carbon Intelligence analysis."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Comparison Mode
# -----------------------------------------------------------------------------
if enable_comparison and result_2 is not None:
    with st.container():
        st.markdown('<p class="section-title">Company Comparison</p>', unsafe_allow_html=True)

        c1 = result.get("carbon", {})
        e1 = result.get("esg", {})
        c2 = result_2.get("carbon", {})
        e2 = result_2.get("esg", {})

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Company 1**")
            st.metric("ESG Score", f"{e1.get('final_esg_score', 0):.1f}", "")
            st.metric("Total Emissions", f"{c1.get('total_emissions_tons', 0):,.1f} t CO₂e", "")
            st.markdown(render_risk_badge(e1.get("risk_level", "—")), unsafe_allow_html=True)
        with col_b:
            st.markdown("**Company 2**")
            st.metric("ESG Score", f"{e2.get('final_esg_score', 0):.1f}", "")
            st.metric("Total Emissions", f"{c2.get('total_emissions_tons', 0):,.1f} t CO₂e", "")
            st.markdown(render_risk_badge(e2.get("risk_level", "—")), unsafe_allow_html=True)

        b1 = c1.get("breakdown_tons", {})
        b2 = c2.get("breakdown_tons", {})
        if b1 and b2:
            sources = list(b1.keys())
            labels = [s.replace("_", " ").title() for s in sources]
            fig = go.Figure(
                data=[
                    go.Bar(name="Company 1", x=labels, y=[b1[s] for s in sources], marker_color="#2c5282"),
                    go.Bar(name="Company 2", x=labels, y=[b2[s] for s in sources], marker_color="#4299e1"),
                ],
                layout=go.Layout(
                    barmode="group",
                    xaxis_title="Source",
                    yaxis_title="Emissions (tons CO₂e)",
                    margin=dict(t=20, b=40, l=60, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=320,
                ),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.divider()


# -----------------------------------------------------------------------------
# Section 1: KPI Summary (Single Company)
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<p class="section-title">KPI Summary</p>', unsafe_allow_html=True)

    carbon = result.get("carbon", {})
    esg = result.get("esg", {})
    total_emissions = carbon.get("total_emissions_tons", 0)
    final_esg_score = esg.get("final_esg_score", 0)
    risk_level = esg.get("risk_level", "—")

    col1, col2, col3 = st.columns(3)
    with col1:
        fig = render_esg_gauge(final_esg_score)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with col2:
        st.markdown(
            f'<div style="text-align:center; padding:1.5rem;">'
            f'<div style="font-size:0.9rem; color:#718096; margin-bottom:0.5rem;">Risk Level</div>'
            f'{render_risk_badge(risk_level)}</div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.metric("Total Emissions", f"{total_emissions:,.1f} t CO₂e", "metric tons")

st.divider()


# -----------------------------------------------------------------------------
# Section 2: Emission Breakdown
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<p class="section-title">Emission Breakdown</p>', unsafe_allow_html=True)

    breakdown = carbon.get("breakdown_tons", {})
    if breakdown:
        sources = list(breakdown.keys())
        values = [breakdown[s] for s in sources]
        labels = [s.replace("_", " ").title() for s in sources]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=["#2c5282", "#3182ce", "#4299e1", "#63b3ed"],
                    text=[f"{v:.2f}" for v in values],
                    textposition="outside",
                )
            ],
            layout=go.Layout(
                xaxis_title="Source",
                yaxis_title="Emissions (tons CO₂e)",
                margin=dict(t=20, b=40, l=60, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(size=12),
                height=320,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("No emission breakdown available.")

st.divider()


# -----------------------------------------------------------------------------
# Section 3: Anomaly Detection
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<p class="section-title">Anomaly Detection</p>', unsafe_allow_html=True)

    anomaly = result.get("anomaly", {})
    status = anomaly.get("status", "—")
    anomalies_list = anomaly.get("anomalies_detected", [])

    if status == "Normal":
        st.success("Data quality check passed. No anomalies detected.")
    else:
        st.error("Suspicious data detected. Review the following before relying on results:")
        for item in anomalies_list:
            st.markdown(f"- {item}")

st.divider()


# -----------------------------------------------------------------------------
# Section 4: Compliance Analysis
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<p class="section-title">Compliance Analysis</p>', unsafe_allow_html=True)

    compliance_text = result.get("compliance", "")
    if compliance_text:
        with st.expander("View compliance assessment", expanded=False):
            st.markdown(compliance_text)
    else:
        st.caption("No compliance analysis available.")

st.divider()


# -----------------------------------------------------------------------------
# Section 5: Executive ESG Report + PDF Download
# -----------------------------------------------------------------------------
with st.container():
    st.markdown('<p class="section-title">Executive ESG Report</p>', unsafe_allow_html=True)

    report_text = result.get("report", "")
    if report_text:
        st.markdown(report_text)
        st.divider()

        pdf_bytes = generate_pdf(report_text)
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name="esg_report.pdf",
            mime="application/pdf",
            type="secondary",
        )
    else:
        st.caption("No report generated.")
