import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


SIMULATION_DATA_FILE = (
    config.RUNTIME_DIR
    / "operational_log.csv"
)

REAL_MOTOR_DATA_FILE = (
    config.RUNTIME_DIR
    / "real_motor_operational_log.csv"
)

REVIEW_LOG_FILE = (
    config.RUNTIME_DIR
    / "model_review_log.csv"
)

REQUIRED_CONSECUTIVE_ABNORMAL = 3

LIVE_REFRESH_SECONDS = 2

MODE_OPTIONS = {
    "Simulation Mode": "simulation",
    "Real Motor Mode": "real_motor",
}


st.set_page_config(
    page_title="Motor Intelligence Center",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root {
    --bg: #07111F;
    --panel: #0D1B2A;
    --panel-2: #102338;
    --border: #1B3850;
    --text: #E8F0F7;
    --muted: #8EA6B8;
    --cyan: #21D4FD;
    --green: #19D3AE;
    --yellow: #F5C451;
    --red: #FF5E6C;
}

.stApp {
    background:
        radial-gradient(
            circle at 82% 2%,
            rgba(33, 212, 253, 0.08),
            transparent 28%
        ),
        linear-gradient(
            135deg,
            #07111F 0%,
            #091625 55%,
            #07111F 100%
        );
}

[data-testid="stSidebar"] {
    background: #06101C;
    border-right: 1px solid #183044;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.2rem;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    letter-spacing: 0.02em;
}

h1 {
    font-size: 2rem !important;
}

h2 {
    color: #E8F0F7;
    margin-top: 1.2rem;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(16, 35, 56, 0.96),
        rgba(10, 25, 41, 0.96)
    );
    border: 1px solid #1B3850;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    min-height: 115px;
    box-shadow: 0 8px 26px rgba(0, 0, 0, 0.18);
}

[data-testid="stMetricLabel"] {
    color: #8EA6B8;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

[data-testid="stMetricValue"] {
    color: #E8F0F7;
    font-size: 1.7rem;
}

[data-testid="stDataFrame"] {
    border: 1px solid #1B3850;
    border-radius: 12px;
    overflow: hidden;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.mode-pill {
    display: inline-block;
    border: 1px solid #19D3AE;
    color: #19D3AE;
    border-radius: 999px;
    padding: 0.28rem 0.7rem;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.machine-header {
    border: 1px solid #1B3850;
    border-radius: 16px;
    background: linear-gradient(
        120deg,
        rgba(13, 27, 42, 0.97),
        rgba(16, 35, 56, 0.78)
    );
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.9rem;
}

.machine-subtitle {
    color: #8EA6B8;
    font-size: 0.82rem;
    margin-top: -0.5rem;
}

.section-label {
    color: #21D4FD;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 1.1rem 0 0.55rem 0;
}

.alarm-strip {
    border: 1px solid rgba(245, 196, 81, 0.45);
    border-left: 5px solid #F5C451;
    border-radius: 10px;
    background: rgba(245, 196, 81, 0.08);
    padding: 0.7rem 1rem;
    color: #F5C451;
    margin: 0.7rem 0 1rem 0;
}

.alarm-strip-normal {
    border-color: rgba(25, 211, 174, 0.35);
    border-left-color: #19D3AE;
    background: rgba(25, 211, 174, 0.07);
    color: #19D3AE;
}

.sidebar-brand {
    border-bottom: 1px solid #183044;
    padding-bottom: 1rem;
    margin-bottom: 1rem;
}

.sidebar-brand-title {
    color: #E8F0F7;
    font-size: 1.05rem;
    font-weight: 800;
}

.sidebar-brand-subtitle {
    color: #8EA6B8;
    font-size: 0.72rem;
    margin-top: 0.3rem;
}

.small-muted {
    color: #8EA6B8;
    font-size: 0.8rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def get_data_file(run_mode):
    if run_mode == "simulation":
        return SIMULATION_DATA_FILE

    return REAL_MOTOR_DATA_FILE


def show_sidebar():
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                ⚙ MOTOR INTELLIGENCE
            </div>
            <div class="sidebar-brand-subtitle">
                DC SERIES MOTOR · PREDICTIVE MAINTENANCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="section-label">System mode</div>',
        unsafe_allow_html=True,
    )

    selected_mode_name = st.sidebar.radio(
        "Choose operating environment",
        options=list(MODE_OPTIONS.keys()),
        index=0,
    )

    run_mode = MODE_OPTIONS[
        selected_mode_name
    ]

    st.sidebar.divider()

    st.sidebar.markdown(
        '<div class="section-label">Navigation</div>',
        unsafe_allow_html=True,
    )

    selected_page = st.sidebar.radio(
        "Dashboard section",
        options=[
            "Overview",
            "Live Monitoring",
            "Trends",
            "Model Governance",
            "System Health",
        ],
        index=0,
    )

    st.sidebar.divider()

    if run_mode == "simulation":
        st.sidebar.success(
            "● SIMULATION STREAM ACTIVE"
        )
    else:
        st.sidebar.info(
            "● REAL-MOTOR STREAM ACTIVE"
        )

    st.sidebar.caption(
        "Data source: "
        + str(get_data_file(run_mode))
    )

    return run_mode, selected_page


def get_warning_status(
    data,
    predicted_fault,
):
    abnormal = (
        data["Predicted_Fault"] != 0
    ).astype(int)

    consecutive_abnormal = 0

    for value in reversed(
        abnormal.tolist()
    ):
        if value == 1:
            consecutive_abnormal += 1
        else:
            break

    if (
        consecutive_abnormal
        >= REQUIRED_CONSECUTIVE_ABNORMAL
    ):
        return (
            "EARLY WARNING",
            consecutive_abnormal,
        )

    if predicted_fault != 0:
        return (
            "POSSIBLE ABNORMALITY",
            consecutive_abnormal,
        )

    return (
        "NORMAL",
        consecutive_abnormal,
    )


def show_header(run_mode):
    mode_name = (
        "SIMULATION MODE"
        if run_mode == "simulation"
        else "REAL MOTOR MODE"
    )

    current_time = datetime.now().strftime(
        "%d %b %Y · %H:%M:%S"
    )

    st.markdown(
        f"""
        <div class="machine-header">
            <div style="display:flex;
                        justify-content:space-between;
                        align-items:center;
                        gap:1rem;">
                <div>
                    <h1 style="margin:0;">
                        DC Series Motor
                    </h1>
                    <div class="machine-subtitle">
                        Motor M01 · Intelligent condition
                        monitoring and predictive maintenance
                    </div>
                </div>
                <div style="text-align:right;">
                    <span class="mode-pill">
                        {mode_name}
                    </span>
                    <div class="small-muted"
                         style="margin-top:0.45rem;">
                        ● SYSTEM ONLINE · {current_time}
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_alarm_strip(
    warning_status,
    predicted_fault,
    confidence,
):
    if warning_status == "EARLY WARNING":
        message = (
            f"EARLY WARNING · Fault {predicted_fault} "
            f"detected across consecutive readings · "
            f"confidence {confidence * 100:.2f}%"
        )
        css_class = "alarm-strip"
    elif warning_status == "POSSIBLE ABNORMALITY":
        message = (
            f"POSSIBLE ABNORMALITY · Fault "
            f"{predicted_fault} predicted · "
            f"confidence {confidence * 100:.2f}%"
        )
        css_class = "alarm-strip"
    else:
        message = (
            "SYSTEM NORMAL · No abnormal prediction "
            "is currently active"
        )
        css_class = (
            "alarm-strip alarm-strip-normal"
        )

    st.markdown(
        f'<div class="{css_class}">⚠ {message}</div>',
        unsafe_allow_html=True,
    )


def show_governance():
    st.markdown(
        '<div class="section-label">Model governance</div>',
        unsafe_allow_html=True,
    )

    if not REVIEW_LOG_FILE.exists():
        st.info(
            "No candidate review record is available. "
            "The production model remains active."
        )
        return

    review_data = pd.read_csv(
        REVIEW_LOG_FILE
    )

    if review_data.empty:
        st.info(
            "The review log is empty. The production "
            "model remains active."
        )
        return

    latest = review_data.iloc[-1]

    reviewer_decision = str(
        latest.get(
            "Reviewer_Decision",
            "PENDING",
        )
    )

    system_decision = str(
        latest.get(
            "System_Decision",
            "UNKNOWN",
        )
    )

    reviewer_name = str(
        latest.get(
            "Reviewer_Name",
            "",
        )
    )

    production_accuracy = float(
        latest.get(
            "Production_Accuracy",
            0,
        )
    )

    candidate_accuracy = float(
        latest.get(
            "Candidate_Accuracy",
            0,
        )
    )

    accuracy_change = float(
        latest.get(
            "Accuracy_Change_Percentage_Points",
            0,
        )
    )

    candidate_fault_4_recall = float(
        latest.get(
            "Candidate_Fault_4_Recall",
            0,
        )
    )

    fault_4_change = float(
        latest.get(
            "Fault_4_Recall_Change_Percentage_Points",
            0,
        )
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Active model",
        "Production",
    )

    col2.metric(
        "System decision",
        system_decision,
    )

    col3.metric(
        "Reviewer decision",
        reviewer_decision,
    )

    col4.metric(
        "Reviewer",
        reviewer_name or "Pending",
    )

    col5, col6, col7 = st.columns(3)

    col5.metric(
        "Production accuracy",
        f"{production_accuracy:.2f}%",
    )

    col6.metric(
        "Candidate accuracy",
        f"{candidate_accuracy:.2f}%",
        delta=f"{accuracy_change:+.2f} pp",
    )

    col7.metric(
        "Candidate Fault 4 recall",
        f"{candidate_fault_4_recall:.2f}%",
        delta=f"{fault_4_change:+.2f} pp",
    )

    notes = str(
        latest.get(
            "Review_Notes",
            "",
        )
    )

    if reviewer_decision == "REQUEST_MORE_DATA":
        st.warning(
            "Candidate not promoted · More independently "
            "verified data is required."
        )
    elif reviewer_decision == "APPROVE":
        st.success(
            "Candidate approved by reviewer."
        )
    elif reviewer_decision == "REJECT":
        st.error(
            "Candidate rejected · Production model remains active."
        )
    else:
        st.info(
            "Candidate review is pending."
        )

    st.caption(
        f"Reviewer: {reviewer_name or 'Pending'} · "
        f"Notes: {notes}"
    )


def show_simulation_details(latest):
    st.markdown(
        '<div class="section-label">Simulation verification</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current phase",
        str(latest["Phase"]),
    )

    col2.metric(
        "Target fault",
        f"Fault {int(latest['Target_Fault'])}",
    )

    col3.metric(
        "Actual simulated class",
        f"Fault {int(latest['Actual_Label'])}",
    )

    if int(latest["Is_Correct"]) == 1:
        st.success(
            "Simulation verification passed: latest "
            "prediction matches the simulated class."
        )
    else:
        st.error(
            "Simulation verification alert: latest "
            "prediction differs from the simulated class."
        )


def show_sensor_table(latest):
    st.markdown(
        '<div class="section-label">Live sensor instrumentation</div>',
        unsafe_allow_html=True,
    )

    sensor_table = pd.DataFrame(
        {
            "Sensor": config.FEATURE_COLUMNS,
            "Current value": [
                latest[column]
                for column in config.FEATURE_COLUMNS
            ],
        }
    )

    st.dataframe(
        sensor_table,
        hide_index=True,
        use_container_width=True,
    )


def show_trends(data):
    st.markdown(
        '<div class="section-label">Trend analysis</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Temperature trend")

        temperature_data = data[
            ["timestamp", "Temp1", "Temp2"]
        ].copy()

        temperature_data = (
            temperature_data.set_index(
                "timestamp"
            )
        )

        st.line_chart(
            temperature_data,
            height=280,
        )

    with right:
        st.markdown("#### RPM trend")

        rpm_data = data[
            ["timestamp", "RPM"]
        ].copy()

        rpm_data = rpm_data.set_index(
            "timestamp"
        )

        st.line_chart(
            rpm_data,
            height=280,
        )

    st.markdown("#### Prediction distribution")

    prediction_counts = (
        data["Predicted_Fault"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        prediction_counts,
        height=260,
    )


def show_health(data):
    st.markdown(
        '<div class="section-label">System health</div>',
        unsafe_allow_html=True,
    )

    missing_sensor_values = int(
        data[
            config.FEATURE_COLUMNS
        ].isna().sum().sum()
    )

    duplicate_rows = int(
        data.duplicated(
            subset=config.FEATURE_COLUMNS
        ).sum()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Records available",
        len(data),
    )

    col2.metric(
        "Missing sensor values",
        missing_sensor_values,
    )

    col3.metric(
        "Duplicate readings",
        duplicate_rows,
    )

    if missing_sensor_values == 0:
        st.success(
            "Sensor data quality check passed."
        )
    else:
        st.warning(
            "Missing sensor values detected."
        )

    if duplicate_rows == 0:
        st.success(
            "No duplicate sensor rows detected."
        )
    else:
        st.warning(
            "Duplicate sensor rows detected."
        )


def show_recent_records(data):
    st.markdown(
        '<div class="section-label">Recent operational records</div>',
        unsafe_allow_html=True,
    )

    base_columns = [
        "timestamp",
        "Predicted_Fault",
        "Confidence",
    ]

    optional_columns = [
        "Phase",
        "Target_Fault",
        "Actual_Label",
        "Is_Correct",
    ]

    columns = [
        column
        for column in base_columns + optional_columns
        if column in data.columns
    ]

    st.dataframe(
        data[columns].tail(25),
        hide_index=True,
        use_container_width=True,
    )


def show_overview(
    data,
    run_mode,
):
    latest = data.iloc[-1]

    predicted_fault = int(
        latest["Predicted_Fault"]
    )

    confidence = float(
        latest["Confidence"]
    )

    warning_status, consecutive_abnormal = (
        get_warning_status(
            data,
            predicted_fault,
        )
    )

    show_header(run_mode)

    show_alarm_strip(
        warning_status,
        predicted_fault,
        confidence,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Motor status",
        "RUNNING",
    )

    col2.metric(
        "Model prediction",
        f"Fault {predicted_fault}",
    )

    col3.metric(
        "Confidence",
        f"{confidence * 100:.2f}%",
    )

    col4.metric(
        "Warning status",
        warning_status,
    )

    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Consecutive abnormal",
        consecutive_abnormal,
    )

    col6.metric(
        "RPM",
        f"{float(latest['RPM']):.1f}",
    )

    col7.metric(
        "Temp1",
        f"{float(latest['Temp1']):.1f} °C",
    )

    col8.metric(
        "Current",
        f"{float(latest['Current']):.2f} A",
    )

    if run_mode == "simulation":
        show_simulation_details(
            latest,
        )
    else:
        st.info(
            "Real Motor Mode: actual fault labels are "
            "not available until external maintenance "
            "verification is completed."
        )

    show_sensor_table(latest)
    show_recent_records(data)
    show_trends(data)


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def show_live_monitoring(
    data_file,
    run_mode,
):
    st.markdown(
        '<div class="section-label">'
        'LIVE MONITORING · AUTO REFRESH ACTIVE'
        '</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"Reading the latest data every "
        f"{LIVE_REFRESH_SECONDS} seconds"
    )

    if not data_file.exists():
        st.warning(
            "Waiting for live sensor data..."
        )
        return

    try:
        data = pd.read_csv(
            data_file
        )
    except (
        pd.errors.EmptyDataError,
        PermissionError,
    ):
        st.warning(
            "The live data file is currently "
            "being written. Retrying..."
        )
        return

    if data.empty:
        st.warning(
            "The selected live data file is empty."
        )
        return

    required_columns = (
        ["timestamp"]
        + config.FEATURE_COLUMNS
        + [
            "Predicted_Fault",
            "Confidence",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        st.error(
            "Selected data source is missing "
            f"columns: {missing_columns}"
        )
        return

    latest = data.iloc[-1]

    predicted_fault = int(
        latest["Predicted_Fault"]
    )

    confidence = float(
        latest["Confidence"]
    )

    warning_status, consecutive_abnormal = (
        get_warning_status(
            data,
            predicted_fault,
        )
    )

    show_alarm_strip(
        warning_status,
        predicted_fault,
        confidence,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Model prediction",
        f"Fault {predicted_fault}",
    )

    col2.metric(
        "Confidence",
        f"{confidence * 100:.2f}%",
    )

    col3.metric(
        "Warning status",
        warning_status,
    )

    col4.metric(
        "Consecutive abnormal",
        consecutive_abnormal,
    )

    if run_mode == "simulation":
        show_simulation_details(
            latest,
        )
    else:
        st.info(
            "Real Motor Mode: actual fault labels are "
            "unavailable until external maintenance "
            "verification."
        )

    show_sensor_table(latest)

    st.markdown(
        '<div class="section-label">'
        'Recent operational records'
        '</div>',
        unsafe_allow_html=True,
    )

    base_columns = [
        "timestamp",
        "Predicted_Fault",
        "Confidence",
    ]

    optional_columns = [
        "Phase",
        "Target_Fault",
        "Actual_Label",
        "Is_Correct",
    ]

    columns = [
        column
        for column in (
            base_columns + optional_columns
        )
        if column in data.columns
    ]

    st.dataframe(
        data[columns].tail(25),
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Last dashboard refresh: "
        + datetime.now().strftime(
            "%H:%M:%S"
        )
    )


def main():
    run_mode, selected_page = show_sidebar()

    data_file = get_data_file(
        run_mode
    )

    if selected_page == "Model Governance":
        show_header(run_mode)
        show_governance()
        return

    if selected_page == "Live Monitoring":
        show_header(run_mode)
        show_live_monitoring(
            data_file,
            run_mode,
        )
        return

    if not data_file.exists():
        show_header(run_mode)

        st.warning(
            "Waiting for live sensor data..."
        )

        st.info(
            "Start the appropriate data source "
            f"for {run_mode}."
        )

        return

    data = pd.read_csv(
        data_file
    )

    if data.empty:
        show_header(run_mode)

        st.warning(
            "The selected live data file is empty."
        )

        return

    required_columns = (
        ["timestamp"]
        + config.FEATURE_COLUMNS
        + [
            "Predicted_Fault",
            "Confidence",
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        show_header(run_mode)

        st.error(
            "Selected data source is missing "
            f"columns: {missing_columns}"
        )

        return

    if selected_page == "Overview":
        show_overview(
            data,
            run_mode,
        )

    elif selected_page == "Trends":
        show_header(run_mode)
        show_trends(data)

    elif selected_page == "System Health":
        show_header(run_mode)
        show_health(data)


main()