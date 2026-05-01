
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import joblib

# ── Page config ──────────────────────────────────
st.set_page_config(
    page_title="Anglian Education Group — Student At-Risk System",
    page_icon="🎓",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .main-header h1 {
        color: white;
        font-size: 28px;
        margin: 0;
        padding: 0;
    }
    .main-header p {
        color: #b8d4f0;
        margin: 5px 0 0 0;
        font-size: 14px;
    }
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 5px solid #2d6a9f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .kpi-card.red { border-left-color: #E24B4A; }
    .kpi-card.green { border-left-color: #1D9E75; }
    .kpi-card.amber { border-left-color: #EF9F27; }
    .alert-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .section-header {
        font-size: 18px;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #2d6a9f;
        padding-bottom: 8px;
        margin: 20px 0 15px 0;
    }
    .risk-high {
        background: #FFE0E0;
        border-radius: 6px;
        padding: 4px 10px;
        color: #A32D2D;
        font-weight: 600;
    }
    .risk-medium {
        background: #FFF3CD;
        border-radius: 6px;
        padding: 4px 10px;
        color: #854F0B;
        font-weight: 600;
    }
    .risk-low {
        background: #D4EDDA;
        border-radius: 6px;
        padding: 4px 10px;
        color: #27500A;
        font-weight: 600;
    }
    .student-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("predictions.csv")
    df["withdrawal_probability"] = df["withdrawal_probability"].round(1)
    return df

# ── Load RF model ─────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("rf_model.pkl")

df       = load_data()
rf_model = load_model()

# ── Helper variables ──────────────────────────────
total_students  = len(df)
high_risk       = (df["risk_tier"] == "High Risk").sum()
medium_risk     = (df["risk_tier"] == "Medium Risk").sum()
low_risk        = (df["risk_tier"] == "Low Risk").sum()
withdrawal_rate = round((df["final_result"] == "Withdrawn").mean() * 100, 1)
avg_attendance  = round(df["avg_attendance"].mean(), 1)
today           = datetime.today().strftime("%d %B %Y")

# ── Encoding maps (must match step4_ml.ipynb exactly) ────────────
GENDER_MAP = {"Male": 0, "Female": 1, "Other": 2}

TRANSPORT_MAP = {"Walk": 0, "Cycle": 1, "Car": 2, "Bus": 3, "Train": 4}

EDUCATION_MAP = {
    "No formal quals"        : 1,
    "Lower than A level"     : 2,
    "A level or equivalent"  : 3,
    "HE qualification"       : 4,
    "Post Graduate"          : 5
}

IMD_MAP = {
    "90-100%": 1, "80-90%": 2, "70-80%": 3,
    "60-70%": 4,  "50-60%": 5, "40-50%": 6,
    "30-40%": 7,  "20-30%": 8, "10-20%": 9, "0-10%": 10
}

# ── Sidebar ───────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:10px 0'>
    <h2 style='color:#1a3a5c; font-size:20px'>🎓 AEG</h2>
    <p style='color:#666; font-size:12px'>Anglian Education Group</p>
    <p style='color:#999; font-size:11px'>Student At-Risk System</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Dashboard",
     "⚠️ Early Warnings",
     "📋 Risk Register",
     "🔍 Student Profile",
     "🔮 Predict New Student"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Live Summary")
st.sidebar.metric("Total Students", f"{total_students:,}")
st.sidebar.metric("Withdrawal Rate", f"{withdrawal_rate}%")

col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric("🔴 High", f"{high_risk:,}")
    st.metric("🟡 Medium", f"{medium_risk:,}")
with col2:
    st.metric("🟢 Low", f"{low_risk:,}")
    st.metric("📅 Avg Att.", f"{avg_attendance}%")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<p style='color:#999;font-size:11px'>Last updated: {today}</p>",
    unsafe_allow_html=True
)

# ════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ════════════════════════════════════════════════
if page == "🏠 Dashboard":

    st.markdown(f"""
    <div class='main-header'>
        <h1>🎓 Student At-Risk Prediction System</h1>
        <p>Anglian Education Group &nbsp;|&nbsp; 
        Last updated: {today}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👥 Total Students", f"{total_students:,}")
    with col2:
        st.metric("📉 Withdrawal Rate", f"{withdrawal_rate}%")
    with col3:
        st.metric("🔴 High Risk", f"{high_risk:,}",
                  delta="Needs immediate action", delta_color="off")
    with col4:
        st.metric("🟡 Medium Risk", f"{medium_risk:,}",
                  delta="Monitor closely", delta_color="off")
    with col5:
        st.metric("📅 Avg Attendance", f"{avg_attendance}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        outcome_counts = df["final_result"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        outcome_counts["Percentage"] = (
            outcome_counts["Count"] / total_students * 100
        ).round(1)
        outcome_counts["Label"] = (
            outcome_counts["Count"].astype(str) +
            " (" + outcome_counts["Percentage"].astype(str) + "%)"
        )
        fig1 = px.bar(
            outcome_counts, x="Outcome", y="Count",
            color="Outcome", text="Label",
            color_discrete_map={
                "Pass": "#378ADD", "Withdrawn": "#E24B4A",
                "Distinction": "#1D9E75", "Fail": "#EF9F27"
            },
            title="📊 Student Outcome Distribution"
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(showlegend=False, plot_bgcolor="white",
                           yaxis_range=[0, 8000],
                           yaxis_title="Number of Students")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        risk_counts = df["risk_tier"].value_counts().reset_index()
        risk_counts.columns = ["Risk Tier", "Count"]
        fig2 = px.pie(
            risk_counts, names="Risk Tier", values="Count",
            color="Risk Tier",
            color_discrete_map={
                "High Risk": "#E24B4A",
                "Medium Risk": "#EF9F27",
                "Low Risk": "#1D9E75"
            },
            title="🎯 Risk Tier Distribution", hole=0.4
        )
        fig2.update_traces(textposition="outside",
                           textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        course_w = df.groupby("course_name").apply(
            lambda x: round(
                (x["final_result"] == "Withdrawn").mean() * 100, 1),
            include_groups=False
        ).reset_index()
        course_w.columns = ["Course", "Withdrawal Rate %"]
        course_w = course_w.sort_values("Withdrawal Rate %", ascending=True)
        avg_w = round((df["final_result"] == "Withdrawn").mean() * 100, 1)
        fig3 = px.bar(
            course_w, x="Withdrawal Rate %", y="Course",
            orientation="h", color="Withdrawal Rate %",
            text="Withdrawal Rate %",
            color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
            title="📚 Withdrawal Rate by Course"
        )
        fig3.add_vline(x=avg_w, line_dash="dash", line_color="#185FA5",
                       annotation_text=f"Average: {avg_w}%")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        age_order = ["0-18","19-24","25-34","35-54","55+"]
        age_w = df.groupby("age_band").apply(
            lambda x: round(
                (x["final_result"] == "Withdrawn").mean() * 100, 1),
            include_groups=False
        ).reindex(age_order).reset_index()
        age_w.columns = ["Age Band", "Withdrawal Rate %"]
        fig4 = px.bar(
            age_w, x="Age Band", y="Withdrawal Rate %",
            color="Withdrawal Rate %", text="Withdrawal Rate %",
            color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
            title="👥 Withdrawal Rate by Age Band"
        )
        fig4.add_hline(y=avg_w, line_dash="dash", line_color="#185FA5",
                       annotation_text=f"Average: {avg_w}%")
        fig4.update_traces(textposition="outside")
        fig4.update_layout(coloraxis_showscale=False, plot_bgcolor="white",
                           yaxis_range=[0, 45])
        st.plotly_chart(fig4, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        att_outcome = df.groupby("final_result")[
            "avg_attendance"].mean().round(1).reset_index()
        att_outcome.columns = ["Outcome", "Avg Attendance %"]
        fig5 = px.bar(
            att_outcome, x="Outcome", y="Avg Attendance %",
            color="Outcome", text="Avg Attendance %",
            color_discrete_map={
                "Pass": "#378ADD", "Withdrawn": "#E24B4A",
                "Distinction": "#1D9E75", "Fail": "#EF9F27"
            },
            title="📅 Average Attendance by Outcome"
        )
        fig5.add_hline(y=80, line_dash="dash", line_color="red",
                       annotation_text="80% minimum threshold",
                       annotation_position="top right")
        fig5.update_traces(textposition="outside")
        fig5.update_layout(showlegend=False, plot_bgcolor="white",
                           yaxis_range=[0, 115])
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        imd_order = [
            "0-10%","10-20%","20-30%","30-40%","40-50%",
            "50-60%","60-70%","70-80%","80-90%","90-100%"
        ]
        imd_w = df.groupby("imd_band").apply(
            lambda x: round(
                (x["final_result"] == "Withdrawn").mean() * 100, 1),
            include_groups=False
        ).reindex(imd_order).reset_index()
        imd_w.columns = ["IMD Band", "Withdrawal Rate %"]
        fig6 = px.bar(
            imd_w, x="IMD Band", y="Withdrawal Rate %",
            color="Withdrawal Rate %",
            color_continuous_scale=["#1D9E75", "#EF9F27", "#E24B4A"],
            title="🏘️ Withdrawal by Deprivation (IMD)"
        )
        fig6.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
        st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════════════════════
# PAGE 2 — EARLY WARNINGS
# ════════════════════════════════════════════════
elif page == "⚠️ Early Warnings":

    st.markdown(f"""
    <div class='main-header'>
        <h1>⚠️ Early Warning System</h1>
        <p>Students requiring immediate attention — {today}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.error(f"🔴 {high_risk:,} students are HIGH RISK")
    with col2:
        st.warning(f"🟡 {medium_risk:,} students are MEDIUM RISK")
    with col3:
        funding_at_risk = high_risk * 6000
        st.error(f"💰 £{funding_at_risk:,} ESFA funding at risk")

    st.markdown("---")

    st.markdown(
        "<div class='section-header'>🚨 Top 20 Most At-Risk Students — Act Now</div>",
        unsafe_allow_html=True
    )

    top20 = df.nlargest(20, "withdrawal_probability")[[
        "full_name", "course_name", "age_band",
        "withdrawal_probability", "risk_tier",
        "avg_attendance", "num_prev_attempts",
        "disability", "part_time_job"
    ]].reset_index(drop=True)

    top20["withdrawal_probability"] = top20[
        "withdrawal_probability"].apply(lambda x: f"{x:.1f}%")
    top20["disability"]    = top20["disability"].map({0: "No", 1: "Yes"})
    top20["part_time_job"] = top20["part_time_job"].map({0: "No", 1: "Yes"})
    top20["avg_attendance"] = top20["avg_attendance"].apply(
        lambda x: f"{x:.1f}%")
    top20.columns = [
        "Name", "Course", "Age", "Risk %", "Risk Tier",
        "Attendance", "Prev Attempts", "Disability", "Part Time Job"
    ]

    def colour_risk(val):
        if val == "High Risk":
            return "background-color: #FFE0E0; color: #A32D2D; font-weight: bold"
        return ""

    st.dataframe(
        top20.style.map(colour_risk, subset=["Risk Tier"]),
        use_container_width=True, height=600
    )

    st.download_button(
        label="📥 Download High Risk List",
        data=df[df["risk_tier"] == "High Risk"][[
            "full_name", "course_name", "age_band",
            "withdrawal_probability", "risk_tier",
            "avg_attendance", "num_prev_attempts"
        ]].to_csv(index=False),
        file_name=f"high_risk_students_{today}.csv",
        mime="text/csv"
    )

    st.markdown("---")
    st.markdown(
        "<div class='section-header'>📊 High Risk by Department</div>",
        unsafe_allow_html=True
    )

    dept_risk = df[df["risk_tier"] == "High Risk"].groupby(
        "department").size().reset_index()
    dept_risk.columns = ["Department", "High Risk Count"]
    dept_risk = dept_risk.sort_values("High Risk Count", ascending=False)

    fig_dept = px.bar(
        dept_risk, x="Department", y="High Risk Count",
        color="High Risk Count", text="High Risk Count",
        color_continuous_scale=["#EF9F27", "#E24B4A"],
        title="High Risk Students by Department"
    )
    fig_dept.update_traces(textposition="outside")
    fig_dept.update_layout(coloraxis_showscale=False, plot_bgcolor="white")
    st.plotly_chart(fig_dept, use_container_width=True)

# ════════════════════════════════════════════════
# PAGE 3 — RISK REGISTER
# ════════════════════════════════════════════════
elif page == "📋 Risk Register":

    st.markdown(f"""
    <div class='main-header'>
        <h1>📋 Risk Register</h1>
        <p>All 15,000 students ranked by withdrawal risk</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        risk_filter = st.selectbox(
            "🎯 Risk Tier",
            ["All", "High Risk", "Medium Risk", "Low Risk"]
        )
    with col2:
        course_filter = st.selectbox(
            "📚 Course",
            ["All"] + sorted(df["course_name"].unique().tolist())
        )
    with col3:
        dept_filter = st.selectbox(
            "🏢 Department",
            ["All"] + sorted(df["department"].unique().tolist())
        )
    with col4:
        search = st.text_input("🔎 Search by Name")

    filtered = df.copy()
    if risk_filter != "All":
        filtered = filtered[filtered["risk_tier"] == risk_filter]
    if course_filter != "All":
        filtered = filtered[filtered["course_name"] == course_filter]
    if dept_filter != "All":
        filtered = filtered[filtered["department"] == dept_filter]
    if search:
        filtered = filtered[
            filtered["full_name"].str.contains(search, case=False)]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Showing {len(filtered):,} students**")
    with col2:
        hr = (filtered["risk_tier"] == "High Risk").sum()
        st.markdown(f"**🔴 High Risk: {hr:,}**")
    with col3:
        mr = (filtered["risk_tier"] == "Medium Risk").sum()
        st.markdown(f"**🟡 Medium Risk: {mr:,}**")

    display_cols = [
        "full_name", "course_name", "department",
        "age_band", "risk_tier",
        "withdrawal_probability", "avg_attendance", "final_result"
    ]
    filtered_display = filtered[display_cols].sort_values(
        "withdrawal_probability", ascending=False
    ).reset_index(drop=True)

    filtered_display["withdrawal_probability"] = (
        filtered_display["withdrawal_probability"]
        .apply(lambda x: f"{x:.1f}%")
    )
    filtered_display["avg_attendance"] = (
        filtered_display["avg_attendance"]
        .apply(lambda x: f"{x:.1f}%")
    )
    filtered_display.columns = [
        "Name", "Course", "Department", "Age",
        "Risk Tier", "Risk %", "Attendance", "Outcome"
    ]

    def colour_risk_register(val):
        if val == "High Risk":   return "background-color: #FFE0E0"
        elif val == "Medium Risk": return "background-color: #FFF3CD"
        elif val == "Low Risk":    return "background-color: #D4EDDA"
        return ""

    st.dataframe(
        filtered_display.style.map(
            colour_risk_register, subset=["Risk Tier"]),
        use_container_width=True, height=550
    )

    st.download_button(
        label="📥 Download Filtered List",
        data=filtered_display.to_csv(index=False),
        file_name=f"risk_register_{today}.csv",
        mime="text/csv"
    )

# ════════════════════════════════════════════════
# PAGE 4 — STUDENT PROFILE
# ════════════════════════════════════════════════
elif page == "🔍 Student Profile":

    st.markdown(f"""
    <div class='main-header'>
        <h1>🔍 Student Profile</h1>
        <p>Search for any student to see their full risk profile 
        and personalised recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    search_name = st.text_input(
        "🔎 Type a student name",
        placeholder="e.g. James Smith, Aisha Khan..."
    )

    if search_name:
        matches = df[df["full_name"].str.contains(search_name, case=False)]

        if len(matches) == 0:
            st.warning(
                f"No students found matching '{search_name}'. "
                "Try a different name."
            )
        else:
            student_options = matches.apply(
                lambda x: f"{x['full_name']} ({x['student_id']}) — {x['course_name']}",
                axis=1
            ).tolist()

            selected    = st.selectbox(
                f"Found {len(matches)} student(s) — select one:",
                student_options
            )
            selected_id = selected.split("(")[1].split(")")[0]
            student     = df[df["student_id"] == selected_id].iloc[0]

            st.markdown("---")

            risk = student["risk_tier"]
            prob = student["withdrawal_probability"]

            if risk == "High Risk":
                st.error(
                    f"🔴 HIGH RISK — {prob:.1f}% withdrawal probability "
                    f"| Immediate action required")
            elif risk == "Medium Risk":
                st.warning(
                    f"🟡 MEDIUM RISK — {prob:.1f}% withdrawal probability "
                    f"| Monitor closely")
            else:
                st.success(
                    f"🟢 LOW RISK — {prob:.1f}% withdrawal probability "
                    f"| On track")

            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(
                    "<div class='section-header'>👤 Personal Details</div>",
                    unsafe_allow_html=True)
                st.write(f"**Name:** {student['full_name']}")
                st.write(f"**Student ID:** {student['student_id']}")
                st.write(f"**Gender:** {student['gender']}")
                st.write(f"**Age Band:** {student['age_band']}")
                st.write(f"**Region:** {student['region']}")
                st.write(f"**Ethnicity:** {student['ethnicity']}")
                st.write(f"**IMD Band:** {student['imd_band']}")

            with col2:
                st.markdown(
                    "<div class='section-header'>📚 Academic Details</div>",
                    unsafe_allow_html=True)
                st.write(f"**Course:** {student['course_name']}")
                st.write(f"**Department:** {student['department']}")
                st.write(f"**Level:** {student['level']}")
                st.write(f"**Credits:** {student['studied_credits']}")
                st.write(f"**Previous Attempts:** {student['num_prev_attempts']}")
                st.write(f"**Final Outcome:** {student['final_result']}")
                st.write(f"**Education Level:** {student['highest_education']}")

            with col3:
                st.markdown(
                    "<div class='section-header'>⚠️ Risk Factors</div>",
                    unsafe_allow_html=True)
                disability = "Yes ⚠️" if student["disability"]==1 else "No ✅"
                job        = "Yes ⚠️" if student["part_time_job"]==1 else "No ✅"
                mentor_s   = "Yes ✅" if student["mentor_assigned"]==1 else "No ⚠️"
                english    = "No ⚠️"  if student["english_first_lang"]==1 else "Yes ✅"
                st.write(f"**Disability:** {disability}")
                st.write(f"**Part Time Job:** {job}")
                st.write(f"**Transport:** {student['transport']}")
                st.write(f"**English First Language:** {english}")
                st.write(f"**Support Sessions:** {student['support_sessions']} / 8")
                st.write(f"**Mentor Assigned:** {mentor_s}")

            st.markdown("---")

            st.markdown(
                "<div class='section-header'>📊 Performance Metrics</div>",
                unsafe_allow_html=True)
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Avg Score", f"{student['avg_score']}%",
                          delta="vs 55% avg" if student['avg_score'] > 55 else None)
            with col2:
                att_delta = round(student['avg_attendance'] - avg_attendance, 1)
                st.metric("Attendance", f"{student['avg_attendance']}%",
                          delta=f"{att_delta}% vs college avg")
            with col3:
                st.metric("Weekly Clicks", f"{student['avg_weekly_clicks']}")
            with col4:
                st.metric("Assignments Done", f"{int(student['total_submitted'])}")
            with col5:
                st.metric("Late Submissions", f"{int(student['late_submissions'])}")

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=float(student["avg_score"]),
                    domain={"x": [0,1], "y": [0,1]},
                    title={"text": "Average Assessment Score"},
                    delta={"reference": 55},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2d6a9f"},
                        "steps": [
                            {"range": [0,40],  "color": "#FFE0E0"},
                            {"range": [40,70], "color": "#FFF3CD"},
                            {"range": [70,100],"color": "#D4EDDA"}
                        ],
                        "threshold": {
                            "line": {"color":"red","width":4},
                            "thickness": 0.75, "value": 40
                        }
                    }
                ))
                fig_gauge.update_layout(height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                fig_att = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=float(student["avg_attendance"]),
                    domain={"x": [0,1], "y": [0,1]},
                    title={"text": "Attendance Rate"},
                    delta={"reference": 80},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1D9E75"},
                        "steps": [
                            {"range": [0,60],  "color": "#FFE0E0"},
                            {"range": [60,80], "color": "#FFF3CD"},
                            {"range": [80,100],"color": "#D4EDDA"}
                        ],
                        "threshold": {
                            "line": {"color":"red","width":4},
                            "thickness": 0.75, "value": 80
                        }
                    }
                ))
                fig_att.update_layout(height=280)
                st.plotly_chart(fig_att, use_container_width=True)

            st.markdown("---")

            st.markdown(
                "<div class='section-header'>📌 Personalised Recommendations</div>",
                unsafe_allow_html=True)

            if risk == "High Risk":
                st.error("⚠️ IMMEDIATE ACTION REQUIRED")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**This week:**")
                    st.markdown("- 📞 Call student for welfare check")
                    st.markdown("- 👥 Assign personal tutor immediately")
                    st.markdown("- 💰 Refer to hardship fund")
                    st.markdown("- 📋 Create learning support plan")
                with col2:
                    st.markdown("**Based on risk factors:**")
                    if student["disability"] == 1:
                        st.markdown("- ♿ Review disability support plan")
                    if student["part_time_job"] == 1:
                        st.markdown("- ⏰ Offer flexible deadlines")
                    if student["num_prev_attempts"] >= 2:
                        st.markdown("- 🔄 Review previous withdrawal reasons")
                    if student["transport"] == "Train":
                        st.markdown("- 🚂 Apply for transport bursary")
                    if student["imd_band"] in ["0-10%","10-20%","20-30%"]:
                        st.markdown("- 💷 Priority hardship fund referral")
                    if student["mentor_assigned"] == 0:
                        st.markdown("- 👤 Assign mentor urgently")
                    if student["support_sessions"] < 2:
                        st.markdown("- 📚 Increase support sessions")

            elif risk == "Medium Risk":
                st.warning("👀 MONITOR CLOSELY")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Next 2 weeks:**")
                    st.markdown("- 📅 Schedule tutorial")
                    st.markdown("- 📊 Monitor attendance weekly")
                    st.markdown("- 💬 Informal check-in at next class")
                    st.markdown("- 🎯 Set clear assessment targets")
                with col2:
                    st.markdown("**Based on risk factors:**")
                    if student["avg_attendance"] < 80:
                        st.markdown("- 📋 Discuss attendance concerns")
                    if student["support_sessions"] < 2:
                        st.markdown("- 👥 Encourage support services")
                    if student["mentor_assigned"] == 0:
                        st.markdown("- 👤 Consider peer mentor")
                    if student["avg_score"] < 50:
                        st.markdown("- 📚 Academic support sessions")

            else:
                st.success("✅ STUDENT IS ON TRACK")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Ongoing:**")
                    st.markdown("- ✅ Continue current support level")
                    st.markdown("- 🌟 Celebrate good attendance")
                    st.markdown("- 📈 Encourage higher achievement")
                    st.markdown("- 📅 Standard termly review")
                with col2:
                    st.markdown("**Opportunities:**")
                    st.markdown("- 👥 Consider as peer mentor")
                    st.markdown("- 🏆 Nominate for achievement award")
                    st.markdown("- 📚 Suggest enrichment activities")

# ════════════════════════════════════════════════
# PAGE 5 — PREDICT NEW STUDENT
# ════════════════════════════════════════════════
elif page == "🔮 Predict New Student":

    st.markdown(f"""
    <div class='main-header'>
        <h1>🔮 Predict New Student Risk</h1>
        <p>Enter a prospective student's details to get 
        an instant withdrawal risk prediction</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "💡 Use this tool at enrolment to identify at-risk "
        "students before they start — enabling proactive support "
        "from day one. Powered by the same Random Forest model "
        "trained on 15,000 students."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Details**")
        gender        = st.selectbox("Gender", ["Male","Female","Other"])
        age_band      = st.selectbox("Age Band",
                            ["0-18","19-24","25-34","35-54","55+"])
        disability    = st.selectbox("Disability", ["No","Yes"])
        part_time_job = st.selectbox("Part Time Job", ["No","Yes"])
        transport     = st.selectbox("Transport",
                            ["Walk","Cycle","Car","Bus","Train"])
        english_first = st.selectbox("English First Language", ["Yes","No"])

    with col2:
        st.markdown("**📚 Academic Details**")
        prev_attempts = st.slider("Previous Attempts", 0, 2, 0)
        credits       = st.selectbox("Credits", [30, 60, 90, 120])
        highest_edu   = st.selectbox("Highest Education", [
                            "No formal quals",
                            "Lower than A level",
                            "A level or equivalent",
                            "HE qualification",
                            "Post Graduate"
                        ])
        course        = st.selectbox("Course",
                            sorted(df["course_name"].unique().tolist()))

    with col3:
        st.markdown("**🤝 Support Details**")
        imd_band         = st.selectbox("IMD Band", [
                               "0-10%","10-20%","20-30%","30-40%","40-50%",
                               "50-60%","60-70%","70-80%","80-90%","90-100%"
                           ])
        support_sessions = st.slider("Support Sessions Planned", 0, 8, 2)
        mentor           = st.selectbox("Mentor to be Assigned", ["No","Yes"])

    st.markdown("---")

    if st.button("🔮 Predict Withdrawal Risk", type="primary"):

        # ── Build feature vector (same order as training) ──────────
        input_data = np.array([[
            GENDER_MAP[gender],                      # gender_encoded
            1 if disability == "Yes" else 0,         # disability
            prev_attempts,                           # num_prev_attempts
            credits,                                 # studied_credits
            1 if part_time_job == "Yes" else 0,      # part_time_job
            0 if english_first == "Yes" else 1,      # english_first_lang
            TRANSPORT_MAP[transport],                # transport_encoded
            EDUCATION_MAP[highest_edu],              # education_encoded
            IMD_MAP[imd_band],                       # imd_encoded
            support_sessions,                        # support_sessions
            1 if mentor == "Yes" else 0              # mentor_assigned
        ]])

        # ── Random Forest prediction ────────────────────────────────
        risk_score = round(
            rf_model.predict_proba(input_data)[0][1] * 100, 1)

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        # Gauge chart
        fig_pred = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={"x":[0,1],"y":[0,1]},
            title={"text":"Withdrawal Risk Score (Random Forest)"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":"#2d6a9f"},
                "steps":[
                    {"range":[0,40],  "color":"#D4EDDA"},
                    {"range":[40,70], "color":"#FFF3CD"},
                    {"range":[70,100],"color":"#FFE0E0"}
                ],
                "threshold":{
                    "line":{"color":"red","width":4},
                    "thickness":0.75,
                    "value":70
                }
            }
        ))
        fig_pred.update_layout(height=300)
        st.plotly_chart(fig_pred, use_container_width=True)

        col1, col2 = st.columns(2)

        if risk_score >= 70:
            with col1:
                st.error(
                    f"🔴 HIGH RISK — {risk_score}% withdrawal probability")
                st.markdown("**Immediate actions at enrolment:**")
                st.markdown("- 📞 Pre-enrolment welfare call")
                st.markdown("- 👥 Assign personal tutor day one")
                st.markdown("- 💰 Refer to financial support")
                st.markdown("- 📋 Learning support plan")
                st.markdown("- 📅 Weekly check-ins from week one")
            with col2:
                st.markdown("**Risk factors identified:**")
                if prev_attempts >= 2:
                    st.markdown("- ⚠️ Multiple previous attempts")
                if disability == "Yes":
                    st.markdown("- ⚠️ Disability support needed")
                if part_time_job == "Yes":
                    st.markdown("- ⚠️ Employment pressure")
                if transport in ["Train","Bus"]:
                    st.markdown("- ⚠️ Transport dependency")
                if imd_band in ["0-10%","10-20%","20-30%"]:
                    st.markdown("- ⚠️ High deprivation area")
                if age_band in ["35-54","55+"]:
                    st.markdown("- ⚠️ Mature learner pressures")

        elif risk_score >= 40:
            with col1:
                st.warning(
                    f"🟡 MEDIUM RISK — {risk_score}% withdrawal probability")
                st.markdown("**Recommended at enrolment:**")
                st.markdown("- 📅 Early tutorial in week 2")
                st.markdown("- 👀 Monitor attendance weekly")
                st.markdown("- 💬 Introduce to support services")
                st.markdown("- 🎯 Set clear expectations early")
            with col2:
                st.markdown("**Watch points:**")
                if part_time_job == "Yes":
                    st.markdown("- 👀 Monitor work-study balance")
                if transport in ["Train","Bus"]:
                    st.markdown("- 👀 Watch for transport issues")
                if prev_attempts == 1:
                    st.markdown("- 👀 Review previous withdrawal reason")

        else:
            with col1:
                st.success(
                    f"🟢 LOW RISK — {risk_score}% withdrawal probability")
                st.markdown("**Standard enrolment process:**")
                st.markdown("- ✅ Normal induction")
                st.markdown("- 📅 Termly review")
                st.markdown("- 🌟 Encourage peer mentoring")
            with col2:
                st.markdown("**Positive indicators:**")
                if prev_attempts == 0:
                    st.markdown("- ✅ First attempt student")
                if disability == "No":
                    st.markdown("- ✅ No additional learning needs")
                if transport in ["Walk","Cycle","Car"]:
                    st.markdown("- ✅ Reliable transport")
                if mentor == "Yes":
                    st.markdown("- ✅ Mentor assigned")
