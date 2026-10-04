import io
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from core import AutoDataCleaner, FeatureBuilder, ModelEngine, PipelinePredictor

st.set_page_config(page_title="Predictive OS", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

# -----------------------------
# Theme
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp { background:#f8fafc; font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; }
.block-container { max-width:1400px; padding:2rem 2.5rem 3rem; }
[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e5e7eb; }
h1,h2,h3,h4 { color:#111827; letter-spacing:-.01em; font-weight:600; }
p, label, .stCaption { color:#475467; font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; }
.metric-card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:16px 18px; min-height:110px; box-shadow:0 1px 2px rgba(0,0,0,0.05); }
.metric-label { font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:.05em; color:#6b7280; }
.metric-value { font-size:24px; font-weight:600; color:#111827; margin-top:6px; }
.metric-help { font-size:12px; color:#9ca3af; margin-top:4px; }
.uvp { background:#111827; color:#fff; border-radius:10px; padding:18px 20px; margin:8px 0 20px; box-shadow:0 4px 6px rgba(0,0,0,0.1); }
.uvp-title { font-size:11px; font-weight:600; letter-spacing:.08em; text-transform:uppercase; color:#9ca3af; }
.uvp-main { font-size:20px; font-weight:600; margin:4px 0 6px; }
.uvp-copy { color:#d1d5db; font-size:13px; line-height:1.5; max-width:960px; }
.section { margin-top:20px; margin-bottom:8px; }
.small-note { font-size:12px; color:#9ca3af; }
.nav-label { font-size:11px; font-weight:600; color:#9ca3af; letter-spacing:.06em; text-transform:uppercase; margin:6px 0 8px; }
[data-testid="stSidebar"] .stButton > button { width:100%; justify-content:flex-start; text-align:left; border:1px solid transparent; background:transparent; color:#475467; border-radius:6px; padding:8px 12px; font-size:14px; font-weight:500; margin:2px 0; font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif; }
[data-testid="stSidebar"] .stButton > button:hover { background:#f3f4f6; border-color:#e5e7eb; color:#111827; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:#f3f4f6; color:#111827; border-color:#d1d5db; font-weight:600; }
.report-card { background:#fff; border:1px solid #e5e7eb; border-radius:8px; padding:16px 18px; box-shadow:0 1px 2px rgba(0,0,0,0.05); }
.status-pill { display:inline-block; padding:3px 8px; border-radius:999px; font-size:11px; font-weight:600; background:#ecfdf3; color:#059669; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# State
# -----------------------------
def init_state():
    defaults = {
        "model_engine": None, "cleaner": None, "feature_builder": None,
        "results_dict": {}, "best_score": None, "X_cols": [],
        "drop_columns": [], "training_rows": 0, "training_features": 0,
        "training_time": None, "last_predictions": None, "audit_log": [],
        "confidence_threshold": 75, "drift_tolerance": 0.15,
        "slack_notifications": False, "email_summaries": False, "webhook_url": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

init_state()


def card(title, value, help_text, progress_percent=None):
    progress_html = ""
    if progress_percent is not None:
        progress_width = min(100, max(0, progress_percent))
        progress_color = "#059669" if progress_percent >= 70 else "#d97706" if progress_percent >= 40 else "#dc2626"
        progress_html = f'<div style="margin-top:8px;height:4px;background:#e5e7eb;border-radius:2px;overflow:hidden"><div style="height:100%;width:{progress_width}%;background:{progress_color};border-radius:2px"></div></div>'
    return f'''<div class="metric-card"><div class="metric-label">{title}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div>{progress_html}</div>'''


def add_audit(event, details):
    st.session_state.audit_log.insert(0, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Event": event,
        "Details": details,
    })
    st.session_state.audit_log = st.session_state.audit_log[:100]


def make_pdf_report():
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 55
    c.setFont("Helvetica-Bold", 18); c.drawString(45, y, "Predictive OS — Governance Report")
    y -= 28
    c.setFont("Helvetica", 10)
    lines = [
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Best model: {st.session_state.model_engine.best_model_name if st.session_state.model_engine else 'N/A'}",
        f"Validation score: {st.session_state.best_score:.4f}" if st.session_state.best_score is not None else "Validation score: N/A",
        f"CV folds: {getattr(st.session_state.model_engine, 'cv_folds', 'N/A') if st.session_state.model_engine else 'N/A'}",
        f"Training rows: {st.session_state.training_rows}",
        f"Features: {st.session_state.training_features}",
        f"Human-audit threshold: {st.session_state.confidence_threshold}%",
        "",
        "Audit events:",
    ]
    for line in lines:
        c.drawString(45, y, line); y -= 16
    for event in st.session_state.audit_log[:20]:
        text = f"{event['Timestamp']} | {event['Event']} | {event['Details']}"
        c.drawString(55, y, text[:115]); y -= 14
        if y < 55:
            c.showPage(); y = height - 55; c.setFont("Helvetica", 10)
    c.save(); buf.seek(0)
    return buf.getvalue()

# -----------------------------
# Sidebar
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "Overview"

def navigate(label):
    st.session_state.page = label

st.sidebar.markdown("<div style='padding:8px 0 20px'><div style='font-size:16px;font-weight:600;color:#111827;letter-spacing:-.01em'>Predictive OS</div><div style='font-size:11px;color:#6b7280;margin-top:2px'>Industrial decision intelligence</div></div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='nav-label'>Workspace</div>", unsafe_allow_html=True)
for label, key in [("Overview", "nav_overview"), ("Make Predictions", "nav_live"), ("Intelligence Reports", "nav_reports"), ("System Settings", "nav_settings")]:
    if st.sidebar.button(label, key=key, use_container_width=True, type="primary" if st.session_state.page == label else "secondary"):
        navigate(label)
        st.rerun()
page = st.session_state.page

status = "Operational" if st.session_state.model_engine else "Awaiting model"
st.sidebar.markdown(f"<div style='position:fixed;bottom:22px;width:235px;background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;padding:12px'><b style='font-size:12px;color:#374151'>SYSTEM</b><br><span style='font-size:12px;color:{'#059669' if status == 'Operational' else '#d97706'}'>● {status}</span><br><span style='font-size:11px;color:#9ca3af'>Predictive OS v1.1</span></div>", unsafe_allow_html=True)

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    st.title("Predictive OS")
    st.caption("Decision infrastructure for industrial telemetry — from raw sensor data to governed maintenance decisions.")
    
    if not st.session_state.model_engine:
        st.markdown("""
        <div class="uvp"><div class="uvp-title">Welcome to Predictive OS</div>
        <div class="uvp-main">Quick Start Guide</div>
        <div class="uvp-copy">Get started in 3 simple steps:<br>
        1. <b>Upload</b> your training telemetry CSV below<br>
        2. <b>Select</b> your target variable and task type<br>
        3. <b>Train</b> to benchmark models and deploy the best performer</div></div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="uvp"><div class="uvp-title">Decision intelligence for industrial operations</div>
        <div class="uvp-main">Predict with confidence. Escalate with control.</div>
        <div class="uvp-copy">Predictive OS benchmarks multiple ML models, scores unseen telemetry, and applies a confidence gate before an automated decision is accepted. Uncertain cases are routed to human review and recorded in an auditable decision trail.</div></div>
        """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    if st.session_state.model_engine:
        score_percent = st.session_state.best_score * 100
        c1.markdown(card("Status", "READY", "Pipeline trained and available"), unsafe_allow_html=True)
        c2.markdown(card("Best Model", st.session_state.model_engine.best_model_name, "Selected by cross-validation"), unsafe_allow_html=True)
        c3.markdown(card("Validation", f"{st.session_state.best_score:.3f}", "F1 macro for classification", progress_percent=score_percent), unsafe_allow_html=True)
        c4.markdown(card("Features", st.session_state.training_features, "Stable inference schema"), unsafe_allow_html=True)
    else:
        c1.markdown(card("Status", "SETUP", "Upload telemetry to begin"), unsafe_allow_html=True)
        c2.markdown(card("Models", "3", "RF · XGBoost · LightGBM"), unsafe_allow_html=True)
        c3.markdown(card("Validation", "5-fold", "Adaptive when classes are small"), unsafe_allow_html=True)
        c4.markdown(card("Audit", "READY", "Confidence-gated review"), unsafe_allow_html=True)

    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    
    if not st.session_state.model_engine:
        st.markdown("""
        <div class='report-card' style='border-left:4px solid #111827'>
            <div style='font-size:13px;font-weight:600;color:#111827;margin-bottom:6px'>Workflow Overview</div>
            <div style='font-size:12px;color:#667085;line-height:1.6'>
            <b>1. Train:</b> Upload data → Select target → Benchmark models<br>
            <b>2. Infer:</b> Upload new data → Get predictions with confidence scores<br>
            <b>3. Audit:</b> Review human-escalated cases → Export governance reports
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader("Train a decision pipeline")
        uploaded = st.file_uploader("Training telemetry CSV", type="csv", key="train_upload")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                df.columns = df.columns.str.strip()
                if df.empty:
                    st.error("The CSV is empty.")
                else:
                    default_idx = list(df.columns).index("Machine failure") if "Machine failure" in df.columns else len(df.columns)-1
                    a,b = st.columns(2)
                    with a:
                        target = st.selectbox("Target variable", df.columns, index=default_idx, help="Select the column you want to predict (e.g., failure status, temperature)")
                        task = st.radio("Task", ["classification", "regression"], horizontal=True, help="Classification: predict categories (e.g., fail/pass). Regression: predict numbers (e.g., temperature).")
                    with b:
                        drops = st.multiselect("Optional columns to exclude", [c for c in df.columns if c != target], help="Remove irrelevant columns like IDs or timestamps to improve model performance")
                        st.caption(f"{len(df):,} rows · {len(df.columns):,} columns")

                    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
                    st.subheader("Data exploration")
                    
                    with st.expander("View feature distributions", expanded=False):
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        if numeric_cols:
                            cols_to_show = numeric_cols[:6]
                            n_cols = min(3, len(cols_to_show))
                            if n_cols > 0:
                                chart_cols = st.columns(n_cols)
                                for idx, col in enumerate(cols_to_show):
                                    with chart_cols[idx % n_cols]:
                                        fig = px.histogram(df, x=col, nbins=30, title=col)
                                        fig.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                                        fig.update_traces(marker_color="#111827", marker_line_color="#e5e7eb", marker_line_width=1)
                                        st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No numeric columns found for distribution visualization.")

                    with st.expander("View correlation heatmap", expanded=False):
                        numeric_df = df.select_dtypes(include=[np.number])
                        if len(numeric_df.columns) > 1:
                            corr_matrix = numeric_df.corr()
                            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", range_color=[-1, 1])
                            fig.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Need at least 2 numeric columns for correlation analysis.")

                    if st.button("Train & benchmark", type="primary", use_container_width=True):
                        if target in drops:
                            st.error("The target cannot also be excluded.")
                        else:
                            with st.spinner("Cleaning telemetry, benchmarking models, and fitting the winner…"):
                                cleaner = AutoDataCleaner(df)
                                cleaned = cleaner.clean_all()
                                if target not in cleaned.columns:
                                    raise ValueError("The selected target was removed during cleaning. Check missing values in the target column.")
                                y = cleaned[target]
                                X_raw = cleaned.drop(columns=[target] + drops, errors="ignore")
                                if X_raw.shape[1] == 0:
                                    raise ValueError("No predictor features remain after exclusions.")
                                builder = FeatureBuilder()
                                X = builder.fit_transform(X_raw)
                                engine = ModelEngine(task_type=task)
                                results, best = engine.train_and_benchmark(X, y)

                                st.session_state.update({
                                    "cleaner": cleaner, "feature_builder": builder, "model_engine": engine,
                                    "results_dict": results, "best_score": results[best],
                                    "X_cols": list(X.columns), "drop_columns": drops,
                                    "training_rows": len(cleaned), "training_features": X.shape[1],
                                    "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "last_predictions": None,
                                })
                                add_audit("MODEL_TRAINED", f"{best} selected from RF/XGBoost/LightGBM")
                                st.success(f"Pipeline ready. {best} selected with validation score {results[best]:.4f}.")
                                st.rerun()
            except Exception as exc:
                st.error(f"Training failed safely: {exc}")
    with right:
        st.subheader("Model intelligence")
        if st.session_state.model_engine:
            scores = pd.DataFrame({"Model": list(st.session_state.results_dict.keys()), "Score": list(st.session_state.results_dict.values())})
            scores = scores.sort_values("Score", ascending=False).reset_index(drop=True)
            scores["Score"] = scores["Score"].map(lambda x: f"{x:.3f}")
            st.dataframe(scores, use_container_width=True, hide_index=True)
            model = st.session_state.model_engine.best_model
            if hasattr(model, "feature_importances_") and len(st.session_state.X_cols) > 0:
                imp = pd.DataFrame({"Feature": st.session_state.X_cols, "Importance": model.feature_importances_}).nlargest(7, "Importance").sort_values("Importance")
                fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Blues")
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), showlegend=False)
                fig.update_traces(marker_line_color="#e5e7eb", marker_line_width=1)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Train a pipeline to populate model comparison and feature impact.")

# -----------------------------
# Live inference
# -----------------------------
elif page == "Make Predictions":
    st.title("Make Predictions")
    st.caption("Run unseen telemetry through the exact training-time cleaning and feature schema.")
    if not st.session_state.model_engine:
        st.markdown("""
        <div class='report-card'>
            <div style='font-size:14px;font-weight:600;color:#111827;margin-bottom:8px'>No model trained yet</div>
            <div style='font-size:13px;color:#667085;margin-bottom:12px'>You need to train a decision pipeline before running inference on new data.</div>
            <div style='font-size:12px;color:#475467;background:#f3f4f6;padding:8px 12px;border-radius:6px'>
            → Go to <b>Overview</b> tab to upload training data and train a model
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        f = st.file_uploader("Unseen telemetry CSV", type="csv", key="infer_upload")
        if f:
            try:
                test_df = pd.read_csv(f)
                test_df.columns = test_df.columns.str.strip()
                st.caption(f"Loaded {len(test_df):,} records")
                if st.button("Generate decisions", type="primary", use_container_width=True):
                    with st.spinner("Scoring telemetry…"):
                        predictor = PipelinePredictor(st.session_state.cleaner, st.session_state.feature_builder, st.session_state.model_engine)
                        test_clean = test_df.drop(columns=st.session_state.drop_columns, errors="ignore")
                        predictions = predictor.predict_unseen(test_clean, st.session_state.confidence_threshold)
                        st.session_state.last_predictions = predictions
                        review_count = int(predictions["Requires Human Audit"].sum()) if "Requires Human Audit" in predictions else 0
                        add_audit("INFERENCE_RUN", f"{len(predictions)} records scored; {review_count} routed to human review")
            except Exception as exc:
                st.error(f"Inference failed safely: {exc}")

        if st.session_state.last_predictions is not None:
            pred = st.session_state.last_predictions
            m1,m2,m3 = st.columns(3)
            m1.markdown(card("Records scored", f"{len(pred):,}", "Unseen telemetry"), unsafe_allow_html=True)
            if "Requires Human Audit" in pred:
                review = int(pred["Requires Human Audit"].sum())
                m2.markdown(card("Human review", f"{review:,}", f"Below {st.session_state.confidence_threshold}% confidence"), unsafe_allow_html=True)
                m3.markdown(card("Auto decisions", f"{len(pred)-review:,}", "Above configured threshold"), unsafe_allow_html=True)
            
            st.markdown("<div class='section'></div>", unsafe_allow_html=True)
            
            with st.expander("Explainability Panel - Feature Contributions", expanded=False):
                st.caption("Analyze which features most influenced specific predictions")
                row_idx = st.slider("Select record to analyze", 0, min(len(pred)-1, 99), 0, help="Slide to view different predictions")
                
                col_left, col_right = st.columns([1, 1.5])
                with col_left:
                    selected_row = pred.iloc[row_idx]
                    st.markdown(f"""
                    <div class='report-card'>
                        <div style='font-size:12px;color:#667085;margin-bottom:8px'>Prediction #{row_idx + 1}</div>
                        <div style='font-size:20px;font-weight:600;color:#111827;margin:4px 0'>{selected_row.get('Prediction', 'N/A')}</div>
                        <div style='font-size:13px;color:#667085'>Confidence: {selected_row.get('Confidence Score (%)', 'N/A')}%</div>
                        <div style='margin-top:8px'><span class='status-pill'>{selected_row.get('Decision Status', 'N/A')}</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_right:
                    model = st.session_state.model_engine.best_model
                    if hasattr(model, "feature_importances_") and len(st.session_state.X_cols) > 0:
                        imp = pd.DataFrame({"Feature": st.session_state.X_cols, "Importance": model.feature_importances_})
                        imp = imp.sort_values("Importance", ascending=False).head(8)
                        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis")
                        fig.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0), showlegend=False, xaxis_title="Feature Influence", yaxis_title="")
                        fig.update_traces(marker_line_color="#e5e7eb", marker_line_width=1)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption("Top features driving model decisions (global importance)")
                    else:
                        st.info("Feature importance not available for this model type.")
            
            st.markdown("<div class='section'></div>", unsafe_allow_html=True)
            st.dataframe(pred.head(250), use_container_width=True, hide_index=True)
            st.download_button("Download decision CSV", pred.to_csv(index=False), "predictive_os_decisions.csv", "text/csv", type="primary")

# -----------------------------
# Intelligence Reports
# -----------------------------
elif page == "Intelligence Reports":
    st.title("Intelligence Reports")
    st.caption("A concise operational record of model performance, decision controls, and recent activity.")
    if not st.session_state.model_engine:
        st.markdown("""
        <div class='report-card'>
            <div style='font-size:14px;font-weight:600;color:#111827;margin-bottom:8px'>No intelligence available yet</div>
            <div style='font-size:13px;color:#667085;margin-bottom:12px'>Governance reports and decision analytics become available after training a model.</div>
            <div style='font-size:12px;color:#475467;background:#f3f4f6;padding:8px 12px;border-radius:6px'>
            → Train a model in <b>Overview</b>, then run <b>Make Predictions</b> to populate this dashboard
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(card("Production candidate", st.session_state.model_engine.best_model_name, "Highest cross-validation score"), unsafe_allow_html=True)
        c2.markdown(card("Validation score", f"{st.session_state.best_score:.3f}", "Macro F1" if st.session_state.model_engine.task_type == "classification" else "R²"), unsafe_allow_html=True)
        c3.markdown(card("Records trained", f"{st.session_state.training_rows:,}", "Telemetry rows used"), unsafe_allow_html=True)
        c4.markdown(card("Review threshold", f"{st.session_state.confidence_threshold}%", "Human-review gate"), unsafe_allow_html=True)

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Model benchmark")
            scores = pd.DataFrame({"Model": list(st.session_state.results_dict.keys()), "Score": list(st.session_state.results_dict.values())})
            scores = scores.sort_values("Score", ascending=False).reset_index(drop=True)
            scores["Score"] = scores["Score"].round(4)
            scores["Selection"] = scores["Model"].eq(st.session_state.model_engine.best_model_name).map({True: "Selected", False: "Evaluated"})
            st.dataframe(scores, use_container_width=True, hide_index=True)
            st.caption(f"Selection based on {st.session_state.model_engine.cv_folds}-fold cross-validation.")
        with right:
            st.subheader("Governance posture")
            st.markdown(
                f"<div class='report-card'><div style='font-size:12px;color:#667085'>Decision gate</div>"
                f"<div style='font-size:22px;font-weight:700;color:#111827;margin:4px 0 12px'>Confidence ≥ {st.session_state.confidence_threshold}%</div>"
                f"<div style='font-size:13px;color:#667085;line-height:1.55'>Predictions below the configured threshold are marked for human review instead of being treated as automatic decisions.</div>"
                f"<div style='margin-top:14px'><span class='status-pill'>AUDIT ENABLED</span></div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        st.subheader("Business impact")
        
        if st.session_state.last_predictions is not None and "Decision Status" in st.session_state.last_predictions:
            pred = st.session_state.last_predictions
            total_predictions = len(pred)
            auto_decisions = int((pred["Decision Status"] == "Auto Decision").sum())
            human_reviews = int((pred["Decision Status"] == "Human Review").sum())
            
            b1, b2, b3, b4 = st.columns(4)
            
            auto_rate = (auto_decisions / total_predictions * 100) if total_predictions > 0 else 0
            b1.markdown(card("Automation rate", f"{auto_rate:.1f}%", "Decisions made without human intervention"), unsafe_allow_html=True)
            
            estimated_hours_saved = human_reviews * 0.25
            b2.markdown(card("Hours saved", f"{estimated_hours_saved:.1f}h", "Est. human review time avoided (15min per case)"), unsafe_allow_html=True)
            
            estimated_cost_saved = estimated_hours_saved * 50
            b3.markdown(card("Cost savings", f"${estimated_cost_saved:,.0f}", "Est. labor cost saved ($50/hour)"), unsafe_allow_html=True)
            
            model_accuracy = st.session_state.best_score * 100
            b4.markdown(card("Model accuracy", f"{model_accuracy:.1f}%", "Validation score on training data", progress_percent=model_accuracy), unsafe_allow_html=True)
            
            st.markdown("<div class='section'></div>", unsafe_allow_html=True)
            impact_left, impact_right = st.columns([1, 1])
            with impact_left:
                st.markdown("""
                <div class='report-card'>
                    <div style='font-size:13px;font-weight:600;color:#111827;margin-bottom:8px'>Key benefits</div>
                    <div style='font-size:12px;color:#667085;line-height:1.8'>
                    • <b>Reduced manual review:</b> Only low-confidence predictions require human oversight<br>
                    • <b>Faster decisions:</b> High-confidence cases are automated instantly<br>
                    • <b>Cost efficiency:</b> Labor costs scale with complexity, not volume<br>
                    • <b>Audit trail:</b> All decisions are logged for compliance and analysis
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with impact_right:
                st.markdown("""
                <div class='report-card'>
                    <div style='font-size:13px;font-weight:600;color:#111827;margin-bottom:8px'>ROI factors</div>
                    <div style='font-size:12px;color:#667085;line-height:1.8'>
                    • <b>Model performance:</b> Higher accuracy = more automation<br>
                    • <b>Threshold tuning:</b> Balance automation vs. risk tolerance<br>
                    • <b>Volume scaling:</b> Savings grow with prediction volume<br>
                    • <b>Human-in-the-loop:</b> Maintains safety while reducing effort
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Run inference to populate business impact metrics.")

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Decision activity")
            logs = pd.DataFrame(st.session_state.audit_log)
            if logs.empty:
                st.info("No governance events recorded yet. Run inference or update a setting to create the first event.")
            else:
                st.dataframe(logs, use_container_width=True, hide_index=True)
        
        with right:
            st.subheader("Decision timeline")
            if st.session_state.last_predictions is not None and "Decision Status" in st.session_state.last_predictions:
                pred = st.session_state.last_predictions
                decision_counts = pred["Decision Status"].value_counts().reset_index()
                decision_counts.columns = ["Decision Type", "Count"]
                
                fig = px.pie(decision_counts, values="Count", names="Decision Type", hole=0.6, 
                            color_discrete_map={"Auto Decision": "#059669", "Human Review": "#d97706"})
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=0), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Distribution of auto vs. human-reviewed decisions")
            else:
                st.info("Run inference to populate decision timeline visualization.")

        st.markdown("<div class='section'></div>", unsafe_allow_html=True)
        b1,b2 = st.columns(2)
        with b1:
            st.download_button("Export governance report", make_pdf_report(), "predictive_os_governance.pdf", "application/pdf", type="primary", use_container_width=True)
        with b2:
            if st.session_state.last_predictions is not None:
                st.download_button("Export latest decisions", st.session_state.last_predictions.to_csv(index=False), "predictive_os_decisions.csv", "text/csv", use_container_width=True)
            else:
                st.button("Export latest decisions", disabled=True, use_container_width=True)

# -----------------------------
# Settings
# -----------------------------
else:
    st.title("System Settings")
    st.caption("Tune governance behavior without changing the trained model.")
    
    st.markdown("""
    <div class='report-card'>
        <div style='font-size:13px;font-weight:600;color:#111827;margin-bottom:6px'>Settings Guide</div>
        <div style='font-size:12px;color:#667085;line-height:1.6'>
        <b>Confidence Threshold:</b> Controls when predictions require human review (50-99%)<br>
        <b>Drift Tolerance:</b> Sensitivity to data distribution changes (0.0-1.0)<br>
        <b>Integrations:</b> Configure alerts and webhooks for monitoring
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section'></div>", unsafe_allow_html=True)
    a,b = st.columns(2)
    with a:
        st.subheader("Decision governance")
        threshold = st.slider("Human-review confidence threshold", 50, 99, st.session_state.confidence_threshold, help="Lower = more conservative (more human reviews). Higher = more automation.")
        drift = st.slider("Data-drift tolerance", 0.0, 1.0, st.session_state.drift_tolerance, 0.01, help="Sensitivity to distribution changes in incoming data")
        st.info("Predictions below the confidence threshold are routed to human review. This is the Predictive OS safety gate.")
    with b:
        st.subheader("Integrations")
        slack = st.toggle("Slack alerts", st.session_state.slack_notifications, help="Receive alerts when predictions require human review")
        email = st.toggle("Email audit summaries", st.session_state.email_summaries, help="Get periodic summaries of decision activity")
        webhook = st.text_input("Webhook URL", st.session_state.webhook_url, help="POST decision events to your custom endpoint")
    if st.button("Save configuration", type="primary"):
        st.session_state.update({"confidence_threshold": threshold, "drift_tolerance": drift, "slack_notifications": slack, "email_summaries": email, "webhook_url": webhook})
        add_audit("SETTINGS_UPDATED", f"Confidence threshold: {threshold}%, Drift tolerance: {drift}")
        st.success("Configuration saved successfully.")
