import streamlit as st
import pandas as pd
import os
import glob
import uuid
from langgraph.types import Command
from graph_builder import agent_graph  # Import your compiled graph

st.set_page_config(page_title="Agentic Data Science Pipeline", layout="wide")
st.title("🤖 Autonomous Data Science Agent")

# 1. Session State Initialization
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"streamlit_run_{uuid.uuid4().hex[:6]}"
if "pipeline_started" not in st.session_state:
    st.session_state.pipeline_started = False
if "awaiting_input" not in st.session_state:
    st.session_state.awaiting_input = False
if "interrupt_data" not in st.session_state:
    st.session_state.interrupt_data = None

config = {"configurable": {"thread_id": st.session_state.thread_id}}

# 2. Sidebar: File Upload & Dynamic Target Selection
with st.sidebar:
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV", type="csv")
    
    if uploaded_file is not None:
        # Save the uploaded file locally so pandas and REPL can access it
        os.makedirs("../data", exist_ok=True)
        file_path = f"../data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Read columns dynamically so the user can choose the target variable
        df_preview = pd.read_csv(file_path, nrows=5)
        st.subheader("2. Configure Target")
        target_col = st.selectbox(
            "Select Target Variable:", 
            options=df_preview.columns.tolist(),
            index=len(df_preview.columns) - 1  # Defaults to the last column
        )
        
        if st.button("🚀 Start Pipeline"):
            # Generate a fresh thread_id and clear old plots for a clean run
            st.session_state.thread_id = f"streamlit_run_{uuid.uuid4().hex[:6]}"
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            os.makedirs("../plots", exist_ok=True)
            for old_plot in glob.glob("../plots/*.png"):
                os.remove(old_plot)
                
            initial_state = {
                "dataset_path": file_path, 
                "target_column": target_col
            }
            
            with st.spinner(f"Running EDA on {uploaded_file.name} (Target: {target_col})..."):
                for _ in agent_graph.stream(initial_state, config=config):
                    pass
                
                st.session_state.pipeline_started = True
                snapshot = agent_graph.get_state(config)
                if snapshot.next and snapshot.tasks and snapshot.tasks[0].interrupts:
                    st.session_state.awaiting_input = True
                    st.session_state.interrupt_data = snapshot.tasks[0].interrupts[0].value
                else:
                    st.session_state.awaiting_input = False
                st.rerun()

# 3. Main Stage: Persistent EDA, Live Plots & Model Results
if st.session_state.pipeline_started:
    # Fetch the full accumulated state from the checkpointer
    snapshot = agent_graph.get_state(config)
    full_state = snapshot.values if snapshot else {}
    
    # 3a. EDA Summary & Distribution Plots
    if full_state.get("dataset_summary"):
        with st.expander("📊 Exploratory Data Analysis (EDA) Summary", expanded=True):
            st.code(full_state["dataset_summary"], language="text")
            
        plot_files = sorted(glob.glob("../plots/*_eda_distribution.png"))
        if plot_files:
            st.subheader("📈 Feature Distributions")
            cols = st.columns(3)
            for idx, plot_path in enumerate(plot_files):
                with cols[idx % 3]:
                    col_name = os.path.basename(plot_path).replace("_eda_distribution.png", "")
                    st.image(plot_path, caption=f"Distribution: {col_name}", use_container_width=True)

    # 3b. Active Pipeline Decisions Tracker
    decisions = {
        "Target Column": full_state.get("target_column"),
        "Imputation": full_state.get("imputation_strategy"),
        "Encoding": full_state.get("encoding_strategy"),
        "Scaling": full_state.get("scaling_strategy"),
        "Model": full_state.get("model_type"),
        "Tuning": full_state.get("tuning_strategy")
    }
    active_decisions = {k: v for k, v in decisions.items() if v is not None}
    if active_decisions:
        st.info(" | ".join([f"**{k}:** `{v}`" for k, v in active_decisions.items()]))

    # 3c. Final Model Training Results
    if full_state.get("model_metrics"):
        st.divider()
        st.subheader("🎯 Model Training Results")
        st.code(full_state["model_metrics"], language="text")
        
        saved_path = full_state.get("saved_model_path")
        if saved_path:
            st.success(f"Model saved to: {saved_path}")
            if os.path.exists(saved_path):
                with open(saved_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Trained Model Artifact",
                        data=f,
                        file_name=os.path.basename(saved_path),
                        mime="application/octet-stream"
                    )

# 4. Human-In-The-Loop Control Panel
if st.session_state.awaiting_input and st.session_state.interrupt_data:
    st.divider()
    st.subheader("⚠️ Human Intervention Required")
    
    question = st.session_state.interrupt_data.get("question", "Make a selection:")
    options = st.session_state.interrupt_data.get("options", [])
    
    user_choice = st.radio(question, options, key=f"radio_{question}")
    
    if st.button("Submit Decision", type="primary"):
        with st.spinner(f"Applying '{user_choice}'..."):
            st.session_state.awaiting_input = False
            
            # Resume graph with the user's choice
            for _ in agent_graph.stream(Command(resume=user_choice), config=config):
                pass
                
            # Check if the graph paused at the next HITL node
            snapshot = agent_graph.get_state(config)
            if snapshot.next and snapshot.tasks and snapshot.tasks[0].interrupts:
                st.session_state.awaiting_input = True
                st.session_state.interrupt_data = snapshot.tasks[0].interrupts[0].value
            else:
                st.session_state.awaiting_input = False
                st.session_state.interrupt_data = None
            
            st.rerun()