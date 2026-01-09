import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from src.agent_logic import MentalHealthAgent

st.set_page_config(page_title="Geometric Emotion Drift Agent", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Agent for Geometric Emotion Drift Detection")
st.markdown("### Using DistilBERT & Riemannian Gradient Flow (RGF)")

# Initialize Agent in Session State
if 'agent' not in st.session_state:
    with st.spinner("Initializing AI Model (this may take a moment)..."):
        st.session_state.agent = MentalHealthAgent()
    st.success("System Initialized.")

agent = st.session_state.agent

# Sidebar
st.sidebar.header("Controls")
if st.sidebar.button("Reset Agent Memory"):
    st.session_state.agent = MentalHealthAgent()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("**Theory**")
st.sidebar.info(
    "**Drift**: Cosine distance between consecutive emotion states.\n\n"
    "**RGF**: Riemannian Gradient Flow models the 'force' required to shift emotional states."
)

# Main Input
user_input = st.text_area("How are you feeling right now?", height=100, placeholder="Type here (e.g., 'I feel anxious about the deadline but happy I made progress')")

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing Cognitive State..."):
            record = agent.process_input(user_input)
        st.experimental_rerun() if hasattr(st, 'experimental_rerun') else st.rerun()

# Display Results if history exists
if agent.history:
    latest = agent.history[-1]
    
    # Dashboard Layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Emotional Profile")
        # Bar Chart of Probabilities
        emotions = latest['emotions']
        df_emotions = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Probability'])
        fig_bar = px.bar(df_emotions, x='Emotion', y='Probability', color='Probability', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Feedback Box
        status_color = "green" if latest['status'] == "Stable" else ("orange" if latest['status'] == "Volatile" else "red")
        st.markdown(f"#### Status: :{status_color}[{latest['status']}]")
        if latest['status'] == "Stable":
            st.success(latest['feedback'])
        elif latest['status'] == "Volatile":
            st.warning(latest['feedback'])
        else:
            st.error(latest['feedback'])

    with col2:
        st.subheader("Geometric Trajectory (PCA)")
        # Perform PCA on all historical vectors + current trajectory
        all_vectors = [h['vector'] for h in agent.history]
        current_traj = latest['trajectory']
        
        # Combine for PCA fit
        combined_data = np.vstack(all_vectors + [current_traj])
        
        # Need at least 2 points and 2 dimensions
        if combined_data.shape[0] > 1:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(combined_data)
            
            # Split back
            n_history = len(all_vectors)
            history_coords = coords[:n_history]
            traj_coords = coords[n_history:]
            
            fig_traj = go.Figure()
            
            # Plot History Points
            fig_traj.add_trace(go.Scatter(
                x=history_coords[:,0], y=history_coords[:,1],
                mode='markers+lines', name='Interaction History',
                marker=dict(size=10, color='gray'),
                line=dict(color='gray', dash='dot')
            ))
            
            # Plot Current RGF Trajectory
            fig_traj.add_trace(go.Scatter(
                x=traj_coords[:,0], y=traj_coords[:,1],
                mode='lines', name='Cognitive Flow (RGF)',
                line=dict(color='red', width=3)
            ))
            
            # Start/End of trajectory
            fig_traj.add_trace(go.Scatter(
                x=[traj_coords[0,0]], y=[traj_coords[0,1]],
                mode='markers', name='Stimulus Start', marker=dict(size=8, color='blue')
            ))
            fig_traj.add_trace(go.Scatter(
                x=[traj_coords[-1,0]], y=[traj_coords[-1,1]],
                mode='markers', name='Settled State', marker=dict(size=12, color='red', symbol='star')
            ))

            fig_traj.update_layout(title="2D Cognitive Manifold Projection", xaxis_title="PC1", yaxis_title="PC2")
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("Not enough data for Geometric View yet.")

    # Drift Timeline
    st.subheader("Emotional Drift Over Time")
    drift_data = [h['drift'] for h in agent.history]
    fig_drift = px.line(x=range(len(drift_data)), y=drift_data, markers=True, 
                        labels={'x': 'Interaction Step', 'y': 'Drift Score (Cosine Distance)'})
    # Add threshold lines
    fig_drift.add_hline(y=agent.drift_threshold_warning, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig_drift.add_hline(y=agent.drift_threshold_alert, line_dash="dash", line_color="red", annotation_text="Critical")
    st.plotly_chart(fig_drift, use_container_width=True)

    # Log Data
    with st.expander("Analysis Log"):
        st.dataframe(agent.get_history_dataframe())

else:
    st.info("Start by entering your thoughts above.")

