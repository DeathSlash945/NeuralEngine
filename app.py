import streamlit as st
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import umap
from model import NeuralPersonalityRVAE

st.set_page_config(page_title="NeuralEngine", layout="wide")
st.title("NeuralEngine: Multi-trajectory mapping dashboard")
st.markdown("---")

# Load trained model checkpoint
@st.cache_resource
def load_engine():
    model = NeuralPersonalityRVAE(latent_dim=16)
    try:
        model.load_state_dict(torch.load("data/personality_engine.pth", map_location=torch.device('cpu')))
    except Exception:
        pass
    model.eval()
    return model
model = load_engine()

@st.cache_resource
def generate_global_manifold():
    st.info("Establishing stable latent manifold topological map...")
    
    # Generate two polar opposite mock memory tracks to anchor 2D coordinates
    nurtured_env = np.tile(np.array([0.6, 0.1, 0.5, 0.0]), (100, 1))
    traumatized_env = np.tile(np.array([-0.6, 0.9, -0.5, 0.9]), (100, 1))
    
    # Process through model to extract pristine multi-dimensional latent vectors
    with torch.no_grad():
        _, _, _, _, z_nurture = model(torch.tensor([nurtured_env], dtype=torch.float32))
        _, _, _, _, z_trauma = model(torch.tensor([traumatized_env], dtype=torch.float32))
        
    z_background = torch.cat([z_nurture[0], z_trauma[0]], dim=0).numpy()
    
    # Fit global mapper with highly stable hyperparameter configurations
    mapper = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=42)
    mapper.fit(z_background)
    # Pre-project anchor points for background context mapping
    coords_2d = mapper.transform(z_background)
    return mapper, coords_2d[:100], coords_2d[100:]

umap_projector, background_nurture, background_trauma = generate_global_manifold()

# sidebar config
col_sidebar, col_display = st.columns([1, 2])

with col_sidebar:
    st.header("Base conditions")
    init_social = st.slider("Initial Social Valence", -1.0, 1.0, 0.4)
    init_stress = st.slider("Initial System Stress", 0.0, 1.0, 0.2)
    init_reward = st.slider("Initial Reward Density", -1.0, 1.0, 0.3)
    init_malice = st.slider("Initial Environmental Malice", 0.0, 1.0, 0.1)
    
    st.markdown("---")
    st.header("External trauma injection (mwehehehehe)")
    enable_shock = st.checkbox("Inject mid-run trauma event", value=True)
    shock_step = st.slider("Trauma onset step (t)", 10, 80, 25)
    shock_duration = st.slider("Trauma sustenance duration", 5, 35, 15)
    
    st.markdown("---")
    st.header("Global settings")
    total_steps = st.slider("Simulation horizon cycles", 40, 120, 80, step=10)
    st.markdown("---")
    st.header("Playback control")
    playback_cutoff = st.slider("Visualize up to step", 1, total_steps, total_steps)
    run_simulation = st.button("Execute run", use_container_width=True)

# Simulation Execution Loop
if run_simulation or "cached_run" in st.session_state:
    if run_simulation or "cached_run" not in st.session_state:
        current_env = np.array([init_social, init_stress, init_reward, init_malice])
        interaction_history = [current_env]
        trait_history, behavioral_axes_history, latent_history = [], [], []
        shock_active_mask = []
        
        for t in range(total_steps):
            is_shock_step = enable_shock and (shock_step <= t < shock_step + shock_duration)
            shock_active_mask.append(is_shock_step)
            
            input_tensor = torch.tensor([interaction_history], dtype=torch.float32)
            with torch.no_grad():
                pred_traits, action_logits, _, _, z = model(input_tensor)
                
            latest_traits = pred_traits[0, -1].numpy()
            latest_logits = action_logits[0, -1].numpy()
            latest_latent = z[0, -1].numpy()
            
            scaling_factor = 0.1
            dom_axis = np.tanh((latest_logits[2] - latest_logits[0]) * scaling_factor)
            aff_axis = np.tanh((latest_logits[1] - latest_logits[0]) * scaling_factor)
            
            trait_history.append(latest_traits)
            behavioral_axes_history.append([dom_axis, aff_axis])
            latent_history.append(latest_latent)
            
            if is_shock_step:
                next_social = np.random.uniform(-1.0, -0.7)
                next_stress = np.random.uniform(0.8, 1.0)
                next_reward = np.random.uniform(-0.9, -0.5)
                next_malice = np.random.uniform(0.8, 1.0)
            else:
                next_social = np.clip(0.4 * aff_axis + (0.5 * init_social) + np.random.uniform(-0.1, 0.1), -1.0, 1.0)
                next_stress = np.clip(0.3 * (dom_axis * -aff_axis) + (0.7 * init_stress) + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
                next_reward = np.clip(0.3 * dom_axis + (0.5 * init_reward) + np.random.uniform(-0.1, 0.1), -1.0, 1.0)
                next_malice = np.clip(0.5 * (-aff_axis if aff_axis < 0 else 0.0) + (0.5 * init_malice), 0.0, 1.0)
            
            next_env = np.array([next_social, next_stress, next_reward, next_malice])
            interaction_history.append(next_env)
            current_env = next_env
            
        # Cache full simulation run array state in memory to preserve values during slider adjustments
        st.session_state.cached_run = {
            "traits": np.array(trait_history),
            "behavior": np.array(behavioral_axes_history),
            "latent": np.array(latent_history),
            "shock_mask": shock_active_mask
        }

    # Extract current state based on playback controls
    traits_all = st.session_state.cached_run["traits"][:playback_cutoff]
    behavior_all = st.session_state.cached_run["behavior"][:playback_cutoff]
    latent_all = st.session_state.cached_run["latent"][:playback_cutoff]
    shock_mask_all = st.session_state.cached_run["shock_mask"][:playback_cutoff]

    live_agent_2d = umap_projector.transform(latent_all)

    # Charts display
    with col_display:
        st.subheader(f"Active simulation state at Step {playback_cutoff}")
        final_traits = traits_all[-1]
        t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
        t_col1.metric("Trust", f"{final_traits[0]:.2f}")
        t_col2.metric("Confidence", f"{final_traits[1]:.2f}")
        t_col3.metric("Aggression", f"{final_traits[2]:.2f}")
        t_col4.metric("Curiosity", f"{final_traits[3]:.2f}")
        t_col5.metric("Optimism", f"{final_traits[4]:.2f}")
        
        st.markdown("---")
        
        fig = make_subplots(
            rows=3, cols=1,
            vertical_spacing=0.08,
            subplot_titles=(
                "Continuous trait vectors dynamics profile", 
                "Continuous interpersonal circulatory Manifestations",
                "UMAP latent manifold displacement trajectory (basically R^16 -> R^2)"
            )
        )
        
        #Structural timelines
        traits_labels = ["Trust", "Confidence", "Aggression", "Curiosity", "Optimism"]
        for i, label in enumerate(traits_labels):
            fig.add_trace(go.Scatter(y=traits_all[:, i], name=label, mode='lines', line=dict(width=2.5)), row=1, col=1)
            
        fig.add_trace(go.Scatter(y=behavior_all[:, 0], name="Agency/Dominance", mode='lines+markers', line=dict(dash='dash')), row=2, col=1)
        fig.add_trace(go.Scatter(y=behavior_all[:, 1], name="Communion/Affiliation", mode='lines+markers', line=dict(dash='dot')), row=2, col=1)
        
        # bg context map and real time context map
        fig.add_trace(go.Scatter(x=background_nurture[:, 0], y=background_nurture[:, 1], mode='markers', name="Nurtured reference territory", marker=dict(color='rgba(100, 220, 140, 0.15)', size=5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=background_trauma[:, 0], y=background_trauma[:, 1], mode='markers', name="Trauma reference territory", marker=dict(color='rgba(240, 100, 100, 0.15)', size=5)), row=3, col=1)
        
        # Overlaid current dynamic path
        fig.add_trace(go.Scatter(
            x=live_agent_2d[:, 0], y=live_agent_2d[:, 1],
            mode='lines+markers',
            name="Current path trajectory",
            marker=dict(size=8, color=list(range(len(live_agent_2d))), colorscale='Electric', showscale=True, colorbar=dict(title="Time Frame", x=1.02, y=0.15, len=0.3))
        ), row=3, col=1)
        
        # Render horizontal/vertical threshold marker indicator highlights
        shock_indices = [i for i, x in enumerate(shock_mask_all) if x]
        if len(shock_indices) > 0:
            for r in [1, 2]:
                fig.add_vline(x=shock_indices[0], line_width=2, line_dash="dash", line_color="red", row=r, col=1)
                if shock_indices[-1] < len(shock_mask_all) - 1:
                    fig.add_vline(x=shock_indices[-1], line_width=2, line_dash="dash", line_color="orange", row=r, col=1)
                    
        fig.update_layout(height=1000, margin=dict(l=20, r=20, t=40, b=20))
        fig.update_yaxes(range=[0, 1], row=1, col=1)
        fig.update_yaxes(range=[-1, 1], row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
else:
    with col_display:
        st.info("Configure variables and execute run to project stable structural paths.")