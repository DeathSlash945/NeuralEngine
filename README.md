# Neural state-space engine: autonomous closed-loop behavioral simulator

An end-to-end deep learning and simulation framework that models the continuous behavioral drift of an autonomous agent inside a dynamic, responsive environment. 

By pairing a **Recurrent Variational Autoencoder (R-VAE)** with a closed-loop environmental feedback engine, this architecture maps historical environmental interactions into a continuous, 16-dimensional Gaussian latent manifold, projecting real-time "internal states" and actions onto psychological behavioral frameworks.

---

## Key Architecture Features

* **Variational latent manifold:** Implements an R-VAE using the reparameterization trick ($z = \mu + \sigma \odot \epsilon$) to compress temporal interaction history into smooth, continuous hidden distributions instead of rigid, deterministic RNN states.
* **Closed-Loop feedback dynamics:** Actions chosen by the agent alter the incoming environmental variables (Social Valence, System Stress, Reward Density, Threat/Malice) at step $t+1$, which in turn updates the agent's internal state in an unbroken simulation loop.
* **Continuous behavioral manifestations:** Maps unnormalized model logits to the continuous coordinate planes of the **Interpersonal Circumplex Model** along the axes of **Agency/Dominance** and **Communion/Affiliation** using temperature-scaled hyperbolic tangent ($\tanh$) functions.
* **Exogenous trauma injector:** Simulates unforeseen catastrophic ambient shifts (unprovoked environmental collapse or localized shock) to evaluate structural state-space resilience and adaptation mid-trajectory.
* **Stable UMAP topological projections:** Solves the visual chaos of live low-context dimensionality reduction by pre-fitting a global **Uniform Manifold Approximation and Projection (UMAP)** mapping engine over polar archetypal memory anchors, letting the active simulation path track smoothly over a fixed landscape.

---

## Tech Stack & Dependencies

* **Core deep learning:** `PyTorch` (Sequential VAE Architecture & Multi-Task Loss optimization)
* **Manifold projections:** `umap-learn` (Topological Dimensionality Reduction)
* **Dashboard engine:** `Streamlit` (Interactive state controls, sidebar injectors, and runtime loops)
* **Data visualization:** `Plotly` (Real-time subplots, continuous path tracking, and timeline shading)
* **Numerical processing:** `NumPy`
  
How to simulate and test random stuff:
* Run the simulation in streamlit using the command: 'streamlit run app.py' in the project folder
