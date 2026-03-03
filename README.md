# Spiral of Silence - IRL Dashboard

Interactive Streamlit dashboard for visualizing Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) applied to spiral of silence agent behavior.

## Features

- **Trajectory Extraction**: Load and process agent behavior CSV files
- **Transition Matrix Visualization**: Explore state-action-state transitions
- **MaxEnt IRL**: Learn reward functions from expert demonstrations
- **Interactive Visualizations**:
  - Value Iteration convergence process with step-by-step Q-values
  - Policy evolution across epochs
  - Occupancy rollout simulation
  - Gradient convergence tracking
  - Final learned reward and policy heatmaps

## Deployment

### Running Locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

### Deploy on Render

1. Fork/clone this repository
2. Create a new Web Service on [Render](https://render.com)
3. Connect your GitHub repository
4. Configure the service:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0`

## Data Files

The `results/` directory contains sample CSV files with agent trajectory data from spiral of silence simulations.

## Algorithm

Implements MaxEnt IRL (Ziebart et al., 2008) with:
- State space: 5 opposition levels (Complete Agreement → Large Disagreement)
- Action space: 3 willingness changes (-1, 0, +1)
- Features: State-action one-hot encoding (15 features)
- Value iteration from Q=0 with interactive convergence visualization
