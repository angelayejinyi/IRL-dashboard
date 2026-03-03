from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from irl_maxent import maxent_irl_onehot

# Set random seed for reproducibility
np.random.seed(42)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
N_STATES = 5
N_ACTIONS = 3


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_stats_table(stats: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(stats.keys()), "value": list(stats.values())})


OPINION_MAP = {
    "strongly against": -2,
    "slightly against": -1,
    "neutral towards": 0,
    "slightly support": 1,
    "strongly support": 2,
}


def map_opinion(v):
    if pd.isna(v):
        return None
    if isinstance(v, str):
        v_lower = v.lower()
        for key, value in OPINION_MAP.items():
            if key in v_lower:
                return value
        return None
    return int(v)


def calculate_perceived_majority(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["op_numeric"] = df["Opinion"].apply(map_opinion)
    df = df.dropna(subset=["op_numeric"])

    def get_majority_for_iteration(group: pd.DataFrame):
        if len(group) == 0:
            return np.nan

        opinions = group["op_numeric"].values
        willingness = group["Willingness to Speak"].values

        mask = ~(np.isnan(opinions) | np.isnan(willingness))
        opinions = opinions[mask]
        willingness = willingness[mask]

        if len(opinions) == 0:
            return np.nan

        weighted_avg = np.average(opinions, weights=willingness)
        possible_values = np.array([-2, -1, 0, 1, 2])
        quantized = possible_values[np.argmin(np.abs(possible_values - weighted_avg))]
        return quantized

    majority_dict = {}
    for iteration, group in df.groupby("Iteration"):
        majority_dict[iteration] = get_majority_for_iteration(group)

    df["Perceived Majority Opinion"] = df["Iteration"].map(majority_dict)
    return df


def build_trajectories_with_distance(df: pd.DataFrame, max_timestep: int | None = None):
    df = df.copy()

    if max_timestep is not None and "Iteration" in df.columns:
        df = df[df["Iteration"] <= max_timestep]

    df = df.drop_duplicates(subset=["Agent Name", "Iteration"], keep="first")
    
    # Ensure all (Agent Name, Iteration) combinations exist in the selected range
    if "Iteration" in df.columns and "Agent Name" in df.columns and len(df) > 0:
        df = df.sort_values(["Agent Name", "Iteration"]).reset_index(drop=True)
        
        unique_agents = sorted(df["Agent Name"].unique())
        min_iter = int(df["Iteration"].min())
        max_iter = int(df["Iteration"].max())
        
        new_rows = []
        
        for agent in unique_agents:
            agent_df = df[df["Agent Name"] == agent].set_index("Iteration").sort_index()
            agent_iters = sorted(agent_df.index)
            
            if len(agent_iters) == 0:
                continue
            
            # Fill from agent's first iteration to global max iteration
            first_agent_iter = agent_iters[0]
            agent_full_idx = pd.Index(range(first_agent_iter, max_iter + 1))
            
            # Reindex and forward fill all columns
            agent_full = agent_df.reindex(agent_full_idx)
            agent_full = agent_full.fillna(method="ffill")
            
            # For newly added iterations, set Willingness to Speak to 0
            original_iters = set(agent_df.index)
            for iteration in agent_full_idx:
                if iteration not in original_iters:
                    agent_full.loc[iteration, "Willingness to Speak"] = 0
            
            agent_full["Agent Name"] = agent
            agent_full["Iteration"] = agent_full.index
            new_rows.append(agent_full.reset_index(drop=True))
        
        if new_rows:
            df = pd.concat(new_rows, ignore_index=True)
    
    df["op_numeric"] = df["Opinion"].apply(map_opinion)
    df = df.dropna(subset=["op_numeric", "Perceived Majority Opinion"])
    
    def compute_opposition_state(agent_op, majority_op):
        # 0: complete agreement (distance=0)
        if agent_op == majority_op:
            return 0

        same_sign = (agent_op > 0 and majority_op > 0) or (agent_op < 0 and majority_op < 0)
        opposite_sign = (agent_op > 0 and majority_op < 0) or (agent_op < 0 and majority_op > 0)
        abs_agent = abs(agent_op)
        abs_majority = abs(majority_op)

        # 1: mostly agreement (slightly +/- vs strongly +/-)
        if same_sign and {abs_agent, abs_majority} == {1, 2}:
            return 1

        # 2: small disagreement (neutral vs slightly +/-)
        if (abs_agent == 0 and abs_majority == 1) or (abs_agent == 1 and abs_majority == 0):
            return 2

        # 3: medium disagreement
        #    - slightly +/- vs slightly -/+ OR neutral vs strongly +/-
        if (opposite_sign and abs_agent == 1 and abs_majority == 1) or (
            (abs_agent == 0 and abs_majority == 2) or (abs_agent == 2 and abs_majority == 0)
        ):
            return 3

        # 4: large disagreement
        #    - strongly +/- vs strongly -/+ OR strongly +/- vs slightly -/+
        if opposite_sign and (abs_agent == 2 or abs_majority == 2):
            return 4

        # Fallback (should be rare)
        return 3
    
    df["state"] = df.apply(
        lambda row: compute_opposition_state(row["op_numeric"], row["Perceived Majority Opinion"]),
        axis=1
    )
    df = df.sort_values(["Agent Name", "Iteration"])

    def map_action_from_reason(reason):
        if pd.isna(reason):
            return None

        text = str(reason).lower().strip()

        if "feel more encouraged" in text or "increase" in text:
            return 1
        if "feel less encouraged" in text or "decrease" in text:
            return -1
        if "neutral" in text or "don't feel more or less" in text or "do not feel more or less" in text:
            return 0

        return None

    df["action"] = df["Willingness Reason"].apply(map_action_from_reason)

    trajectories: list[list[tuple[int, int]]] = []
    for _, g in df.groupby("Agent Name"):
        g2 = g.dropna(subset=["state", "action"])
        traj = list(zip(g2["state"].astype(int), g2["action"].astype(int)))
        if len(traj) >= 3:
            trajectories.append(traj)

    return trajectories, df


def feature_matrix_sa_onehot(nS, nA):
    """
    Create one-hot feature matrix for state-action pairs.
    Shape: (nS*nA, nS*nA)
    """
    F = np.eye(nS * nA)
    return F


def estimate_transition(trajs, nS=N_STATES, nA=N_ACTIONS):
    """
    Estimate transition probability matrix P from trajectories.
    Uses empirical counting with trajectory averaging (similar to find_feature_expectations logic).
    
    Args:
        trajs: List of trajectories (each is list of (state, action) tuples)
        nS: Number of states
        nA: Number of actions
    
    Returns: 
        P: Transition matrix of shape (nS, nA, nS)
           P[s, a, s'] = P(s' | s, a) - conditional probability
        counts: Raw count matrix for diagnostics
    """
    counts = np.zeros((nS, nA, nS))
    
    # Count state-action-next_state transitions from all trajectories
    for traj in trajs:
        for i in range(len(traj) - 1):
            s, a = traj[i]
            s_next, _ = traj[i + 1]
            
            # Remap action: -1→0, 0→1, 1→2
            a_idx = int(a) + 1
            s = int(s)
            s_next = int(s_next)
            
            if 0 <= s < nS and 0 <= a_idx < nA and 0 <= s_next < nS:
                counts[s, a_idx, s_next] += 1
    
    # Save raw counts for diagnostics
    raw_counts = counts.copy()
    
    # Laplace smoothing for numerical stability
    counts += 1e-6
    
    # Normalize: For each (s,a) pair, compute conditional probability over next states
    # P[s, a, s'] = count[s, a, s'] / sum_{s'} count[s, a, s']
    P = counts / counts.sum(axis=2, keepdims=True)
    
    return P, raw_counts

def render_irl_pipeline():
    """Render MaxEnt IRL pipeline with formulas."""
    st.markdown("---")
    st.subheader("📋 MaxEnt IRL: Pipeline & Formulas")
    
    with st.expander("🔍 Algorithm Steps & Formulas", expanded=False):
        st.markdown("""
**MaxEnt IRL Process:**

**Step 1-3: Preprocessing**
- Expert trajectories → State-action sequences
- Empirical P(s'|s,a) from transition counts
- Expert occupancy μ_E from demonstrations

**Step 4-10: Iterative Learning (each epoch)**
1. Init: α ~ Uniform(0, 1)
2. Reward: r = F @ α
3. **Value Iteration:** Start Q=0, iterate to convergence
   - V(s) = max_a Q(s,a)
   - Q(s,a) = r(s,a) + γ Σ P[s,a,s'] V(s')
4. Policy: π = softmax(Q)
5. Occupancy: μ_π from policy rollout
6. Gradient: ∇ = μ_E - μ_π
7. Update: α := α + λ∇
        """)
    st.markdown("**Key Formulas:**")
    st.markdown("Transitions")
    st.latex(r"P(s' \mid s,a) = \frac{\text{count}(s,a,s') + \epsilon}{\sum_{\bar{s}} \text{count}(s,a,\bar{s}) + \epsilon}")
    st.markdown("Value Iteration Update")
    st.latex(r"Q^{k+1}(s,a) = r(s,a) + \gamma \sum_{s'} P(s,a,s') V^k(s')")
    st.markdown("Policy")
    st.latex(r"\pi(a \mid s) = \frac{e^{Q(s,a)}}{\sum_{\bar{a}} e^{Q(s,\bar{a})}}")
    st.markdown("IRL Update")
    st.latex(r"\alpha \leftarrow \alpha + \lambda (\mu_E - \mu_{\pi})")


def render_vi_history_viewer(P, r, nS, nA, state_names, action_names, gamma=0.9):
    """Show Q-value convergence process."""
    try:
        from irl_maxent import value_iteration_Q
        Q_final, V_final, vi_hist = value_iteration_Q(P, r, gamma=gamma, threshold=1e-4, history=True)
        
        st.caption(f"Value Iteration converged in {len(vi_hist)-1} iterations")
        
        # VI step slider
        vi_iter = st.slider(
            "Value Iteration Step",
            0, len(vi_hist) - 1,
            len(vi_hist) - 1,
            key=f"vi_{hash(tuple(r.flatten()))}"
        )
        
        Q_t = vi_hist[vi_iter]
        diff = Q_t - Q_final
        inf_norm = float(np.max(np.abs(diff)))
        l2_norm = float(np.linalg.norm(diff))
        progress_pct = 100.0 * (vi_iter / max(1, len(vi_hist) - 1))

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("||Q_t - Q*||∞", f"{inf_norm:.6f}")
        with m2:
            st.metric("||Q_t - Q*||₂", f"{l2_norm:.6f}")
        with m3:
            st.metric("VI Progress", f"{progress_pct:.1f}%")

        Q_df = pd.DataFrame(
            Q_t, 
            index=[f"s={i}" for i in range(nS)],
            columns=[a[:5] for a in action_names]
        )
        st.dataframe(Q_df.style.format("{:.5f}").background_gradient(cmap="RdYlGn"), use_container_width=False)
        
        # V values for each state
        V_t = Q_t.max(axis=1)
        cols = st.columns(nS)
        for i, col in enumerate(cols):
            with col:
                st.metric(f"V(s{i})", f"{V_t[i]:.4f}")
                
    except Exception as e:
        st.error(f"VI viewer error: {e}")


def main() -> None:
    st.set_page_config(page_title="IRL Demo", layout="wide")
    
    # Create centered container with max width 1200
    _, center, _ = st.columns([1, 5, 1])
    
    with center:
        st.title("Spiral of Silence IRL Demo")
        st.caption("Load results file, extract trajectories, and analyze opposition states.")
        
        if not RESULTS_DIR.exists():
            st.error(f"Could not find results directory: {RESULTS_DIR}")
            return
        
        result_files = sorted(RESULTS_DIR.glob("*.csv"))
        if not result_files:
            st.warning("No CSV files found in results/")
            return
        
        selected_result = st.selectbox(
            "Choose a results file",
            options=result_files,
            format_func=lambda path: path.name,
        )
        
        selected_df_preview = load_csv(selected_result)
        if "Iteration" in selected_df_preview.columns:
            max_iteration_in_file = int(pd.to_numeric(selected_df_preview["Iteration"], errors="coerce").max())
            iterations_to_consider = st.number_input(
                "Max iteration to consider",
                min_value=0,
                max_value=max_iteration_in_file,
                value=max_iteration_in_file,
                step=1,
                help="Only rows with Iteration <= this value are used.",
            )
        else:
            iterations_to_consider = None
            st.warning("Selected file has no 'Iteration' column; iteration filtering is disabled.")
        
        st.write(f"Selected file: **{selected_result.name}**")
        
        if st.button("Extract Trajectories", type="primary"):
            with st.spinner("Building trajectories..."):
                result_df = load_csv(selected_result)
                
                # Calculate perceived majority if needed (for v0 files)
                if "v0" in selected_result.name.lower():
                    result_df = calculate_perceived_majority(result_df)
                elif "Perceived Majority Opinion" not in result_df.columns:
                    st.error("Selected file does not contain 'Perceived Majority Opinion'. Please choose a CSV that already has this column.")
                    return
                
                # Build trajectories
                trajectories, processed_df = build_trajectories_with_distance(
                    result_df,
                    int(iterations_to_consider) if iterations_to_consider is not None else None,
                )
                
                # Estimate transition matrix
                transition_matrix, transition_counts = estimate_transition(trajectories, nS=N_STATES, nA=N_ACTIONS)
                
                # Compute feature expectations
                nS, nA = N_STATES, N_ACTIONS
                # Convert trajectories: action -1→0, 0→1, 1→2
                trajs_remapped = []
                for traj in trajectories:
                    traj_remapped = [(int(s), int(a) + 1) for s, a in traj]
                    trajs_remapped.append(traj_remapped)
                
                F = feature_matrix_sa_onehot(nS, nA)
                
                # Compute feature expectations normalized by trajectory length
                feature_expectations = np.zeros(F.shape[1])
                total_steps = 0
                for traj in trajs_remapped:
                    for state, action in traj:
                        idx = state * nA + action
                        feature_expectations += F[idx]
                        total_steps += 1
                
                # Normalize by total number of steps across all trajectories
                if total_steps > 0:
                    feature_expectations /= total_steps
                
                # Run MaxEnt IRL
                irl_outputs = maxent_irl_onehot(
                    P=transition_matrix,
                    trajectories=trajectories,
                    nS=N_STATES,
                    nA=N_ACTIONS,
                    gamma=0.9,
                    epochs=50,
                    learning_rate=0.5,
                    vi_threshold=1e-4,
                    horizon=None,
                    seed=42,
                )
                
                # Store in session state to persist across reruns
                st.session_state["trajectories"] = trajectories
                st.session_state["processed_df"] = processed_df
                st.session_state["transition_matrix"] = transition_matrix
                st.session_state["transition_counts"] = transition_counts
                st.session_state["result_df"] = result_df
                st.session_state["feature_expectations"] = feature_expectations
                st.session_state["irl_outputs"] = irl_outputs
                st.session_state["current_epoch"] = 0
            
            st.success("Trajectory extraction complete!")
        
        # Display results if they exist in session state
        if "trajectories" in st.session_state:
            trajectories = st.session_state["trajectories"]
            processed_df = st.session_state["processed_df"]
            transition_matrix = st.session_state["transition_matrix"]
            transition_counts = st.session_state["transition_counts"]
            result_df = st.session_state["result_df"]
            feature_expectations = st.session_state.get("feature_expectations", None)
            
            # Display stats
            stats = {
                "total_rows_in_file": len(result_df),
                "unique_agents": int(result_df["Agent Name"].nunique()) if "Agent Name" in result_df.columns else "N/A",
                "total_trajectories": len(trajectories),
                "mean_trajectory_length": round(float(np.mean([len(t) for t in trajectories])), 2) if trajectories else 0,
            }
            
            st.subheader("Trajectory Summary")
            st.dataframe(build_stats_table(stats), use_container_width=True)
            
            st.subheader("Trajectory Samples")
            sample_count = min(3, len(trajectories))
            if sample_count == 0:
                st.warning("No trajectories found with length >= 3.")
            else:
                for idx in range(sample_count):
                    st.write(f"**Trajectory {idx + 1}** (length={len(trajectories[idx])})")
                    st.code(str(trajectories[idx]), language="python")
            
            st.subheader("Processed Data Sample")
            display_cols = [
                "Agent Name",
                "Iteration",
                "Opinion",
                "Perceived Majority Opinion",
                "Willingness to Speak",
                "state",
                "action",
            ]
            existing_cols = [c for c in display_cols if c in processed_df.columns]
            st.dataframe(processed_df[existing_cols].head(20), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Transition Probability Estimation")
            
            with st.expander("📊 How estimate_transition() works"):
                st.markdown("""
**Algorithm (based on maxent.py's `find_feature_expectations` logic):**

1. **Count transitions** from trajectories:
   - For each trajectory, count every (s, a, s') transition
   - Accumulate counts into a 3D matrix: `counts[s, a, s']`

2. **Laplace smoothing** for numerical stability:
   - Add $\\epsilon = 10^{-6}$ to avoid division by zero
   - Ensures smooth probability distributions

3. **Normalize conditionally**:
   - For each state-action pair (s, a), compute:
   $$P(s' | s, a) = \\frac{\\text{count}[s, a, s'] + \\epsilon}{\\sum_{s'} \\text{count}[s, a, s'] + \\epsilon}$$
   - Result: probability distribution over next states

**Output:**
- $P \\in \\mathbb{R}^{5 \\times 3 \\times 5}$ where $P[s, a, s'] = P(s' | s, a)$
- Each slice sums to 1.0 (valid probability distribution)
                """)
            
            st.markdown("**Transition Probability Matrix**")
            st.caption("$P[s, a, s']$ = probability of transitioning from state $s$ to state $s'$ under action $a$")
            
            action_labels = {
                0: "Action -1 (Decrease)",
                1: "Action 0 (Stay)",
                2: "Action 1 (Increase)",
            }
            
            state_labels = [
                "Complete Agreement",
                "Mostly Agreement",
                "Small Disagreement",
                "Medium Disagreement",
                "Large Disagreement",
            ]
            
            action_to_show = st.selectbox(
                "Choose action slice to view",
                options=[0, 1, 2],
                format_func=lambda x: action_labels[x],
            )
            
            # Create display dataframe
            slice_probs = transition_matrix[:, action_to_show, :]
            slice_counts = transition_counts[:, action_to_show, :]
            
            # Display with counts and proportions in parenthesis
            display_data = []
            for i in range(len(state_labels)):
                row = []
                for j in range(len(state_labels)):
                    count = int(slice_counts[i, j])
                    prob = slice_probs[i, j]
                    row.append(f"{count} ({prob:.4f})")
                display_data.append(row)
            
            slice_df = pd.DataFrame(
                display_data,
                index=[f"from {s}" for s in state_labels],
                columns=[f"to {s}" for s in state_labels]
            )
            
            st.write(f"**{action_labels[action_to_show]}**")
            st.caption("Format: count (probability)")
            st.dataframe(slice_df, use_container_width=True)
            
            # Check if uniform
            if np.allclose(slice_probs, 1.0 / 3):
                st.warning(f"⚠️ **Uniform distribution detected**: All probabilities are ≈0.333 (1/3). This suggests few or no observed transitions for this action.")
            
            # Display feature expectations
            if feature_expectations is not None:
                st.markdown("---")
                st.subheader("Feature Expectations (Empirical)")
                
                with st.expander("📊 How find_feature_expectations() works"):
                    st.markdown("""
**Algorithm (state-action one-hot):**

1. **Initialize** feature expectations vector: $\\mu \\in \\mathbb{R}^D$ where $D$ is feature dimension
2. **Accumulate features** from all trajectories:
   - For each trajectory, for each (state, action) pair:
   - Convert to feature vector using feature matrix: $F[s \\cdot n_A + a]$
   - Add to running sum: $\\mu \\leftarrow \\mu + F[s \\cdot n_A + a]$
3. **Normalize by total steps**:
   $$\\mu = \\frac{1}{\\text{total steps}} \\sum_{\\text{traj}} \\sum_{(s,a) \\in \\text{traj}} F[s \\cdot n_A + a]$$

**Feature Matrix Structure:**
- One-hot per state-action pair: index = $s \cdot n_A + a$
- Total: $n_S \times n_A = 15$ features for $n_S=5, n_A=3$

**Output:**
- Empirical occupancy over state-action pairs (sums to 1 over all 9 features)
                    """)
                
                st.markdown("**Feature Expectations (State-Action Matrix)**")
                st.caption("Each cell = empirical proportion of steps at (state, action)")
                
                # Labels and reshape for 3x3 display
                state_names = [
                    "Complete Agreement",
                    "Mostly Agreement",
                    "Small Disagreement",
                    "Medium Disagreement",
                    "Large Disagreement",
                ]
                action_names = ["Decrease (-1)", "Stay (0)", "Increase (+1)"]

                sa_matrix = feature_expectations.reshape(N_STATES, N_ACTIONS)
                sa_df = pd.DataFrame(
                    sa_matrix,
                    index=[f"state: {s}" for s in state_names],
                    columns=[f"action: {a}" for a in action_names],
                )
                st.dataframe(sa_df.style.format("{:.4f}"), use_container_width=True)

                with st.expander("Show vector"):
                    feature_labels = [
                        f"(s={s_name}, a={a_name})"
                        for s_name in state_names
                        for a_name in action_names
                    ]
                    feat_df = pd.DataFrame(
                        {"Feature": feature_labels, "Expectation": feature_expectations}
                    )
                    st.dataframe(feat_df.style.format({"Expectation": "{:.4f}"}), use_container_width=True)
            
            # MaxEnt IRL Visualization
                # Display algorithm pipeline and formulas
                render_irl_pipeline()

            if "irl_outputs" in st.session_state and len(st.session_state["irl_outputs"]) > 0:
                st.markdown("---")
                st.subheader("MaxEnt IRL: Epoch-by-Epoch Learning")
                st.caption("Step through each epoch to visualize reward learning, policy updates, and occupancy convergence.")
                st.caption("Step-to-module map: Step 2 → Reward, Step 3 → Value Iteration (Q), Step 4 → Policy, Step 5 → Occupancy, Step 6 → Gradient Norm, Step 7 → α Update")
                
                irl_outputs = st.session_state["irl_outputs"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("⏮ First Epoch"):
                        st.session_state["current_epoch"] = 0
                with col2:
                    st.session_state["current_epoch"] = st.slider(
                        "Epoch",
                        min_value=0,
                        max_value=len(irl_outputs) - 1,
                        value=st.session_state.get("current_epoch", len(irl_outputs) - 1),
                    )
                with col3:
                    if st.button("Last Epoch ⏭"):
                        st.session_state["current_epoch"] = len(irl_outputs) - 1
                
                ep = st.session_state["current_epoch"]
                out = irl_outputs[ep]
                
                # Epoch info
                st.write(f"**Epoch {ep} / {len(irl_outputs) - 1}**")
                st.caption(f"Gradient norm: {out.gradient_norm:.6f}")
                
                # Step 2. Reward matrix
                with st.expander("📊 Step 2 — Learned Reward Matrix", expanded=True):
                    st.caption(f"Reward r(s,a) at epoch {ep}")
                    reward_df = pd.DataFrame(
                        out.reward,
                        index=[f"state: {s}" for s in state_names],
                        columns=[f"action: {a}" for a in action_names],
                    )
                    st.dataframe(reward_df.style.format("{:.4f}"), use_container_width=True)
                
                # Step 3. Q-values from value iteration
                with st.expander("🔢 Step 3 — Q-Values (from Value Iteration)"):
                    st.caption(f"Optimal Q-values V.I., epoch {ep}")
                    Q_df = pd.DataFrame(
                        out.Q,
                        index=[f"state: {s}" for s in state_names],
                        columns=[f"action: {a}" for a in action_names],
                    )
                    st.dataframe(Q_df.style.format("{:.4f}"), use_container_width=True)
                    st.markdown("**Value Iteration Convergence (interactive)**")
                    render_vi_history_viewer(
                        transition_matrix,
                        out.reward,
                        N_STATES,
                        N_ACTIONS,
                        state_names,
                        action_names,
                    )
                
                # Step 4. Policy (softmax over Q)
                with st.expander("🎯 Step 4 — Learned Policy (Softmax(Q))"):
                    st.caption(f"π(a|s) = softmax(Q(s,a)) at epoch {ep}")
                    policy_df = pd.DataFrame(
                        out.policy,
                        index=[f"state: {s}" for s in state_names],
                        columns=[f"action: {a}" for a in action_names],
                    )
                    st.dataframe(policy_df.style.format("{:.4f}"), use_container_width=True)

                # Step 5. Occupancy comparison
                with st.expander("📈 Step 5 — Expert vs Learned Occupancy"):
                    st.caption(f"μ_E (expert) vs μ_π (learned) at epoch {ep}")
                    
                    mu_E_matrix = out.mu_E.reshape(N_STATES, N_ACTIONS)
                    mu_pi_matrix = out.mu_pi.reshape(N_STATES, N_ACTIONS)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Expert Occupancy μ_E**")
                        mu_E_df = pd.DataFrame(
                            mu_E_matrix,
                            index=[f"state: {s}" for s in state_names],
                            columns=[f"action: {a}" for a in action_names],
                        )
                        st.dataframe(mu_E_df.style.format("{:.4f}"), use_container_width=True)
                    with col2:
                        st.write("**Learned Occupancy μ_π**")
                        mu_pi_df = pd.DataFrame(
                            mu_pi_matrix,
                            index=[f"state: {s}" for s in state_names],
                            columns=[f"action: {a}" for a in action_names],
                        )
                        st.dataframe(mu_pi_df.style.format("{:.4f}"), use_container_width=True)
                    
                    # Difference heatmap
                    st.write("**Occupancy Difference (μ_E - μ_π)**")
                    diff_matrix = mu_E_matrix - mu_pi_matrix
                    diff_df = pd.DataFrame(
                        diff_matrix,
                        index=[f"state: {s}" for s in state_names],
                        columns=[f"action: {a}" for a in action_names],
                    )
                    st.dataframe(diff_df.style.format("{:.4f}"), use_container_width=True)

                    with st.expander("🧠 How μ_π is computed (interactive rollout)"):
                        st.markdown("Learned occupancy is computed by rolling out the current policy over the transition model.")
                        st.markdown("- Start from empirical initial state distribution")
                        st.latex(r"d_0")
                        st.markdown("- Per step occupancy")
                        st.latex(r"\rho_t(s,a) = d_t(s)\,\pi(a\mid s)")
                        st.markdown("- Propagate state distribution")
                        st.latex(r"d_{t+1}(s') = \sum_{s,a} \rho_t(s,a)\,P(s,a,s')")
                        st.markdown("- Average over horizon")
                        st.latex(r"\mu_{\pi}(s,a)=\frac{1}{T}\sum_{t=0}^{T-1}\rho_t(s,a)")

                        lens = [len(t) for t in trajectories if len(t) > 0]
                        default_horizon = int(np.median(lens)) if lens else 10
                        max_horizon = max(2, min(50, max(lens) if lens else 20))
                        horizon_T = st.slider(
                            "Rollout horizon T",
                            min_value=1,
                            max_value=max_horizon,
                            value=min(default_horizon, max_horizon),
                            key=f"occ_horizon_ep_{ep}",
                        )

                        p0 = np.zeros(N_STATES, dtype=float)
                        if len(trajectories) > 0:
                            for traj in trajectories:
                                s0, _ = traj[0]
                                p0[int(s0)] += 1.0
                            if p0.sum() > 0:
                                p0 /= p0.sum()
                            else:
                                p0[:] = 1.0 / N_STATES
                        else:
                            p0[:] = 1.0 / N_STATES

                        d_t_list = []
                        rho_t_list = []
                        d = p0.copy()
                        for _ in range(horizon_T):
                            d_t_list.append(d.copy())
                            rho_t = d[:, None] * out.policy
                            rho_t_list.append(rho_t.copy())

                            d_next = np.zeros(N_STATES, dtype=float)
                            for s in range(N_STATES):
                                for a in range(N_ACTIONS):
                                    d_next += rho_t[s, a] * transition_matrix[s, a, :]
                            d = d_next

                        step_t = st.slider(
                            "Inspect rollout step t",
                            min_value=0,
                            max_value=horizon_T - 1,
                            value=min(1, horizon_T - 1),
                            key=f"occ_step_ep_{ep}",
                        )

                        c1, c2 = st.columns(2)
                        with c1:
                            st.write(f"**State distribution d_{step_t}**")
                            d_df = pd.DataFrame(
                                {"State": state_names, "Probability": d_t_list[step_t]}
                            )
                            st.dataframe(d_df.style.format({"Probability": "{:.4f}"}), use_container_width=True)
                        with c2:
                            st.write(f"**Per-step occupancy ρ_{step_t}(s,a)**")
                            rho_df = pd.DataFrame(
                                rho_t_list[step_t],
                                index=[f"state: {s}" for s in state_names],
                                columns=[f"action: {a}" for a in action_names],
                            )
                            st.dataframe(rho_df.style.format("{:.4f}"), use_container_width=True)

                        mu_pi_rollout = np.sum(rho_t_list, axis=0) / horizon_T
                        rollout_diff = float(np.max(np.abs(mu_pi_rollout.reshape(-1) - out.mu_pi)))
                        st.metric("max |μ_π(rollout) - μ_π(shown)|", f"{rollout_diff:.6f}")
                
                # Step 6. Convergence trace
                with st.expander("📉 Step 6 — Convergence: Gradient Norm Over Epochs"):
                    grad_norms = [out.gradient_norm for out in irl_outputs]
                    conv_df = pd.DataFrame({
                        "Epoch": range(len(grad_norms)),
                        "Gradient Norm": grad_norms,
                    })
                    st.line_chart(conv_df.set_index("Epoch")["Gradient Norm"])
                
                # Step 7. Alpha weights (feature weights)
                with st.expander("⚖️ Step 7 — Alpha Weights (Feature Salience)"):
                    st.caption(f"α (feature weights) at epoch {ep}")
                    alpha_labels = [
                        f"(s={s_name}, a={a_name})"
                        for s_name in state_names
                        for a_name in action_names
                    ]
                    alpha_df = pd.DataFrame({
                        "Feature": alpha_labels,
                        "α": out.alpha,
                    })
                    st.dataframe(alpha_df.style.format({"α": "{:.4f}"}), use_container_width=True)

                st.markdown("---")
                st.subheader("Final Results (Last Epoch)")
                final_out = irl_outputs[-1]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Final Learned Reward Matrix**")
                    final_reward_df = pd.DataFrame(
                        final_out.reward,
                        index=[f"state: {s}" for s in state_names],
                        columns=[f"action: {a}" for a in action_names],
                    )
                    st.dataframe(
                        final_reward_df.style.format("{:.4f}").background_gradient(cmap="RdYlGn", axis=None),
                        use_container_width=True
                    )
                
                with col2:
                    st.write("**Final Learned Policy**")
                    final_policy_df = pd.DataFrame(
                        final_out.policy,
                        index=[f"state: {s}" for s in state_names],
                        columns=[f"action: {a}" for a in action_names],
                    )
                    st.dataframe(
                        final_policy_df.style.format("{:.4f}").background_gradient(cmap="Blues", axis=None),
                        use_container_width=True
                    )

                greedy_actions = np.argmax(final_out.policy, axis=1)
                greedy_labels = [action_names[a] for a in greedy_actions]
                greedy_df = pd.DataFrame({
                    "State": state_names,
                    "Most likely action": greedy_labels,
                })
                st.caption("Greedy action per state from final policy")
                st.dataframe(greedy_df, use_container_width=True)


if __name__ == "__main__":
    main()

