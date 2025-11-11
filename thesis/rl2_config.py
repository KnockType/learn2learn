# thesis/l2l_rl2_config.py

class L2L_RL2:
    def __init__(self):
        # --- Environment and Task ---
        # Choose from: 'HalfCheetahForwardBackward-v1', 'AntForwardBackward-v1', 'AntDirection-v1'
        self.env_name = 'HalfCheetahForwardBackward-v1'

        # --- Logging and Saving ---
        self.num_epsiodes_of_validation = 5 # Episodes used for validation metrics
        self.num_lifetimes_for_validation = 20 # Lifetimes used for validation metrics

        # --- General Settings ---
        self.seeding = True
        self.seed = 42
        self.ol_device = 'cuda' # Outer-loop device
        self.il_device = 'cpu'  # Inner-loop device (Ray workers)

        # --- Data Collection ---
        self.num_outer_loop_updates = 500
        self.num_inner_loops_per_update = 20 # Number of parallel Ray workers
        self.num_il_lifetime_steps = 1000 # Steps per inner-loop lifetime

        # --- Optimizer ---
        self.learning_rate = 3e-4
        self.adam_eps = 1e-5

        # --- Advantage Estimation ---
        self.rewards_normalization = True # Normalize rewards per-task
        self.rewards_target_mean = 0.5
        self.meta_gamma = 0.99
        self.bootstrapping_lambda = 0.95

        # --- RL^2 Agent Architecture ---
        self.rnn_input_size = 64
        self.rnn_type = 'lstm'
        self.rnn_hidden_state_size = 256
        self.initial_std = 0.7

        # --- TBPTT PPO Hyperparameters ---
        self.ppo = {
            "k": 250,  # Truncation length for BPTT
            'update_epochs': 8,
            'num_minibatches': 0, # 0 means one grad step per subsequence
            "normalize_advantage": True,
            "clip_coef": 0.2,
            "entropy_coef": 0.001,
            "valuef_coef": 0.5,
            "clip_grad_norm": True,
            "max_grad_norm": 0.5,
            "target_KL": 0.05
        }

def get_config(config_settings):
    if config_settings == 'l2l_rl2':
        return L2L_RL2()
    else:
        raise ValueError(f"Unsupported config_setting: {config_settings}")