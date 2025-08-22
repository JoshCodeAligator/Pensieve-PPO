# --- timing ---
CHUNK_SEC = 2.0          # L: playback seconds per chunk in your traces
OVH_SEC   = 0.15         # per-request overhead (RTT etc.); set 0 if you want to ignore
BUFFER_NORM_SEC = 10.0
COLDSTART_MBPS = 3.0     


# --- reward (per slides: ignore rebuffer; alpha+beta=1) ---
ALPHA = 0.7              # weight on quality utility
BETA  = 0.3              # weight on quality change
USE_LOG_Q = False        # slide says q(l) = bitrate; set True to use log-bitrate

# --- state featurization ---
HIST_TP = 5              # number of recent throughput samples to keep
HIST_DT = 5              # number of recent download-time samples to keep

# --- feasibility guardrail (optional; speeds learning, avoids obvious stalls) ---
USE_MASK = True          # False = pure RL; True = only allow upgrades if time fits

# --- RL ---
GAMMA = 0.99
ENTROPY_W = 0.01
LR = 3e-4

