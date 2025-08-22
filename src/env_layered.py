import numpy as np
from typing import List
from layer_sizes import LayerSizeDB
from config_layered import (
    CHUNK_SEC, HIST_TP, HIST_DT, BUFFER_NORM_SEC, COLDSTART_MBPS,
    USE_MASK, OVH_SEC, ALPHA, BETA, USE_LOG_Q
)

class LayeredEnv:
    """
    One client, one video/channel.
    Action space: 0=EL1, 1=EL2, 2=SKIP
    """
    def __init__(self, video_id: str, sizes_path: str):
        # --- normalization / scale constants (define BEFORE reset) ---
        self._tp_scale = 1e6   # scale throughput from bps -> ~Mbps order
        self._dt_scale = 5.0   # normalize download-time/history to ~[0,1]
        self._eps      = 1e-6  # numerical safety
        self.vid = video_id
        self.db = LayerSizeDB(sizes_path)
        self.N = self.db.num_chunks(self.vid)
        self.reset()

    # --- public API ---
    @property
    def action_dim(self): return 3

    def reset(self):
        self.k = 0                      # chunk index
        self.buffer_s = 0.0             # seconds ready-to-play (== deadline slack)
        self.last_action = 0            # previous layered depth (for smoothness)
        self.last_q = 0.0               # previous quality utility
        self.tp_hist: List[float] = []  # recent measured throughputs (bps)
        self.dt_hist: List[float] = []  # recent download times (s)
        self.done = False
        return self._state()

    def step(self, action: int):
        # sizes for this chunk (bits), duration
        S_BL, S_E1, S_E2 = self.db.get_sizes_bits(self.vid, self.k)
        L = CHUNK_SEC

        # planned layers by action
        plan = {0:[S_BL], 1:[S_BL,S_E1], 2:[S_BL,S_E1,S_E2]}[action]

        # optional feasibility downgrade (mask)
        if USE_MASK:
            while len(plan) > 1 and not self._fits(plan):
                plan.pop()              # drop E2 then E1 if needed
                action -= 1

        # sequential downloads: BL then enhancements
        total_bits = 0
        stall_s = 0.0
        for i, S in enumerate(plan):
            R_est = self._thr_est()                 # bps scalar
            dt = S / max(R_est, 1e3) + (OVH_SEC if i==0 else 0.0)  # type: ignore # simple model

            # playback drains buffer during download
            if self.buffer_s > 0:
                drained = min(self.buffer_s, dt)
                self.buffer_s -= drained
                self.buffer_s = max(self.buffer_s, 0.0)
                stall_s += max(0.0, dt - drained)
            else:
                stall_s += dt

            total_bits += S

            # BL completion makes chunk playable => add L seconds
            if i == 0:
                self.buffer_s += L

            # measured throughput sample (exclude initial overhead)
            self.tp_hist.append(S / max(dt - (OVH_SEC if i==0 else 0.0), 1e-4))
            self.dt_hist.append(dt)

        # effective bitrate for this chunk (bits/sec)
        R_chunk = total_bits / L
        q_t = np.log(max(R_chunk, 1.0)) if USE_LOG_Q else R_chunk

        # reward per slides (no explicit stall term)
        dq = q_t - self.last_q
        r_t = ALPHA * q_t + BETA * dq

        # roll state
        self.last_q = q_t
        self.last_action = action
        self.k += 1
        self.done = (self.k >= self.N)

        return self._state(), r_t, self.done, {
            "R_chunk": R_chunk, "action": action, "stall": stall_s
        }

    # --- helpers ---
    def _state(self):
        # histories (left-pad with zeros)
        tp = np.array(self.tp_hist[-HIST_TP:], dtype=np.float32)
        dt = np.array(self.dt_hist[-HIST_DT:], dtype=np.float32)
        if tp.size < HIST_TP: tp = np.pad(tp, (HIST_TP-tp.size, 0))
        if dt.size < HIST_DT: dt = np.pad(dt, (HIST_DT-dt.size, 0))

        S_BL, S_E1, S_E2 = self.db.get_sizes_bits(self.vid, self.k % self.N)
        onehot = np.eye(3, dtype=np.float32)[self.last_action]

        feats = np.concatenate([
            np.array([self.buffer_s/float(BUFFER_NORM_SEC)], dtype=np.float32),
            (tp / self._tp_scale).astype(np.float32),
            (dt / self._dt_scale).astype(np.float32),
            np.array([S_BL, S_E1, S_E2], dtype=np.float32) / 1e6,
            onehot
        ], axis=0)
        return feats

    def _thr_est(self):
        if not self.tp_hist:
            return float(COLDSTART_MBPS) * 1e6  # cold-start per coding-configs
        x = np.array(self.tp_hist[-HIST_TP:], dtype=np.float64)
        hm = len(x) / np.sum(1.0 / np.clip(x, 1e3, None))
        return float(hm)

    def _fits(self, sizes_bits: list):
        slack = self.buffer_s
        R = self._thr_est()
        t = 0.0
        for i,S in enumerate(sizes_bits):
            t += S / max(R,1e3) + (OVH_SEC if i==0 else 0.0) # type: ignore
        return t <= slack + 1e-6