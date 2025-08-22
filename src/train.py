import multiprocessing as mp
import numpy as np
import os
import signal
import sys
import shutil
from typing import List, Tuple

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from env_layered import LayeredEnv
import ppo2 as network
# Optional config import with safe defaults
try:
    from config_layered import GAMMA, ENTROPY_W
except Exception:
    GAMMA = 0.99
    ENTROPY_W = 0.01

# ---- Runtime guardrails ----
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TENSORBOARD_NO_TF', '1')  # ensure TB has no TF deps

# ---- Hyperparams / Paths ----
A_DIM = 3                 # actions: BL-only, BL+E1, BL+E1+E2
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 4            # parallel env workers
TRAIN_SEQ_LEN = 50        # steps per rollout per agent
TRAIN_EPOCH = 2000
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
SUMMARY_DIR = './ppo'
MODEL_DIR = './models'
TRAIN_TRACES = './train/'           # kept for compatibility
TEST_LOG_FOLDER = './test_results/' # test.py will write here
LOG_FILE = os.path.join(SUMMARY_DIR, 'log')
EVAL_EVERY = MODEL_SAVE_INTERVAL
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best.pth')

for d in (SUMMARY_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

NN_MODEL = None  # path to warm-start; set to a .pth to restore

# ---- Determine state dimension from a probe env (safe now) ----
_probe = LayeredEnv(video_id="video_01", sizes_path="assets/sizes.json")
S_DIM = _probe.reset().shape[0]
del _probe


# ---- Evaluation helper (calls test.py) ----
def testing(epoch: int, nn_model: str, log_file):
    shutil.rmtree(TEST_LOG_FOLDER, ignore_errors=True)
    os.makedirs(TEST_LOG_FOLDER, exist_ok=True)
    exit_code = os.system(f'"{sys.executable}" src/test.py "{nn_model}"')
    if exit_code != 0:
        # If test failed to run, return neutral metrics
        return 0.0, 0.0

    rewards, entropies = [], []
    for fname in os.listdir(TEST_LOG_FOLDER):
        path = os.path.join(TEST_LOG_FOLDER, fname)
        if not os.path.isfile(path):
            continue
        r_list, e_list = [], []
        with open(path, 'rb') as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    # last two fields by convention: entropy, reward
                    e_list.append(float(parts[-2]))
                    r_list.append(float(parts[-1]))
                except ValueError:
                    continue
        if r_list:
            rewards.append(float(np.mean(r_list[1:] if len(r_list) > 1 else r_list)))
        if e_list:
            entropies.append(float(np.mean(e_list[1:] if len(e_list) > 1 else e_list)))

    if rewards:
        rewards_np = np.array(rewards, dtype=np.float32)
        log_file.write(
            f"{epoch}\t"
            f"{np.min(rewards_np):.6f}\t"
            f"{np.percentile(rewards_np, 5):.6f}\t"
            f"{np.mean(rewards_np):.6f}\t"
            f"{np.percentile(rewards_np, 50):.6f}\t"
            f"{np.percentile(rewards_np, 95):.6f}\t"
            f"{np.max(rewards_np):.6f}\n"
        )
        log_file.flush()
        return float(np.mean(rewards_np)), float(np.mean(entropies) if entropies else 0.0)
    else:
        # no logs produced; return neutral metrics
        return 0.0, 0.0


# ---- Central learner process ----
def central_agent(net_params_queues: List[mp.Queue], exp_queues: List[mp.Queue]):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        actor = network.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        writer = SummaryWriter(SUMMARY_DIR)
        best_reward = -1e9

        # Optional warm-start
        if NN_MODEL is not None and os.path.isfile(NN_MODEL):
            actor.load_model(NN_MODEL)
            print('Model restored from', NN_MODEL)

        for epoch in range(TRAIN_EPOCH):
            # broadcast params to all agents
            params = actor.get_network_params()
            for q in net_params_queues:
                q.put(params)

            # collect rollouts
            s, a, p, v_targets = [], [], [], []
            for q in exp_queues:
                s_, a_, p_, v_ = q.get()
                s += s_
                a += a_
                p += p_
                v_targets += list(v_)

            if not s:
                print('[central] warning: empty rollout batch; skipping update')
                continue

            # stack tensors
            s_batch = np.asarray(s, dtype=np.float32)
            a_batch = np.asarray(a, dtype=np.float32)
            p_batch = np.asarray(p, dtype=np.float32)
            v_batch = np.asarray(v_targets, dtype=np.float32).reshape(-1)

            # single A2C update
            logs = actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            # periodic eval + checkpoint
            if epoch % MODEL_SAVE_INTERVAL == 0:
                ckpt = os.path.join(SUMMARY_DIR, f'nn_model_ep_{epoch}.pth')
                actor.save_model(ckpt)
                avg_reward, avg_entropy = testing(epoch, ckpt, test_log_file)
                # Track and save the best checkpoint
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    actor.save_model(BEST_MODEL_PATH)
                    print(f'[central] new best reward {best_reward:.3f} -> saved {BEST_MODEL_PATH}')
                writer.add_scalar('train/entropy_weight', ENTROPY_W, epoch)
                writer.add_scalar('eval/reward', avg_reward, epoch)
                writer.add_scalar('eval/entropy', avg_entropy, epoch)
                writer.add_scalar('train/loss', logs.get('loss', 0.0), epoch)
                writer.add_scalar('train/policy_loss', logs.get('policy_loss', 0.0), epoch)
                writer.add_scalar('train/value_loss', logs.get('value_loss', 0.0), epoch)
                writer.flush()

    print('[central] finished')


# ---- Environment worker process ----
def agent(agent_id: int, net_params_queue: mp.Queue, exp_queue: mp.Queue):
    env = LayeredEnv(video_id="video_01", sizes_path="assets/sizes.json")
    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)

    # wait for initial params
    actor.set_network_params(net_params_queue.get())

    rng = np.random.RandomState(RANDOM_SEED + agent_id)

    for _ in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []

        for _ in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            # policy inference
            probs = actor.predict(np.asarray(obs, dtype=np.float32))

            # exploration via Gumbel-max
            probs = np.clip(probs, 1e-8, 1.0)  # numerical safety
            g = rng.gumbel(size=probs.shape[-1])
            bit_rate = int(np.argmax(np.log(probs) + g))

            obs, rew, done, info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM, dtype=np.float32)
            action_vec[bit_rate] = 1.0

            a_batch.append(action_vec)
            p_batch.append(probs.astype(np.float32))
            r_batch.append(float(rew))

            if done:
                break

        if not s_batch:
            # Fresh episode terminated before collecting any step; send a tiny no-op batch
            exp_queue.put([[], [], [], np.zeros((0,), dtype=np.float32)])
            # pull latest params for next rollout
            actor.set_network_params(net_params_queue.get())
            continue

        # bootstrap value (shape-aligned helper)
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)

        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

        # pull latest params for next rollout
        actor.set_network_params(net_params_queue.get())


# ---- Orchestration ----
def main():
    mp.set_start_method('spawn', force=True)
    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    net_params_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]
    exp_queues = [mp.Queue(1) for _ in range(NUM_AGENTS)]

    coordinator = mp.Process(target=central_agent, args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = [mp.Process(target=agent, args=(i, net_params_queues[i], exp_queues[i]))
              for i in range(NUM_AGENTS)]
    for p in agents:
        p.start()

    try:
        coordinator.join()
    except KeyboardInterrupt:
        print('\n[main] KeyboardInterrupt: terminating children...')
        for p in agents:
            if p.is_alive():
                p.terminate()
        if coordinator.is_alive():
            coordinator.terminate()
    finally:
        for p in agents:
            p.join(timeout=1)
        if coordinator.is_alive():
            coordinator.join(timeout=1)


if __name__ == '__main__':
    main()
