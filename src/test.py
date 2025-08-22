import os
import re
import glob
import argparse
import numpy as np

import ppo2 as network
from env_layered import LayeredEnv

TEST_LOG_FOLDER = './test_results/'
SUMMARY_DIR = './ppo'   # where train.py saves checkpoints
BEST_MODEL_PATH = os.path.join(SUMMARY_DIR, 'best.pth')


def _latest_ckpt(pattern=os.path.join(SUMMARY_DIR, 'nn_model_ep_*.pth')):
  files = glob.glob(pattern)
  if not files:
    return None
  def _ep(path):
    m = re.search(r'ep_(\d+)\.pth$', os.path.basename(path))
    return int(m.group(1)) if m else -1
  return max(files, key=_ep)


def main():
  parser = argparse.ArgumentParser(
    description="Evaluate a trained policy on the layered BL/E1/E2 environment."
  )
  parser.add_argument(
    'model_path',
    nargs='?',
    default='latest',
    help="Path to .pth checkpoint, or 'latest'/'best' to auto-pick from ./ppo"
  )
  parser.add_argument(
    '--stochastic',
    action='store_true',
    help='Sample an action from the policy instead of greedy argmax.'
  )
  args = parser.parse_args()

  os.makedirs(TEST_LOG_FOLDER, exist_ok=True)
  log_path = os.path.join(TEST_LOG_FOLDER, 'eval.txt')
  f = open(log_path, 'w')

  if args.model_path in ('latest', 'best'):
    if args.model_path == 'best' and os.path.isfile(BEST_MODEL_PATH):
      model_path = BEST_MODEL_PATH
    else:
      model_path = _latest_ckpt()
    if model_path is None:
      print("No checkpoints found in ./ppo. Provide an explicit path: python src/test.py /path/to/model.pth")
      return
  else:
    model_path = args.model_path

  # Build env and net (layered BL/E1/E2 setup)
  env = LayeredEnv(video_id="video_01", sizes_path="assets/sizes.json")
  state = env.reset()
  S_DIM = state.shape[0]
  A_DIM = 3

  actor = network.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=1e-4)
  try:
    actor.load_model(model_path)
  except Exception as e:
    print(f"Failed to load model '{model_path}': {e}")
    return

  done = False
  step = 0
  rng = np.random.RandomState(123)

  while not done:
    # policy inference
    probs = actor.predict(np.asarray(state, dtype=np.float32))
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum()

    # entropy for logging (avoid log(0))
    eps = 1e-12
    entropy = -float(np.sum(probs * np.log(probs + eps)))

    if args.stochastic:
      action = int(rng.choice(A_DIM, p=probs))
    else:
      action = int(np.argmax(probs))

    state, reward, done, info = env.step(action)

    # IMPORTANT: last two fields must remain (entropy, reward) for train.py parser
    f.write(f"{step} {entropy} {reward}\n")
    step += 1

  f.close()
  print(f"Evaluated checkpoint: {model_path}")
  print(f"Wrote log: {log_path}")


if __name__ == "__main__":
  main()
