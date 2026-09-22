import os
import sys
import time
import numpy as np

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_DIR)

# ============================================================
# IMPORTS
# ============================================================

from stable_baselines3 import PPO
from traffic_env_v2 import SumoTrafficEnv


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "models",
    "ppo_v2.zip"
)

NUM_EPISODES = 10


# ============================================================
# PPO EVALUATION
# ============================================================

def evaluate_ppo():

    print()
    print("=" * 60)
    print("PPO V2 EVALUATION")
    print("=" * 60)

    print()
    print("Loading model:")
    print(MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"PPO V2 model not found:\n{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    env = SumoTrafficEnv()

    # --------------------------------------------------------
    # Load PPO model
    # --------------------------------------------------------

    model = PPO.load(
        MODEL_PATH,
        env=env
    )

    print()
    print("PPO V2 model loaded successfully.")

    # --------------------------------------------------------
    # Metric storage
    # --------------------------------------------------------

    rewards = []
    waiting = []
    queues = []
    vehicles = []
    throughputs = []
    final_vehicles = []
    final_waiting = []
    switches = []

    # --------------------------------------------------------
    # Run episodes
    # --------------------------------------------------------

    for episode in range(1, NUM_EPISODES + 1):

        print()
        print(
            "-" * 60
        )

        print(
            f"Episode {episode}/{NUM_EPISODES}"
        )

        observation, info = env.reset()

        done = False
        truncated = False

        episode_reward = 0.0

        step_count = 0

        while not done and not truncated:

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, done, truncated, info = env.step(
                action
            )

            episode_reward += float(reward)

            step_count += 1

        # ----------------------------------------------------
        # Collect environment metrics
        # ----------------------------------------------------

        episode_waiting = float(
            getattr(
                env,
                "total_waiting_time",
                0.0
            )
        )

        metric_steps = int(
            getattr(
                env,
                "metric_steps",
                0
            )
        )

        if metric_steps > 0:

            avg_waiting = (
                episode_waiting /
                metric_steps
            )

        else:

            avg_waiting = float(
                info.get(
                    "total_waiting",
                    0.0
                )
            )

        avg_queue = float(
            getattr(
                env,
                "total_queue",
                0.0
            )
        )

        if metric_steps > 0:

            avg_queue /= metric_steps

        else:

            avg_queue = float(
                info.get(
                    "total_waiting",
                    0.0
                )
            )

        avg_vehicles = float(
            getattr(
                env,
                "total_vehicle_count",
                0.0
            )
        )

        if metric_steps > 0:

            avg_vehicles /= metric_steps

        else:

            avg_vehicles = float(
                info.get(
                    "total_vehicles",
                    0.0
                )
            )

        episode_throughput = int(
            getattr(
                env,
                "total_throughput",
                info.get(
                    "throughput",
                    0
                )
            )
        )

        episode_switches = int(
            getattr(
                env,
                "switch_count",
                info.get(
                    "switches",
                    0
                )
            )
        )

        episode_final_vehicles = float(
            info.get(
                "total_vehicles",
                0
            )
        )

        episode_final_waiting = float(
            info.get(
                "total_waiting",
                0
            )
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        rewards.append(
            episode_reward
        )

        waiting.append(
            avg_waiting
        )

        queues.append(
            avg_queue
        )

        vehicles.append(
            avg_vehicles
        )

        throughputs.append(
            episode_throughput
        )

        final_vehicles.append(
            episode_final_vehicles
        )

        final_waiting.append(
            episode_final_waiting
        )

        switches.append(
            episode_switches
        )

        print(
            f"Steps       : {step_count}"
        )

        print(
            f"Reward      : {episode_reward:.2f}"
        )

        print(
            f"Avg waiting : {avg_waiting:.2f}"
        )

        print(
            f"Avg queue   : {avg_queue:.2f}"
        )

        print(
            f"Avg vehicles: {avg_vehicles:.2f}"
        )

        print(
            f"Throughput  : {episode_throughput}"
        )

        print(
            f"Switches    : {episode_switches}"
        )

    # ========================================================
    # AVERAGES
    # ========================================================

    results = {

        "reward":
            np.mean(rewards),

        "waiting":
            np.mean(waiting),

        "queue":
            np.mean(queues),

        "vehicles":
            np.mean(vehicles),

        "throughput":
            np.mean(throughputs),

        "final_vehicles":
            np.mean(final_vehicles),

        "final_waiting":
            np.mean(final_waiting),

        "switches":
            np.mean(switches)
    }

    # --------------------------------------------------------
    # Close environment
    # --------------------------------------------------------

    env.close()

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("PPO V2 AVERAGE RESULTS")
    print("=" * 60)

    print(
        f"Average reward         : "
        f"{results['reward']:.2f}"
    )

    print(
        f"Average waiting        : "
        f"{results['waiting']:.2f}"
    )

    print(
        f"Average queue          : "
        f"{results['queue']:.2f}"
    )

    print(
        f"Average vehicles       : "
        f"{results['vehicles']:.2f}"
    )

    print(
        f"Average throughput     : "
        f"{results['throughput']:.2f}"
    )

    print(
        f"Average final vehicles : "
        f"{results['final_vehicles']:.2f}"
    )

    print(
        f"Average final waiting  : "
        f"{results['final_waiting']:.2f}"
    )

    print(
        f"Average switches       : "
        f"{results['switches']:.2f}"
    )

    print()
    print("=" * 60)
    print("PPO V2 EVALUATION COMPLETE")
    print("=" * 60)

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    try:

        evaluate_ppo()

    except KeyboardInterrupt:

        print()
        print("Evaluation interrupted.")

    except Exception as e:

        print()
        print("=" * 60)
        print("PPO V2 EVALUATION FAILED")
        print("=" * 60)

        print()
        print(
            "Error:",
            type(e).__name__
        )

        print(
            "Message:",
            e
        )

        raise
