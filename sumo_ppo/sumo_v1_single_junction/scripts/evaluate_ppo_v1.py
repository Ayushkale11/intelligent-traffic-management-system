import os
import sys
import numpy as np

from stable_baselines3 import PPO


# ================================================================
# PATH
# ================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_DIR
)

from traffic_env_v1 import SumoTrafficEnv


# ================================================================
# MODEL
# ================================================================

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "ppo_v1"
)


# ================================================================
# EVALUATION SETTINGS
# ================================================================

NUM_EPISODES = 5


# ================================================================
# EVALUATE ONE EPISODE
# ================================================================

def evaluate_episode(model, episode_number):

    env = SumoTrafficEnv(
        use_gui=False,
        max_episode_steps=120,
        sim_steps_per_action=5,
    )

    obs, info = env.reset()

    total_reward = 0.0

    total_waiting_time = 0.0
    total_queue = 0.0
    total_vehicles = 0.0

    steps = 0

    final_info = info

    while True:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        # Each RL step represents 5 SUMO seconds.
        total_waiting_time += (
            info["total_waiting"]
            * 5
        )

        total_queue += (
            info["total_waiting"]
        )

        total_vehicles += (
            info["total_vehicles"]
        )

        steps += 1

        final_info = info

        if terminated or truncated:
            break

    if steps > 0:

        average_waiting = (
            total_waiting_time
            / (steps * 5)
        )

        average_queue = (
            total_queue
            / steps
        )

        average_vehicles = (
            total_vehicles
            / steps
        )

    else:

        average_waiting = 0.0
        average_queue = 0.0
        average_vehicles = 0.0

    throughput = final_info["throughput"]

    final_vehicles = final_info["total_vehicles"]

    final_waiting = final_info["total_waiting"]

    switches = final_info["switches"]

    print()
    print(
        f"Episode {episode_number}"
    )
    print("-" * 50)

    print(
        f"Reward                  : "
        f"{total_reward:.2f}"
    )

    print(
        f"Average waiting         : "
        f"{average_waiting:.2f}"
    )

    print(
        f"Average queue           : "
        f"{average_queue:.2f}"
    )

    print(
        f"Average vehicles        : "
        f"{average_vehicles:.2f}"
    )

    print(
        f"Throughput              : "
        f"{throughput}"
    )

    print(
        f"Final vehicles          : "
        f"{final_vehicles}"
    )

    print(
        f"Final waiting           : "
        f"{final_waiting}"
    )

    print(
        f"Signal switches         : "
        f"{switches}"
    )

    env.close()

    return {
        "reward": total_reward,
        "waiting": average_waiting,
        "queue": average_queue,
        "vehicles": average_vehicles,
        "throughput": throughput,
        "final_vehicles": final_vehicles,
        "final_waiting": final_waiting,
        "switches": switches,
    }


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 60)
    print("SUMO V1 PPO EVALUATION")
    print("=" * 60)

    print()
    print(
        "Loading model:"
    )

    print(
        MODEL_PATH + ".zip"
    )

    # ------------------------------------------------------------
    # Load PPO
    # ------------------------------------------------------------

    model = PPO.load(
        MODEL_PATH
    )

    print()
    print("Model loaded successfully.")

    # ------------------------------------------------------------
    # Run episodes
    # ------------------------------------------------------------

    results = []

    for episode in range(
        1,
        NUM_EPISODES + 1
    ):

        result = evaluate_episode(
            model,
            episode
        )

        results.append(
            result
        )

    # ============================================================
    # AVERAGES
    # ============================================================

    avg_reward = np.mean(
        [r["reward"] for r in results]
    )

    avg_waiting = np.mean(
        [r["waiting"] for r in results]
    )

    avg_queue = np.mean(
        [r["queue"] for r in results]
    )

    avg_vehicles = np.mean(
        [r["vehicles"] for r in results]
    )

    avg_throughput = np.mean(
        [r["throughput"] for r in results]
    )

    avg_final_vehicles = np.mean(
        [r["final_vehicles"] for r in results]
    )

    avg_final_waiting = np.mean(
        [r["final_waiting"] for r in results]
    )

    avg_switches = np.mean(
        [r["switches"] for r in results]
    )

    # ============================================================
    # RESULTS
    # ============================================================

    print()
    print("=" * 60)
    print("PPO V1 AVERAGE RESULTS")
    print("=" * 60)

    print(
        f"Average reward         : "
        f"{avg_reward:.2f}"
    )

    print(
        f"Average waiting        : "
        f"{avg_waiting:.2f}"
    )

    print(
        f"Average queue          : "
        f"{avg_queue:.2f}"
    )

    print(
        f"Average vehicles       : "
        f"{avg_vehicles:.2f}"
    )

    print(
        f"Average throughput     : "
        f"{avg_throughput:.2f}"
    )

    print(
        f"Average final vehicles : "
        f"{avg_final_vehicles:.2f}"
    )

    print(
        f"Average final waiting  : "
        f"{avg_final_waiting:.2f}"
    )

    print(
        f"Average switches       : "
        f"{avg_switches:.2f}"
    )

    # ============================================================
    # BASELINE COMPARISON
    # ============================================================

    BASELINE_WAITING = 12.52
    BASELINE_THROUGHPUT = 375
    BASELINE_FINAL_VEHICLES = 39

    waiting_improvement = (
        (
            BASELINE_WAITING
            - avg_waiting
        )
        / BASELINE_WAITING
    ) * 100.0

    throughput_change = (
        (
            avg_throughput
            - BASELINE_THROUGHPUT
        )
        / BASELINE_THROUGHPUT
    ) * 100.0

    final_vehicle_change = (
        (
            avg_final_vehicles
            - BASELINE_FINAL_VEHICLES
        )
        / BASELINE_FINAL_VEHICLES
    ) * 100.0

    print()
    print("=" * 60)
    print("PPO vs FIXED-TIME")
    print("=" * 60)

    print(
        f"Waiting change      : "
        f"{waiting_improvement:+.2f}%"
    )

    print(
        f"Throughput change   : "
        f"{throughput_change:+.2f}%"
    )

    print(
        f"Final vehicles      : "
        f"{final_vehicle_change:+.2f}%"
    )

    print()
    print("=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()
