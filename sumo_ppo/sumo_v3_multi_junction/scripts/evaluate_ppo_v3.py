import os
import sys
import numpy as np

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from traffic_env_v3 import SumoTrafficEnvV3


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 10
MAX_STEPS = 120

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "models",
    "ppo_v3.zip"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PPO V3 MULTI-JUNCTION EVALUATION")
    print("=" * 60)

    print()
    print("Project directory :", PROJECT_DIR)
    print("Model             :", MODEL_PATH)
    print("Episodes          :", EPISODES)
    print("Steps / episode   :", MAX_STEPS)

    # ========================================================
    # IMPORT PPO
    # ========================================================

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print()
        print("ERROR: stable-baselines3 is not installed.")
        return

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    print()
    print("Creating V3 environment...")

    env = SumoTrafficEnvV3(
        project_dir=PROJECT_DIR,
        gui=False,
        max_steps=MAX_STEPS,
        step_length=1.0
    )

    print("Environment created successfully.")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("Loading PPO V3 model...")

    if not os.path.exists(MODEL_PATH):

        env.close()

        raise FileNotFoundError(
            f"PPO model not found:\n{MODEL_PATH}"
        )

    model = PPO.load(
        MODEL_PATH,
        env=env
    )

    print("PPO model loaded successfully.")

    # ========================================================
    # RESULT STORAGE
    # ========================================================

    rewards = []
    waiting = []
    queues = []
    vehicles = []
    throughput = []
    final_vehicles = []
    final_waiting = []
    switches = []

    # ========================================================
    # RUN EPISODES
    # ========================================================

    for episode in range(EPISODES):

        print()
        print("-" * 60)
        print(
            f"Running PPO Episode "
            f"{episode + 1}/{EPISODES}"
        )

        try:

            observation, info = env.reset()

            episode_reward = 0.0

            waiting_history = []
            queue_history = []
            vehicle_history = []
            throughput_history = []

            # =================================================
            # CORRECT SWITCH TRACKING
            # =================================================

            previous_action = None
            episode_switches = 0

            # =================================================
            # SIMULATION
            # =================================================

            for step in range(MAX_STEPS):

                # ---------------------------------------------
                # PPO chooses action
                # ---------------------------------------------

                action, _states = model.predict(
                    observation,
                    deterministic=True
                )

                action = np.asarray(
                    action,
                    dtype=np.int64
                ).flatten()

                # ---------------------------------------------
                # Count REAL junction phase changes
                #
                # Example:
                #
                # Previous [0,0,0,0]
                # Current  [1,0,1,0]
                #
                # = 2 switches
                # ---------------------------------------------

                if previous_action is not None:

                    changed_junctions = np.sum(
                        action != previous_action
                    )

                    episode_switches += int(
                        changed_junctions
                    )

                previous_action = action.copy()

                # ---------------------------------------------
                # Environment step
                # ---------------------------------------------

                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info
                ) = env.step(action)

                episode_reward += float(reward)

                # ---------------------------------------------
                # Metrics
                # ---------------------------------------------

                total_waiting = info.get(
                    "total_waiting",
                    0
                )

                total_queue = info.get(
                    "total_queue",
                    0
                )

                total_vehicles = info.get(
                    "total_vehicles",
                    0
                )

                total_throughput = info.get(
                    "throughput",
                    0
                )

                waiting_history.append(
                    float(total_waiting)
                )

                queue_history.append(
                    float(total_queue)
                )

                vehicle_history.append(
                    float(total_vehicles)
                )

                throughput_history.append(
                    float(total_throughput)
                )

                # ---------------------------------------------
                # Progress
                # ---------------------------------------------

                if step % 20 == 0:

                    print(
                        f"Step {step:3d} | "
                        f"Action {action.tolist()} | "
                        f"Waiting {int(total_waiting):3d} | "
                        f"Vehicles {int(total_vehicles):3d} | "
                        f"Queue {int(total_queue):3d} | "
                        f"Reward {float(reward):8.3f}"
                    )

                if terminated or truncated:
                    break

            # =================================================
            # CALCULATE EPISODE METRICS
            # =================================================

            average_waiting = (
                np.mean(waiting_history)
                if waiting_history
                else 0.0
            )

            average_queue = (
                np.mean(queue_history)
                if queue_history
                else 0.0
            )

            average_vehicles = (
                np.mean(vehicle_history)
                if vehicle_history
                else 0.0
            )

            final_throughput = (
                max(throughput_history)
                if throughput_history
                else 0.0
            )

            final_vehicle_count = (
                vehicle_history[-1]
                if vehicle_history
                else 0.0
            )

            final_waiting_count = (
                waiting_history[-1]
                if waiting_history
                else 0.0
            )

            # =================================================
            # STORE RESULTS
            # =================================================

            rewards.append(
                episode_reward
            )

            waiting.append(
                average_waiting
            )

            queues.append(
                average_queue
            )

            vehicles.append(
                average_vehicles
            )

            throughput.append(
                final_throughput
            )

            final_vehicles.append(
                final_vehicle_count
            )

            final_waiting.append(
                final_waiting_count
            )

            switches.append(
                episode_switches
            )

            # =================================================
            # EPISODE SUMMARY
            # =================================================

            print()
            print(
                f"Episode {episode + 1} completed:"
            )

            print(
                f"  Reward         : "
                f"{episode_reward:.2f}"
            )

            print(
                f"  Avg waiting    : "
                f"{average_waiting:.2f}"
            )

            print(
                f"  Avg queue      : "
                f"{average_queue:.2f}"
            )

            print(
                f"  Avg vehicles   : "
                f"{average_vehicles:.2f}"
            )

            print(
                f"  Throughput     : "
                f"{final_throughput:.2f}"
            )

            print(
                f"  Final vehicles : "
                f"{final_vehicle_count:.0f}"
            )

            print(
                f"  Final waiting  : "
                f"{final_waiting_count:.0f}"
            )

            print(
                f"  Switches       : "
                f"{episode_switches}"
            )

        except Exception as e:

            print()
            print(
                f"Episode {episode + 1} failed:"
            )

            print(
                type(e).__name__,
                str(e)
            )

            continue

    # ========================================================
    # CHECK
    # ========================================================

    if len(rewards) == 0:

        env.close()

        raise RuntimeError(
            "No PPO episodes completed."
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("PPO V3 AVERAGE RESULTS")
    print("=" * 60)

    print(
        f"Completed episodes      : "
        f"{len(rewards)}/{EPISODES}"
    )

    print(
        f"Average reward          : "
        f"{np.mean(rewards):.2f}"
    )

    print(
        f"Average waiting         : "
        f"{np.mean(waiting):.2f}"
    )

    print(
        f"Average queue           : "
        f"{np.mean(queues):.2f}"
    )

    print(
        f"Average vehicles        : "
        f"{np.mean(vehicles):.2f}"
    )

    print(
        f"Average throughput      : "
        f"{np.mean(throughput):.2f}"
    )

    print(
        f"Average final vehicles  : "
        f"{np.mean(final_vehicles):.2f}"
    )

    print(
        f"Average final waiting   : "
        f"{np.mean(final_waiting):.2f}"
    )

    print(
        f"Average switches        : "
        f"{np.mean(switches):.2f}"
    )

    print()
    print("=" * 60)
    print("PPO V3 EVALUATION COMPLETE")
    print("=" * 60)

    env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
