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

# IMPORTANT:
# traffic_env_v3.py is in the project root
# and this script is inside scripts/
from traffic_env_v3 import SumoTrafficEnvV3


# ============================================================
# CONFIGURATION
# ============================================================

EPISODES = 10
MAX_STEPS = 120

# Fixed-time signal duration
PHASE_DURATION = 30

# Four V3 junctions
JUNCTIONS = ["J1", "J2", "J3", "J4"]


# ============================================================
# FIXED-TIME CONTROLLER
# ============================================================

def get_fixed_action(step):
    """
    Fixed-time controller.

    Phase 0:
        J1 = 0
        J2 = 0
        J3 = 0
        J4 = 0

    Phase 1:
        J1 = 1
        J2 = 1
        J3 = 1
        J4 = 1

    Every PHASE_DURATION steps, the phase changes.
    """

    phase = (step // PHASE_DURATION) % 2

    return np.array(
        [phase, phase, phase, phase],
        dtype=np.int64
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V3 FIXED-TIME MULTI-JUNCTION EVALUATION")
    print("=" * 60)

    print()
    print("Project directory :", PROJECT_DIR)
    print("Junctions          :", ", ".join(JUNCTIONS))
    print("Episodes           :", EPISODES)
    print("Steps per episode  :", MAX_STEPS)
    print("Phase duration     :", PHASE_DURATION)

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
    # RESULT ARRAYS
    # ========================================================

    all_rewards = []
    all_waiting = []
    all_queue = []
    all_vehicles = []
    all_throughput = []
    all_final_vehicles = []
    all_final_waiting = []
    all_switches = []

    # ========================================================
    # RUN EPISODES
    # ========================================================

    for episode in range(EPISODES):

        print()
        print("-" * 60)
        print(
            f"Running Fixed-Time Episode "
            f"{episode + 1}/{EPISODES}"
        )

        try:

            observation, info = env.reset()

            episode_reward = 0.0

            waiting_history = []
            queue_history = []
            vehicle_history = []
            throughput_history = []

            # -----------------------------------------------
            # Switch tracking
            # -----------------------------------------------

            previous_action = None
            episode_switches = 0

            # -----------------------------------------------
            # Simulation loop
            # -----------------------------------------------

            for step in range(MAX_STEPS):

                action = get_fixed_action(step)

                # -------------------------------------------
                # Count actual changes
                # -------------------------------------------

                if previous_action is not None:

                    changed_junctions = np.sum(
                        action != previous_action
                    )

                    episode_switches += int(
                        changed_junctions
                    )

                previous_action = action.copy()

                # -------------------------------------------
                # Environment step
                # -------------------------------------------

                observation, reward, terminated, truncated, info = env.step(
                    action
                )

                episode_reward += float(reward)

                # -------------------------------------------
                # Read metrics
                # -------------------------------------------

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

                throughput = info.get(
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
                    float(throughput)
                )

                # -------------------------------------------
                # Optional progress output
                # -------------------------------------------

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
            # EPISODE RESULTS
            # =================================================

            if waiting_history:

                average_waiting = np.mean(
                    waiting_history
                )

            else:
                average_waiting = 0.0

            if queue_history:

                average_queue = np.mean(
                    queue_history
                )

            else:
                average_queue = 0.0

            if vehicle_history:

                average_vehicles = np.mean(
                    vehicle_history
                )

            else:
                average_vehicles = 0.0

            if throughput_history:

                final_throughput = max(
                    throughput_history
                )

            else:
                final_throughput = 0.0

            if vehicle_history:

                final_vehicle_count = (
                    vehicle_history[-1]
                )

            else:
                final_vehicle_count = 0.0

            if waiting_history:

                final_waiting_count = (
                    waiting_history[-1]
                )

            else:
                final_waiting_count = 0.0

            # =================================================
            # SAVE RESULTS
            # =================================================

            all_rewards.append(
                episode_reward
            )

            all_waiting.append(
                average_waiting
            )

            all_queue.append(
                average_queue
            )

            all_vehicles.append(
                average_vehicles
            )

            all_throughput.append(
                final_throughput
            )

            all_final_vehicles.append(
                final_vehicle_count
            )

            all_final_waiting.append(
                final_waiting_count
            )

            all_switches.append(
                episode_switches
            )

            # =================================================
            # PRINT EPISODE SUMMARY
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
    # CHECK RESULTS
    # ========================================================

    if len(all_rewards) == 0:

        env.close()

        raise RuntimeError(
            "No fixed-time episodes completed."
        )

    # ========================================================
    # AVERAGE RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("V3 FIXED-TIME AVERAGE RESULTS")
    print("=" * 60)

    print(
        f"Completed episodes      : "
        f"{len(all_rewards)}/{EPISODES}"
    )

    print(
        f"Average reward          : "
        f"{np.mean(all_rewards):.2f}"
    )

    print(
        f"Average waiting         : "
        f"{np.mean(all_waiting):.2f}"
    )

    print(
        f"Average queue           : "
        f"{np.mean(all_queue):.2f}"
    )

    print(
        f"Average vehicles        : "
        f"{np.mean(all_vehicles):.2f}"
    )

    print(
        f"Average throughput      : "
        f"{np.mean(all_throughput):.2f}"
    )

    print(
        f"Average final vehicles  : "
        f"{np.mean(all_final_vehicles):.2f}"
    )

    print(
        f"Average final waiting   : "
        f"{np.mean(all_final_waiting):.2f}"
    )

    print(
        f"Average switches        : "
        f"{np.mean(all_switches):.2f}"
    )

    print()
    print("=" * 60)
    print("FIXED-TIME V3 EVALUATION COMPLETE")
    print("=" * 60)

    # ========================================================
    # CLOSE ENVIRONMENT
    # ========================================================

    env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
