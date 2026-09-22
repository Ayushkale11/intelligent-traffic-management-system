import os
import sys
import time
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

MAX_STEPS = 1800
STEP_DELAY = 0.05

# Fixed-time phase switching interval
PHASE_DURATION = 30


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FIXED-TIME V3 SUMO-GUI DEMONSTRATION")
    print("=" * 60)

    print()
    print("Simulation steps :", MAX_STEPS)
    print("Phase duration    :", PHASE_DURATION)
    print()

    # ========================================================
    # CREATE ENVIRONMENT
    # ========================================================

    print("Creating SUMO-GUI environment...")

    env = SumoTrafficEnvV3(
        project_dir=PROJECT_DIR,
        gui=True,
        max_steps=MAX_STEPS,
        step_length=1.0
    )

    print("Environment created.")

    # ========================================================
    # RESET
    # ========================================================

    print()
    print("Starting SUMO-GUI...")

    observation, info = env.reset()

    print()
    print("SUMO-GUI connected.")
    print()
    print("Fixed-time traffic control running...")
    print()

    total_reward = 0.0
    switches = 0

    # Current phase for each junction
    current_action = np.array(
        [0, 0, 0, 0],
        dtype=np.int64
    )

    # ========================================================
    # RUN SIMULATION
    # ========================================================

    try:

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # FIXED-TIME CONTROL
            # ------------------------------------------------
            #
            # Every PHASE_DURATION seconds,
            # change all traffic lights.
            #
            # 0 -> 1
            # 1 -> 0
            #
            # This is NOT controlled by PPO.
            # ------------------------------------------------

            if step > 0 and step % PHASE_DURATION == 0:

                current_action = 1 - current_action

                switches += 4

            # ------------------------------------------------
            # ENVIRONMENT STEP
            # ------------------------------------------------

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(current_action)

            total_reward += float(reward)

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            if step % 30 == 0:

                waiting = info.get(
                    "total_waiting",
                    0
                )

                queue = info.get(
                    "total_queue",
                    0
                )

                vehicles = info.get(
                    "total_vehicles",
                    0
                )

                print(
                    f"Step {step:4d} | "
                    f"Action {current_action.tolist()} | "
                    f"Waiting {int(waiting):3d} | "
                    f"Queue {int(queue):3d} | "
                    f"Vehicles {int(vehicles):3d}"
                )

            time.sleep(STEP_DELAY)

            if terminated or truncated:
                break

    except KeyboardInterrupt:

        print()
        print("Fixed-time GUI demonstration stopped.")

    finally:

        print()
        print("=" * 60)
        print("FIXED-TIME V3 GUI DEMO RESULTS")
        print("=" * 60)

        print(
            f"Total reward : {total_reward:.2f}"
        )

        print(
            f"Switches     : {switches}"
        )

        print("=" * 60)

        env.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
