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

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "models",
    "ppo_v3.zip"
)

MAX_STEPS = 1800

STEP_DELAY = 0.05


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PPO V3 SUMO-GUI DEMONSTRATION")
    print("=" * 60)

    print()
    print("Model:")
    print(MODEL_PATH)

    # ========================================================
    # LOAD PPO
    # ========================================================

    from stable_baselines3 import PPO

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"PPO model not found:\n{MODEL_PATH}"
        )

    # ========================================================
    # CREATE GUI ENVIRONMENT
    # ========================================================

    print()
    print("Creating SUMO-GUI environment...")

    env = SumoTrafficEnvV3(
        project_dir=PROJECT_DIR,
        gui=True,
        max_steps=MAX_STEPS,
        step_length=1.0
    )

    print("Environment created.")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = PPO.load(
        MODEL_PATH,
        env=env
    )

    print("PPO V3 model loaded.")

    # ========================================================
    # RESET
    # ========================================================

    print()
    print("Starting SUMO-GUI...")
    
    observation, info = env.reset()

    print()
    print("SUMO-GUI connected.")
    print()
    print("PPO adaptive traffic control running...")
    print()
    print("Press Ctrl+C in terminal to stop.")
    print()

    total_reward = 0.0
    previous_action = None
    switches = 0

    try:

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # PPO ACTION
            # ------------------------------------------------

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            action = np.asarray(
                action,
                dtype=np.int64
            ).flatten()

            # ------------------------------------------------
            # COUNT PHASE CHANGES
            # ------------------------------------------------

            if previous_action is not None:

                switches += int(
                    np.sum(
                        action != previous_action
                    )
                )

            previous_action = action.copy()

            # ------------------------------------------------
            # ENVIRONMENT STEP
            # ------------------------------------------------

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(action)

            total_reward += float(reward)

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            if step % 10 == 0:

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
                    f"Step {step:3d} | "
                    f"Action {action.tolist()} | "
                    f"Waiting {int(waiting):3d} | "
                    f"Queue {int(queue):3d} | "
                    f"Vehicles {int(vehicles):3d}"
                )

            time.sleep(STEP_DELAY)

            if terminated or truncated:
                break

    except KeyboardInterrupt:

        print()
        print("GUI demonstration stopped by user.")

    finally:

        print()
        print("=" * 60)
        print("PPO V3 GUI DEMO RESULTS")
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
