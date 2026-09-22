import os
import sys
import time
import numpy as np

from stable_baselines3 import PPO


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "models",
    "ppo_v2.zip"
)

sys.path.insert(0, PROJECT_DIR)

from traffic_env_v2 import SumoTrafficEnv


# ============================================================
# SETTINGS
# ============================================================

MAX_STEPS = 120

GUI_DELAY = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PPO V2 SUMO-GUI VISUALIZATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"PPO model not found:\n{MODEL_PATH}"
        )

    print()
    print("Loading PPO V2 model...")

    model = PPO.load(
        MODEL_PATH
    )

    print("PPO model loaded successfully.")

    # --------------------------------------------------------
    # Create environment
    # --------------------------------------------------------

    print()
    print("Creating V2 environment...")

    env = SumoTrafficEnv()

    # --------------------------------------------------------
    # Tell environment to use SUMO-GUI
    # --------------------------------------------------------

    if hasattr(env, "sumo_binary"):

        env.sumo_binary = "sumo-gui"

    elif hasattr(env, "sumo_cmd"):

        try:

            env.sumo_cmd[0] = "sumo-gui"

        except Exception:
            pass

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    print()
    print("Starting SUMO-GUI...")

    try:

        observation, info = env.reset()

    except Exception:

        env.close()

        raise

    print()
    print("SUMO-GUI connected.")
    print()
    print("=" * 60)
    print("PPO CONTROLLER RUNNING")
    print("=" * 60)

    total_reward = 0.0

    # ========================================================
    # SIMULATION
    # ========================================================

    try:

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # PPO prediction
            # ------------------------------------------------

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            action = int(
                np.asarray(action).flatten()[0]
            )

            # ------------------------------------------------
            # Environment performs the action
            #
            # This is important:
            # traffic_env_v2 owns TraCI.
            # ------------------------------------------------

            result = env.step(action)

            observation = result[0]
            reward = result[1]
            terminated = result[2]
            truncated = result[3]
            info = result[4]

            total_reward += float(reward)

            # ------------------------------------------------
            # Information
            # ------------------------------------------------

            phase = info.get(
                "phase",
                getattr(
                    env,
                    "current_phase",
                    -1
                )
            )

            waiting = info.get(
                "total_waiting",
                0
            )

            vehicles = info.get(
                "total_vehicles",
                0
            )

            switches = info.get(
                "switches",
                getattr(
                    env,
                    "switch_count",
                    0
                )
            )

            phase_name = {
                getattr(env, "NS_GREEN", -1):
                    "NS GREEN",

                getattr(env, "NS_YELLOW", -2):
                    "NS YELLOW",

                getattr(env, "EW_GREEN", -3):
                    "EW GREEN",

                getattr(env, "EW_YELLOW", -4):
                    "EW YELLOW"
            }.get(
                phase,
                str(phase)
            )

            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            print(
                f"Step {step + 1:3d} | "
                f"Action {action} | "
                f"Phase {phase_name:10s} | "
                f"Waiting {waiting:3} | "
                f"Vehicles {vehicles:3} | "
                f"Reward {float(reward):7.3f} | "
                f"Switches {switches:2}"
            )

            # ------------------------------------------------
            # Slow down GUI
            # ------------------------------------------------

            time.sleep(GUI_DELAY)

            if terminated or truncated:

                print()
                print(
                    "Episode finished early."
                )

                break

    except KeyboardInterrupt:

        print()
        print(
            "Visualization interrupted by user."
        )

    finally:

        # ----------------------------------------------------
        # Close environment
        # ----------------------------------------------------

        print()
        print(
            "Closing PPO environment..."
        )

        try:

            env.close()

        except Exception as e:

            print(
                f"Warning while closing: {e}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("PPO V2 VISUALIZATION COMPLETE")
    print("=" * 60)

    print(
        f"Total reward : {total_reward:.2f}"
    )

    print(
        f"Final waiting: "
        f"{info.get('total_waiting', 0)}"
    )

    print(
        f"Final vehicles: "
        f"{info.get('total_vehicles', 0)}"
    )

    print(
        f"Total switches: "
        f"{info.get('switches', 0)}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
