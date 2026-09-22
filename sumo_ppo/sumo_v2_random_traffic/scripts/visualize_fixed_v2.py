import os
import sys
import time


# ============================================================
# PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_DIR)

from traffic_env_v2 import SumoTrafficEnv


# ============================================================
# SETTINGS
# ============================================================

MAX_STEPS = 120

# Number of environment steps for each direction
FIXED_GREEN_STEPS = 25

GUI_DELAY = 0.15


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("FIXED-TIME V2 SUMO-GUI VISUALIZATION")
    print("=" * 60)

    print()
    print("Creating V2 environment...")

    env = SumoTrafficEnv()

    # --------------------------------------------------------
    # Tell environment to use SUMO-GUI
    # --------------------------------------------------------

    if hasattr(env, "sumo_binary"):

        env.sumo_binary = "sumo-gui"

    # --------------------------------------------------------
    # Start environment
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
    print("FIXED-TIME CONTROLLER RUNNING")
    print("=" * 60)

    total_reward = 0.0

    # ========================================================
    # SIMULATION
    # ========================================================

    try:

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # Fixed-time controller
            #
            # 0 = NS
            # 1 = EW
            #
            # Switch every FIXED_GREEN_STEPS
            # ------------------------------------------------

            cycle_position = (
                step %
                (FIXED_GREEN_STEPS * 2)
            )

            if cycle_position < FIXED_GREEN_STEPS:

                action = 0

            else:

                action = 1

            # ------------------------------------------------
            # Environment handles SUMO + TraCI
            # ------------------------------------------------

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += float(reward)

            # ------------------------------------------------
            # Get information
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

                getattr(
                    env,
                    "NS_GREEN",
                    -1
                ):
                    "NS GREEN",

                getattr(
                    env,
                    "NS_YELLOW",
                    -2
                ):
                    "NS YELLOW",

                getattr(
                    env,
                    "EW_GREEN",
                    -3
                ):
                    "EW GREEN",

                getattr(
                    env,
                    "EW_YELLOW",
                    -4
                ):
                    "EW YELLOW"
            }.get(
                phase,
                str(phase)
            )

            # ------------------------------------------------
            # Print status
            # ------------------------------------------------

            print(
                f"Step {step + 1:3d} | "
                f"Fixed Action {action} | "
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
            "Visualization interrupted."
        )

    finally:

        print()
        print(
            "Closing Fixed-Time environment..."
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
    print("FIXED-TIME V2 VISUALIZATION COMPLETE")
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
