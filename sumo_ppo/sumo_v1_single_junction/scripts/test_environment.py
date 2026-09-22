import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.insert(0, PROJECT_ROOT)

from traffic_env_v1 import SumoTrafficEnv


def main():

    print("-----------------------------------")
    print("Testing RL signal control")
    print("-----------------------------------")

    env = SumoTrafficEnv(
        use_gui=False,
        max_episode_steps=50,
        sim_steps_per_action=5,
    )

    obs, info = env.reset()

    print("\nInitial state:")
    print(
        f"Phase {info['phase']} | "
        f"Direction {info['green_direction']} | "
        f"Green {info['green_duration']:.0f}s"
    )

    previous_phase = info["phase"]

    # ---------------------------------------------------------
    # Hold NS for 15 RL steps.
    # ---------------------------------------------------------

    print("\n========== NS GREEN ==========")

    for step in range(15):

        obs, reward, terminated, truncated, info = env.step(0)

        if info["phase"] != previous_phase:

            print(
                f"Step {step:2d} | "
                f"PHASE CHANGE -> {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Green {info['green_duration']:.0f}s"
            )

            previous_phase = info["phase"]

        if step % 5 == 0:

            print(
                f"Step {step:2d} | "
                f"Action 0 | "
                f"Phase {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Waiting {info['total_waiting']:3d} | "
                f"Vehicles {info['total_vehicles']:3d} | "
                f"Reward {reward:7.3f}"
            )

    # ---------------------------------------------------------
    # Request EW.
    # ---------------------------------------------------------

    print("\n========== REQUEST EW ==========")

    for step in range(10):

        obs, reward, terminated, truncated, info = env.step(1)

        if info["phase"] != previous_phase:

            print(
                f"Step {step:2d} | "
                f"PHASE CHANGE -> {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Green {info['green_duration']:.0f}s"
            )

            previous_phase = info["phase"]

        if step % 2 == 0:

            print(
                f"Step {step:2d} | "
                f"Action 1 | "
                f"Phase {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Waiting {info['total_waiting']:3d} | "
                f"Vehicles {info['total_vehicles']:3d} | "
                f"Reward {reward:7.3f}"
            )

    # ---------------------------------------------------------
    # Hold EW.
    # ---------------------------------------------------------

    print("\n========== HOLD EW GREEN ==========")

    for step in range(10):

        obs, reward, terminated, truncated, info = env.step(1)

        if info["phase"] != previous_phase:

            print(
                f"Step {step:2d} | "
                f"PHASE CHANGE -> {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Green {info['green_duration']:.0f}s"
            )

            previous_phase = info["phase"]

        if step % 5 == 0:

            print(
                f"Step {step:2d} | "
                f"Action 1 | "
                f"Phase {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Waiting {info['total_waiting']:3d} | "
                f"Vehicles {info['total_vehicles']:3d} | "
                f"Reward {reward:7.3f}"
            )

    # ---------------------------------------------------------
    # Request NS again.
    # ---------------------------------------------------------

    print("\n========== REQUEST NS ==========")

    for step in range(10):

        obs, reward, terminated, truncated, info = env.step(0)

        if info["phase"] != previous_phase:

            print(
                f"Step {step:2d} | "
                f"PHASE CHANGE -> {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Green {info['green_duration']:.0f}s"
            )

            previous_phase = info["phase"]

        if step % 2 == 0:

            print(
                f"Step {step:2d} | "
                f"Action 0 | "
                f"Phase {info['phase']} | "
                f"Direction {info['green_direction']} | "
                f"Waiting {info['total_waiting']:3d} | "
                f"Vehicles {info['total_vehicles']:3d} | "
                f"Reward {reward:7.3f}"
            )

        if terminated or truncated:
            break

    env.close()

    print("\n===================================")
    print("RL SIGNAL CONTROL TEST COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()
