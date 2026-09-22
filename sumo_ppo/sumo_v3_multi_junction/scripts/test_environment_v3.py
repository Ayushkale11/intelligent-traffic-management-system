import os
import sys
import traceback

import numpy as np


PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_DIR
)

from traffic_env_v3 import SumoTrafficEnvV3


def main():

    print()
    print("=" * 60)
    print("V3 MULTI-JUNCTION ENVIRONMENT TEST")
    print("=" * 60)

    env = None

    try:

        print()
        print("Creating V3 environment...")

        env = SumoTrafficEnvV3(
            gui=False,
            max_steps=120
        )

        print(
            "Environment created successfully."
        )

        print()
        print(
            "Observation space:",
            env.observation_space
        )

        print(
            "Action space:",
            env.action_space
        )

        print()
        print("Resetting environment...")

        observation, info = env.reset()

        print()
        print(
            "Initial observation size:",
            observation.shape
        )

        print(
            "Expected observation size:",
            env.observation_space.shape
        )

        print()
        print(
            "Initial traffic:"
        )

        print(
            "  Vehicles:",
            info["total_vehicles"]
        )

        print(
            "  Waiting:",
            info["total_waiting"]
        )

        print(
            "  Queue:",
            info["total_queue"]
        )

        print()
        print("=" * 60)
        print("RUNNING V3 ENVIRONMENT")
        print("=" * 60)

        total_reward = 0.0

        for step in range(120):

            # ------------------------------------------------
            # Test action:
            #
            # J1 J2 J3 J4
            #  0  0  0  0
            #
            # then change some junctions
            # ------------------------------------------------

            if step < 30:

                action = np.array(
                    [0, 0, 0, 0]
                )

            elif step < 60:

                action = np.array(
                    [1, 0, 1, 0]
                )

            elif step < 90:

                action = np.array(
                    [1, 1, 1, 1]
                )

            else:

                action = np.array(
                    [0, 1, 0, 1]
                )

            (
                observation,
                reward,
                terminated,
                truncated,
                info
            ) = env.step(
                action
            )

            total_reward += reward

            if step % 5 == 0:

                print(
                    f"Step {step:3d} | "
                    f"Action {action.tolist()} | "
                    f"Waiting "
                    f"{info['total_waiting']:3d} | "
                    f"Vehicles "
                    f"{info['total_vehicles']:3d} | "
                    f"Queue "
                    f"{info['total_queue']:3d} | "
                    f"Reward "
                    f"{reward:7.3f}"
                )

            if terminated or truncated:

                break

        print()
        print("=" * 60)
        print("V3 ENVIRONMENT TEST RESULTS")
        print("=" * 60)

        print(
            f"Steps completed : "
            f"{step + 1}"
        )

        print(
            f"Total reward    : "
            f"{total_reward:.3f}"
        )

        print(
            f"Final waiting   : "
            f"{info['total_waiting']}"
        )

        print(
            f"Final vehicles  : "
            f"{info['total_vehicles']}"
        )

        print(
            f"Final queue     : "
            f"{info['total_queue']}"
        )

        print(
            f"Total switches  : "
            f"{info['switches']}"
        )

        print()
        print("=" * 60)
        print("V3 ENVIRONMENT TEST PASSED")
        print("=" * 60)

    except Exception as e:

        print()
        print("=" * 60)
        print("V3 ENVIRONMENT TEST FAILED")
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

        traceback.print_exc()

    finally:

        if env is not None:

            try:

                env.close()

            except Exception:

                pass


if __name__ == "__main__":
    main()
