import os
import sys
import time

# ------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)


from traffic_env_v2 import SumoTrafficEnv


# ============================================================
# MAIN TEST
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SUMO V2 RL ENVIRONMENT TEST")
    print("=" * 60)

    env = None

    try:

        # ----------------------------------------------------
        # Create environment
        # ----------------------------------------------------

        print()
        print("Creating V2 environment...")

        env = SumoTrafficEnv()

        print("Environment created successfully.")

        # ----------------------------------------------------
        # Reset
        # ----------------------------------------------------

        print()
        print("Resetting environment...")

        obs, info = env.reset()

        print()
        print("Initial observation:")
        print(obs)

        print()
        print("Observation shape:")
        print(obs.shape)

        print()
        print("Initial info:")
        print(info)

        # ----------------------------------------------------
        # Basic validation
        # ----------------------------------------------------

        if obs is None:
            raise RuntimeError(
                "Observation is None."
            )

        if len(obs.shape) != 1:
            raise RuntimeError(
                f"Observation must be 1-D, got {obs.shape}"
            )

        print()
        print("Observation check: PASSED")

        # ----------------------------------------------------
        # Run simulation
        # ----------------------------------------------------

        print()
        print("-" * 60)
        print("RUNNING V2 RL ENVIRONMENT")
        print("-" * 60)

        total_reward = 0.0

        max_steps = 100

        for step in range(max_steps):

            # Alternate actions during environment test.
            #
            # 0 = NS
            # 1 = EW
            #
            # This is NOT training.
            # It simply verifies that signal control works.

            if step < 25:
                action = 0

            elif step < 50:
                action = 1

            elif step < 75:
                action = 0

            else:
                action = 1

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += reward

            if step % 5 == 0:

                waiting = info.get(
                    "total_waiting",
                    info.get(
                        "waiting",
                        0
                    )
                )

                vehicles = info.get(
                    "total_vehicles",
                    info.get(
                        "vehicles",
                        0
                    )
                )

                phase = info.get(
                    "phase",
                    "?"
                )

                print(
                    f"Step {step:3d} | "
                    f"Action {action} | "
                    f"Phase {phase} | "
                    f"Waiting {waiting:3} | "
                    f"Vehicles {vehicles:3} | "
                    f"Reward {reward:7.3f}"
                )

            if terminated or truncated:

                print()
                print(
                    f"Environment terminated at step {step}."
                )

                break

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("V2 ENVIRONMENT TEST RESULTS")
        print("=" * 60)

        print(
            f"Steps completed : {step + 1}"
        )

        print(
            f"Total reward    : {total_reward:.3f}"
        )

        print(
            f"Final phase     : "
            f"{info.get('phase', '?')}"
        )

        print(
            f"Final waiting   : "
            f"{info.get('total_waiting', info.get('waiting', 0))}"
        )

        print(
            f"Final vehicles  : "
            f"{info.get('total_vehicles', info.get('vehicles', 0))}"
        )

        print()
        print("=" * 60)
        print("V2 ENVIRONMENT TEST PASSED")
        print("=" * 60)

    except Exception as e:

        print()
        print("=" * 60)
        print("V2 ENVIRONMENT TEST FAILED")
        print("=" * 60)

        print()
        print(
            f"Error: {type(e).__name__}"
        )

        print(
            f"Message: {e}"
        )

        raise

    finally:

        if env is not None:

            try:
                env.close()

            except Exception as e:

                print(
                    f"Warning while closing environment: {e}"
                )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
