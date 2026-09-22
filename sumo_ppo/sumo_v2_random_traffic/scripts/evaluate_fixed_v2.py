import os
import sys
import numpy as np

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
# CONFIG
# ============================================================

NUM_EPISODES = 10
MAX_STEPS = 120

# Fixed-time decision interval
# 25 environment steps of NS, then 25 EW
FIXED_GREEN_STEPS = 25


# ============================================================
# RUN ONE EPISODE
# ============================================================

def run_episode():

    env = SumoTrafficEnv()

    try:

        obs, info = env.reset()

        total_reward = 0.0

        for step in range(MAX_STEPS):

            # ------------------------------------------------
            # Fixed-time controller
            #
            # 0 = NS
            # 1 = EW
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

            obs, reward, terminated, truncated, info = env.step(
                action
            )

            total_reward += float(reward)

            if terminated or truncated:
                break

        # ----------------------------------------------------
        # Collect metrics from environment
        # ----------------------------------------------------

        metric_steps = getattr(
            env,
            "metric_steps",
            0
        )

        total_waiting_time = getattr(
            env,
            "total_waiting_time",
            0.0
        )

        total_queue = getattr(
            env,
            "total_queue",
            0.0
        )

        total_vehicle_count = getattr(
            env,
            "total_vehicle_count",
            0.0
        )

        total_throughput = getattr(
            env,
            "total_throughput",
            0
        )

        switch_count = getattr(
            env,
            "switch_count",
            0
        )

        # ----------------------------------------------------
        # Averages
        # ----------------------------------------------------

        if metric_steps > 0:

            avg_waiting = (
                total_waiting_time /
                metric_steps
            )

            avg_queue = (
                total_queue /
                metric_steps
            )

            avg_vehicles = (
                total_vehicle_count /
                metric_steps
            )

        else:

            avg_waiting = 0.0
            avg_queue = 0.0
            avg_vehicles = 0.0

        # ----------------------------------------------------
        # Final values
        # ----------------------------------------------------

        final_vehicles = float(
            info.get(
                "total_vehicles",
                0
            )
        )

        final_waiting = float(
            info.get(
                "total_waiting",
                0
            )
        )

        return {
            "reward": total_reward,
            "waiting": avg_waiting,
            "queue": avg_queue,
            "vehicles": avg_vehicles,
            "throughput": total_throughput,
            "final_vehicles": final_vehicles,
            "final_waiting": final_waiting,
            "switches": switch_count
        }

    finally:

        try:
            env.close()
        except Exception:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("V2 FIXED-TIME BASELINE EVALUATION")
    print("=" * 60)

    print()
    print(f"Episodes           : {NUM_EPISODES}")
    print(f"Steps per episode  : {MAX_STEPS}")
    print(f"Fixed green period : {FIXED_GREEN_STEPS} steps")

    rewards = []
    waiting = []
    queues = []
    vehicles = []
    throughputs = []
    final_vehicles = []
    final_waiting = []
    switches = []

    # ========================================================
    # EPISODES
    # ========================================================

    for episode in range(
        1,
        NUM_EPISODES + 1
    ):

        print()
        print("-" * 60)

        print(
            f"Running Fixed-Time Episode "
            f"{episode}/{NUM_EPISODES}"
        )

        try:

            result = run_episode()

            rewards.append(
                result["reward"]
            )

            waiting.append(
                result["waiting"]
            )

            queues.append(
                result["queue"]
            )

            vehicles.append(
                result["vehicles"]
            )

            throughputs.append(
                result["throughput"]
            )

            final_vehicles.append(
                result["final_vehicles"]
            )

            final_waiting.append(
                result["final_waiting"]
            )

            switches.append(
                result["switches"]
            )

            print(
                f"Reward          : "
                f"{result['reward']:.2f}"
            )

            print(
                f"Avg waiting     : "
                f"{result['waiting']:.2f}"
            )

            print(
                f"Avg queue       : "
                f"{result['queue']:.2f}"
            )

            print(
                f"Avg vehicles    : "
                f"{result['vehicles']:.2f}"
            )

            print(
                f"Throughput      : "
                f"{result['throughput']}"
            )

            print(
                f"Final vehicles  : "
                f"{result['final_vehicles']:.2f}"
            )

            print(
                f"Final waiting   : "
                f"{result['final_waiting']:.2f}"
            )

            print(
                f"Switches        : "
                f"{result['switches']}"
            )

        except Exception as e:

            print()
            print(
                "Episode failed:"
            )

            print(
                type(e).__name__,
                e
            )

    # ========================================================
    # CHECK
    # ========================================================

    if len(waiting) == 0:

        raise RuntimeError(
            "No fixed-time episodes completed."
        )

    # ========================================================
    # AVERAGES
    # ========================================================

    print()
    print("=" * 60)
    print("V2 FIXED-TIME AVERAGE RESULTS")
    print("=" * 60)

    print(
        f"Average reward         : "
        f"{np.mean(rewards):.2f}"
    )

    print(
        f"Average waiting        : "
        f"{np.mean(waiting):.2f}"
    )

    print(
        f"Average queue          : "
        f"{np.mean(queues):.2f}"
    )

    print(
        f"Average vehicles       : "
        f"{np.mean(vehicles):.2f}"
    )

    print(
        f"Average throughput     : "
        f"{np.mean(throughputs):.2f}"
    )

    print(
        f"Average final vehicles : "
        f"{np.mean(final_vehicles):.2f}"
    )

    print(
        f"Average final waiting  : "
        f"{np.mean(final_waiting):.2f}"
    )

    print(
        f"Average switches       : "
        f"{np.mean(switches):.2f}"
    )

    print()
    print("=" * 60)
    print("FIXED-TIME V2 EVALUATION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
