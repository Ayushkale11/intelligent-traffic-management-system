import os
import sys
import time
import traci
from stable_baselines3 import PPO

# Add the project root to Python's import path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from traffic_env_v1 import SumoTrafficEnv
# ============================================================
# CONFIG
# ============================================================


SUMO_CONFIG = os.path.join(BASE_DIR, "v1.sumocfg")
MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_v1")

TLS_ID = "J"

SIMULATION_TIME = 600

# Controls how quickly the GUI visualization runs.
# Increase this if the simulation is too fast.
FIXED_SLEEP = 0.08
PPO_SLEEP = 0.20


# ============================================================
# UTILITY
# ============================================================

def is_traci_connected():
    try:
        traci.getConnection()
        return True
    except Exception:
        return False


def print_banner(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# FIXED-TIME CONTROLLER
# ============================================================

def run_fixed_time():
    print_banner("FIXED-TIME CONTROLLER")

    print("Starting SUMO GUI...")
    print("Traffic demand: SAME as PPO V1")
    print("Simulation: 600 seconds")
    print()

    stats = {
        "steps": 0,
        "total_waiting": 0.0,
        "total_queue": 0.0,
        "total_vehicles": 0.0,
        "throughput": 0,
        "max_queue": 0,
        "max_vehicles": 0,
        "switches": 0,
    }

    previous_phase = None

    try:
        traci.start([
            "sumo-gui",
            "-c",
            SUMO_CONFIG,
            "--start",
            "--quit-on-end",
        ])

        print("SUMO GUI started.")
        print("Fixed-time signal is running using SUMO's default program.\n")

        while traci.simulation.getTime() < SIMULATION_TIME:

            traci.simulationStep()

            sim_time = traci.simulation.getTime()

            # --------------------------------------------
            # Vehicle statistics
            # --------------------------------------------

            vehicles = traci.vehicle.getIDList()

            vehicle_count = len(vehicles)

            waiting = sum(
                traci.vehicle.getWaitingTime(v)
                for v in vehicles
            )

            halting = sum(
                1
                for v in vehicles
                if traci.vehicle.getSpeed(v) < 0.1
            )

            arrived = traci.simulation.getArrivedNumber()

            # --------------------------------------------
            # Signal phase
            # --------------------------------------------

            phase = traci.trafficlight.getPhase(TLS_ID)

            if previous_phase is not None and phase != previous_phase:
                stats["switches"] += 1

            previous_phase = phase

            # --------------------------------------------
            # Accumulate statistics
            # --------------------------------------------

            stats["steps"] += 1
            stats["total_waiting"] += waiting
            stats["total_queue"] += halting
            stats["total_vehicles"] += vehicle_count
            stats["throughput"] += arrived

            stats["max_queue"] = max(
                stats["max_queue"],
                halting
            )

            stats["max_vehicles"] = max(
                stats["max_vehicles"],
                vehicle_count
            )

            # --------------------------------------------
            # Terminal display
            # --------------------------------------------

            if int(sim_time) % 20 == 0:

                print(
                    f"Time {sim_time:5.0f}s | "
                    f"Phase {phase} | "
                    f"Vehicles {vehicle_count:3d} | "
                    f"Waiting {halting:3d} | "
                    f"Arrived {stats['throughput']:3d}"
                )

            time.sleep(FIXED_SLEEP)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    finally:
        if is_traci_connected():
            traci.close()

    if stats["steps"] > 0:

        avg_waiting = (
            stats["total_waiting"] /
            stats["steps"]
        )

        avg_queue = (
            stats["total_queue"] /
            stats["steps"]
        )

        avg_vehicles = (
            stats["total_vehicles"] /
            stats["steps"]
        )

    else:
        avg_waiting = 0
        avg_queue = 0
        avg_vehicles = 0

    results = {
        "controller": "Fixed-Time",
        "avg_waiting": avg_waiting,
        "avg_queue": avg_queue,
        "avg_vehicles": avg_vehicles,
        "throughput": stats["throughput"],
        "final_vehicles": vehicle_count if stats["steps"] else 0,
        "final_waiting": halting if stats["steps"] else 0,
        "max_queue": stats["max_queue"],
        "max_vehicles": stats["max_vehicles"],
        "switches": stats["switches"],
    }

    print_results(results)

    return results


# ============================================================
# PPO V1 CONTROLLER
# ============================================================

def run_ppo():
    print_banner("PPO V1 CONTROLLER")

    print("Starting SUMO GUI...")
    print("Loading trained PPO model...")
    print("Traffic demand: SAME as Fixed-Time")
    print("Simulation: 600 seconds")
    print()

    model = PPO.load(MODEL_PATH)

    # Use the exact same environment used for PPO training/evaluation.
    env = SumoTrafficEnv(
        use_gui=True,
        max_episode_steps=120,
        sim_steps_per_action=5,
    )

    total_waiting = 0.0
    total_queue = 0.0
    total_vehicles = 0.0
    throughput = 0

    steps = 0

    max_queue = 0
    max_vehicles = 0

    last_info = {}

    try:

        obs, info = env.reset()

        print("PPO V1 loaded.")
        print("Agent is now controlling the traffic signal.\n")

        terminated = False
        truncated = False

        while not terminated and not truncated:

            # --------------------------------------------
            # PPO chooses signal action
            # --------------------------------------------

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = env.step(
                int(action)
            )

            steps += 1

            # --------------------------------------------
            # Metrics
            # --------------------------------------------

            waiting = info.get("total_waiting", 0)
            vehicles = info.get("total_vehicles", 0)
            arrived = info.get("throughput", 0)

            phase = info.get("phase", 0)

            total_waiting += waiting * 5
            total_queue += waiting * 5
            total_vehicles += vehicles * 5

            throughput = arrived

            max_queue = max(
                max_queue,
                waiting
            )

            max_vehicles = max(
                max_vehicles,
                vehicles
            )

            last_info = info

            # --------------------------------------------
            # Terminal display
            # --------------------------------------------

            sim_time = info.get(
                "simulation_time",
                steps * 5
            )

            print(
                f"Time {sim_time:5.0f}s | "
                f"Action {int(action)} | "
                f"Phase {phase} | "
                f"Vehicles {vehicles:3d} | "
                f"Waiting {waiting:3d} | "
                f"Reward {reward:7.3f}"
            )

            time.sleep(PPO_SLEEP)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")

    finally:
        env.close()

    if steps > 0:

        avg_waiting = total_waiting / (steps * 5)
        avg_queue = total_queue / (steps * 5)
        avg_vehicles = total_vehicles / (steps * 5)

    else:
        avg_waiting = 0
        avg_queue = 0
        avg_vehicles = 0

    results = {
        "controller": "PPO V1",
        "avg_waiting": avg_waiting,
        "avg_queue": avg_queue,
        "avg_vehicles": avg_vehicles,
        "throughput": throughput,
        "final_vehicles": last_info.get(
            "total_vehicles",
            0
        ),
        "final_waiting": last_info.get(
            "total_waiting",
            0
        ),
        "max_queue": max_queue,
        "max_vehicles": max_vehicles,
        "switches": last_info.get(
            "switch_count",
            0
        ),
    }

    print_results(results)

    return results


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(results):

    print("\n" + "-" * 65)
    print(f"{results['controller']} RESULTS")
    print("-" * 65)

    print(
        f"Average waiting vehicles : "
        f"{results['avg_waiting']:.2f}"
    )

    print(
        f"Average queue length     : "
        f"{results['avg_queue']:.2f}"
    )

    print(
        f"Average vehicles         : "
        f"{results['avg_vehicles']:.2f}"
    )

    print(
        f"Throughput               : "
        f"{results['throughput']}"
    )

    print(
        f"Final vehicles           : "
        f"{results['final_vehicles']}"
    )

    print(
        f"Final waiting vehicles   : "
        f"{results['final_waiting']}"
    )

    print(
        f"Maximum queue            : "
        f"{results['max_queue']}"
    )

    print(
        f"Maximum vehicles         : "
        f"{results['max_vehicles']}"
    )

    print(
        f"Signal phase changes     : "
        f"{results['switches']}"
    )

    print("-" * 65)


# ============================================================
# COMPARISON
# ============================================================

def compare_results(fixed, ppo):

    print_banner("FIXED-TIME vs PPO V1")

    print(
        f"{'Metric':<25}"
        f"{'Fixed-Time':>15}"
        f"{'PPO V1':>15}"
    )

    print("-" * 55)

    print(
        f"{'Average waiting':<25}"
        f"{fixed['avg_waiting']:>15.2f}"
        f"{ppo['avg_waiting']:>15.2f}"
    )

    print(
        f"{'Average queue':<25}"
        f"{fixed['avg_queue']:>15.2f}"
        f"{ppo['avg_queue']:>15.2f}"
    )

    print(
        f"{'Average vehicles':<25}"
        f"{fixed['avg_vehicles']:>15.2f}"
        f"{ppo['avg_vehicles']:>15.2f}"
    )

    print(
        f"{'Throughput':<25}"
        f"{fixed['throughput']:>15}"
        f"{ppo['throughput']:>15}"
    )

    print(
        f"{'Final vehicles':<25}"
        f"{fixed['final_vehicles']:>15}"
        f"{ppo['final_vehicles']:>15}"
    )

    print(
        f"{'Final waiting':<25}"
        f"{fixed['final_waiting']:>15}"
        f"{ppo['final_waiting']:>15}"
    )

    print(
        f"{'Signal changes':<25}"
        f"{fixed['switches']:>15}"
        f"{ppo['switches']:>15}"
    )

    print("-" * 55)

    waiting_change = (
        (fixed["avg_waiting"] - ppo["avg_waiting"])
        / fixed["avg_waiting"]
    ) * 100

    throughput_change = (
        (ppo["throughput"] - fixed["throughput"])
        / fixed["throughput"]
    ) * 100

    print(
        f"\nWaiting difference  : "
        f"{waiting_change:+.2f}%"
    )

    print(
        f"Throughput difference : "
        f"{throughput_change:+.2f}%"
    )

    print("\nVisualization comparison complete.")


# ============================================================
# MAIN
# ============================================================

def main():

    print_banner(
        "SUMO V1 - VISUAL CONTROLLER COMPARISON"
    )

    print("Choose controller:")
    print()
    print("1. Fixed-Time")
    print("2. PPO V1")
    print("3. Run BOTH sequentially")
    print()

    choice = input("Enter choice [1/2/3]: ").strip()

    if choice == "1":

        run_fixed_time()

    elif choice == "2":

        run_ppo()

    elif choice == "3":

        fixed_results = run_fixed_time()

        print("\n")
        input(
            "Fixed-Time finished. "
            "Press ENTER to launch PPO V1..."
        )

        ppo_results = run_ppo()

        compare_results(
            fixed_results,
            ppo_results
        )

    else:

        print("Invalid choice.")


if __name__ == "__main__":
    main()
