import os
import sys
import traci


# ================================================================
# PATH SETUP
# ================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        ".."
    )
)

SUMO_CONFIG = os.path.join(
    PROJECT_DIR,
    "v1.sumocfg"
)


# ================================================================
# SUMO SETTINGS
# ================================================================

SUMO_BINARY = "sumo"

TLS_ID = "J"

# SUMO phases:
#
# 0 = NS Green
# 1 = NS Yellow
# 2 = EW Green
# 3 = EW Yellow

NS_GREEN = 0
NS_YELLOW = 1

EW_GREEN = 2
EW_YELLOW = 3


# ================================================================
# SIMULATION SETTINGS
# ================================================================

SIMULATION_END = 600

INCOMING_EDGES = [
    "N_J",
    "S_J",
    "E_J",
    "W_J",
]


# ================================================================
# START SUMO
# ================================================================

def start_sumo():

    command = [
        SUMO_BINARY,
        "-c",
        SUMO_CONFIG,

        "--no-step-log",
        "true",

        "--duration-log.disable",
        "true",

        "--time-to-teleport",
        "-1",
    ]

    traci.start(command)


# ================================================================
# GET TRAFFIC STATISTICS
# ================================================================

def get_traffic_stats():

    total_vehicles = 0
    total_waiting = 0

    for edge in INCOMING_EDGES:

        try:

            vehicles = (
                traci.edge
                .getLastStepVehicleNumber(edge)
            )

            waiting = (
                traci.edge
                .getLastStepHaltingNumber(edge)
            )

        except Exception:

            vehicles = 0
            waiting = 0

        total_vehicles += vehicles
        total_waiting += waiting

    return (
        total_vehicles,
        total_waiting
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 60)
    print("SUMO V1 FIXED-TIME BASELINE")
    print("=" * 60)

    print()
    print("Signal program:")
    print("NS Green  : 42 seconds")
    print("NS Yellow :  3 seconds")
    print("EW Green  : 42 seconds")
    print("EW Yellow :  3 seconds")

    print()
    print("Simulation:", SIMULATION_END, "seconds")
    print()

    # ------------------------------------------------------------
    # Start SUMO
    # ------------------------------------------------------------

    start_sumo()

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    total_waiting_time = 0.0

    total_queue = 0.0

    total_vehicle_count = 0.0

    metric_steps = 0

    throughput = 0

    max_queue = 0

    max_vehicles = 0

    signal_switches = 0

    previous_phase = None

    # ------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------

    while True:

        current_time = traci.simulation.getTime()

        if current_time >= SIMULATION_END:
            break

        # Advance exactly one SUMO second.

        traci.simulationStep()

        current_time = traci.simulation.getTime()

        # --------------------------------------------------------
        # Traffic
        # --------------------------------------------------------

        (
            total_vehicles,
            total_waiting
        ) = get_traffic_stats()

        # --------------------------------------------------------
        # Waiting time
        #
        # Approximation:
        # number of waiting vehicles × one second
        # --------------------------------------------------------

        total_waiting_time += (
            total_waiting
        )

        # --------------------------------------------------------
        # Queue
        # --------------------------------------------------------

        total_queue += (
            total_waiting
        )

        # --------------------------------------------------------
        # Vehicles
        # --------------------------------------------------------

        total_vehicle_count += (
            total_vehicles
        )

        metric_steps += 1

        # --------------------------------------------------------
        # Maximums
        # --------------------------------------------------------

        max_queue = max(
            max_queue,
            total_waiting
        )

        max_vehicles = max(
            max_vehicles,
            total_vehicles
        )

        # --------------------------------------------------------
        # Throughput
        #
        # getArrivedNumber() = vehicles that arrived
        # during the current simulation step.
        # --------------------------------------------------------

        try:

            arrived = (
                traci.simulation
                .getArrivedNumber()
            )

            throughput += arrived

        except Exception:

            pass

        # --------------------------------------------------------
        # Signal phase
        # --------------------------------------------------------

        phase = (
            traci.trafficlight
            .getPhase(TLS_ID)
        )

        if (
            previous_phase is not None
            and phase != previous_phase
        ):

            signal_switches += 1

        previous_phase = phase

        # --------------------------------------------------------
        # Progress output
        # --------------------------------------------------------

        if int(current_time) % 50 == 0:

            print(
                f"Time {int(current_time):3d}s | "
                f"Phase {phase} | "
                f"Vehicles {total_vehicles:3d} | "
                f"Waiting {total_waiting:3d} | "
                f"Throughput {throughput:3d}"
            )

    # ============================================================
    # FINAL METRICS
    # ============================================================

    final_vehicles, final_waiting = (
        get_traffic_stats()
    )

    if metric_steps > 0:

        average_waiting = (
            total_waiting_time
            / metric_steps
        )

        average_queue = (
            total_queue
            / metric_steps
        )

        average_vehicles = (
            total_vehicle_count
            / metric_steps
        )

    else:

        average_waiting = 0.0
        average_queue = 0.0
        average_vehicles = 0.0

    # ============================================================
    # RESULTS
    # ============================================================

    print()
    print("=" * 60)
    print("FIXED-TIME BASELINE RESULTS")
    print("=" * 60)

    print(
        f"Average waiting vehicles : "
        f"{average_waiting:.2f}"
    )

    print(
        f"Average queue length     : "
        f"{average_queue:.2f}"
    )

    print(
        f"Average vehicles         : "
        f"{average_vehicles:.2f}"
    )

    print(
        f"Total waiting time       : "
        f"{total_waiting_time:.2f}"
    )

    print(
        f"Throughput               : "
        f"{throughput}"
    )

    print(
        f"Final vehicles           : "
        f"{final_vehicles}"
    )

    print(
        f"Final waiting vehicles   : "
        f"{final_waiting}"
    )

    print(
        f"Maximum queue            : "
        f"{max_queue}"
    )

    print(
        f"Maximum vehicles         : "
        f"{max_vehicles}"
    )

    print(
        f"Signal phase changes     : "
        f"{signal_switches}"
    )

    print()
    print("=" * 60)

    # ------------------------------------------------------------
    # Close SUMO
    # ------------------------------------------------------------

    traci.close()

    print("SUMO closed.")
    print("=" * 60)


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()
