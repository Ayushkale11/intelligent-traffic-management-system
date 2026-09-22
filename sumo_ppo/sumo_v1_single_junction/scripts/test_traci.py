import os
import sys
import traci

# ---------------------------------------------------------
# SUMO configuration
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMO_CONFIG = os.path.join(BASE_DIR, "v1.sumocfg")

SUMO_BINARY = "sumo"

# ---------------------------------------------------------
# Start SUMO
# ---------------------------------------------------------

sumo_cmd = [
    SUMO_BINARY,
    "-c",
    SUMO_CONFIG,
    "--start",
    "--quit-on-end",
]

print("Starting SUMO...")
traci.start(sumo_cmd)

print("TraCI connection established.")

# ---------------------------------------------------------
# Traffic light
# ---------------------------------------------------------

TLS_ID = "J"

# Incoming edges
INCOMING_EDGES = [
    "N_J",
    "S_J",
    "E_J",
    "W_J",
]

# ---------------------------------------------------------
# Simulation
# ---------------------------------------------------------

TOTAL_STEPS = 600

total_waiting = 0

try:

    for step in range(TOTAL_STEPS):

        traci.simulationStep()

        # ---------------------------------------------
        # Vehicle counts
        # ---------------------------------------------

        vehicle_counts = {}

        for edge in INCOMING_EDGES:

            vehicle_counts[edge] = traci.edge.getLastStepVehicleNumber(edge)

        # ---------------------------------------------
        # Waiting vehicle counts
        # ---------------------------------------------

        waiting_counts = {}

        for edge in INCOMING_EDGES:

            waiting_counts[edge] = traci.edge.getLastStepHaltingNumber(edge)

        # ---------------------------------------------
        # Traffic light phase
        # ---------------------------------------------

        current_phase = traci.trafficlight.getPhase(TLS_ID)

        # ---------------------------------------------
        # Total waiting
        # ---------------------------------------------

        current_waiting = sum(waiting_counts.values())

        total_waiting += current_waiting

        # ---------------------------------------------
        # Print every 50 steps
        # ---------------------------------------------

        if step % 50 == 0:

            print(
                f"Step {step:3d} | "
                f"Phase {current_phase} | "
                f"Vehicles {sum(vehicle_counts.values()):3d} | "
                f"Waiting {current_waiting:3d}"
            )

            print(
                "   Vehicles:",
                vehicle_counts
            )

            print(
                "   Waiting :",
                waiting_counts
            )

        # ---------------------------------------------
        # Simple traffic-light test
        # ---------------------------------------------
        #
        # Change phase every 30 simulation seconds.
        #
        if step > 0 and step % 30 == 0:

            next_phase = (current_phase + 1) % 4

            traci.trafficlight.setPhase(
                TLS_ID,
                next_phase
            )

finally:

    traci.close()

# ---------------------------------------------------------
# Final statistics
# ---------------------------------------------------------

average_waiting = total_waiting / TOTAL_STEPS

print()
print("======================================")
print("TRACI TEST COMPLETE")
print("======================================")
print(f"Average waiting vehicles: {average_waiting:.2f}")
