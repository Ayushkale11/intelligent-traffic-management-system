import os
import traci

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CONFIG = os.path.join(BASE_DIR, "v1.sumocfg")

traci.start([
    "sumo",
    "-c",
    CONFIG,
    "--no-warnings",
    "--no-step-log"
])

print("===================================")
print("SUMO SIGNAL TEST")
print("===================================")

last_phase = -1

for step in range(150):

    traci.simulationStep()

    phase = traci.trafficlight.getPhase("J")
    phase_time = traci.trafficlight.getNextSwitch("J") - traci.simulation.getTime()

    if phase != last_phase:

        print(
            f"Time {traci.simulation.getTime():6.1f}s | "
            f"Phase {phase} | "
            f"Time remaining {phase_time:.1f}s"
        )

        last_phase = phase

traci.close()

print()
print("===================================")
print("SIGNAL TEST COMPLETE")
print("===================================")
