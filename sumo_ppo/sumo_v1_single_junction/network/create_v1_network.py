import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NODE_FILE = os.path.join(BASE_DIR, "v1.nod.xml")
EDGE_FILE = os.path.join(BASE_DIR, "v1.edg.xml")
NET_FILE = os.path.join(BASE_DIR, "v1.net.xml")

# ---------------------------------------------------------
# Nodes
# ---------------------------------------------------------
# Single 4-way intersection.
# Each approach has an incoming and outgoing edge.
# Internal junction is controlled by SUMO traffic lights.

nodes = """\
<nodes>
    <node id="N" x="0" y="300" type="priority"/>
    <node id="S" x="0" y="-300" type="priority"/>
    <node id="E" x="300" y="0" type="priority"/>
    <node id="W" x="-300" y="0" type="priority"/>

    <node id="J" x="0" y="0" type="traffic_light"/>
</nodes>
"""

# ---------------------------------------------------------
# Edges
# ---------------------------------------------------------
# 2 lanes per road.
# Speed = 13.89 m/s (~50 km/h)

edges = """\
<edges>

    <!-- North -->
    <edge id="N_J" from="N" to="J"
          numLanes="2" speed="13.89"/>
    <edge id="J_N" from="J" to="N"
          numLanes="2" speed="13.89"/>

    <!-- South -->
    <edge id="S_J" from="S" to="J"
          numLanes="2" speed="13.89"/>
    <edge id="J_S" from="J" to="S"
          numLanes="2" speed="13.89"/>

    <!-- East -->
    <edge id="E_J" from="E" to="J"
          numLanes="2" speed="13.89"/>
    <edge id="J_E" from="J" to="E"
          numLanes="2" speed="13.89"/>

    <!-- West -->
    <edge id="W_J" from="W" to="J"
          numLanes="2" speed="13.89"/>
    <edge id="J_W" from="J" to="W"
          numLanes="2" speed="13.89"/>

</edges>
"""

with open(NODE_FILE, "w") as f:
    f.write(nodes)

with open(EDGE_FILE, "w") as f:
    f.write(edges)

print("Generating SUMO network...")

subprocess.run(
    [
        "netconvert",
        "--node-files", NODE_FILE,
        "--edge-files", EDGE_FILE,
        "--output-file", NET_FILE,

        # Traffic-light settings
        "--tls.default-type", "static",

        # Junction geometry
        "--junctions.corner-detail", "5",
        "--junctions.internal-link-detail", "5",

        # Lane connections
        "--default.lanenumber", "2",
        "--default.speed", "13.89",

        # Keep traffic light controlled junction
        "--no-turnarounds",
    ],
    check=True
)

print()
print("========================================")
print("SUMO V1 NETWORK GENERATED")
print("========================================")
print(f"Network: {NET_FILE}")
print()
