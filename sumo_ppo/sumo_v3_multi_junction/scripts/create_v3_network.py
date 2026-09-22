import os
import subprocess
import sys


# ============================================================
# V3 MULTI-JUNCTION NETWORK
#
# 2 x 2 grid
#
#       N
#       |
#    J1 --- J2
#    |       |
#    |       |
#    J3 --- J4
#       |
#       S
#
# Each junction has its own traffic light.
# ============================================================


PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


NODES_FILE = os.path.join(
    PROJECT_DIR,
    "v3.nod.xml"
)

EDGES_FILE = os.path.join(
    PROJECT_DIR,
    "v3.edg.xml"
)

NET_FILE = os.path.join(
    PROJECT_DIR,
    "v3.net.xml"
)


# ============================================================
# CREATE NODES
# ============================================================

nodes = """<?xml version="1.0" encoding="UTF-8"?>

<nodes>

    <!-- External nodes -->

    <node id="N1" x="0.0" y="400.0" type="priority"/>
    <node id="N2" x="400.0" y="400.0" type="priority"/>

    <node id="S1" x="0.0" y="-400.0" type="priority"/>
    <node id="S2" x="400.0" y="-400.0" type="priority"/>

    <node id="W1" x="-400.0" y="0.0" type="priority"/>
    <node id="W2" x="-400.0" y="-300.0" type="priority"/>

    <node id="E1" x="800.0" y="0.0" type="priority"/>
    <node id="E2" x="800.0" y="-300.0" type="priority"/>


    <!-- Four signalized junctions -->

    <node id="J1" x="0.0" y="0.0"
          type="traffic_light"/>

    <node id="J2" x="400.0" y="0.0"
          type="traffic_light"/>

    <node id="J3" x="0.0" y="-300.0"
          type="traffic_light"/>

    <node id="J4" x="400.0" y="-300.0"
          type="traffic_light"/>

</nodes>
"""


# ============================================================
# CREATE EDGES
# ============================================================

edges = """<?xml version="1.0" encoding="UTF-8"?>

<edges>

    <!-- ================================================== -->
    <!-- NORTH / SOUTH APPROACHES -->
    <!-- ================================================== -->

    <!-- J1 -->

    <edge id="N1_J1"
          from="N1"
          to="J1"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J1_N1"
          from="J1"
          to="N1"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="S1_J1"
          from="S1"
          to="J1"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J1_S1"
          from="J1"
          to="S1"
          priority="1"
          numLanes="2"
          speed="13.9"/>


    <!-- J2 -->

    <edge id="N2_J2"
          from="N2"
          to="J2"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J2_N2"
          from="J2"
          to="N2"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="S2_J2"
          from="S2"
          to="J2"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J2_S2"
          from="J2"
          to="S2"
          priority="1"
          numLanes="2"
          speed="13.9"/>


    <!-- J3 -->

    <edge id="S1_J3"
          from="S1"
          to="J3"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J3_S1"
          from="J3"
          to="S1"
          priority="1"
          numLanes="2"
          speed="13.9"/>


    <!-- J4 -->

    <edge id="S2_J4"
          from="S2"
          to="J4"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J4_S2"
          from="J4"
          to="S2"
          priority="1"
          numLanes="2"
          speed="13.9"/>


    <!-- ================================================== -->
    <!-- HORIZONTAL CONNECTIONS -->
    <!-- ================================================== -->

    <!-- J1 <-> J2 -->

    <edge id="J1_J2"
          from="J1"
          to="J2"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J2_J1"
          from="J2"
          to="J1"
          priority="1"
          numLanes="2"
          speed="13.9"/>


    <!-- J3 <-> J4 -->

    <edge id="J3_J4"
          from="J3"
          to="J4"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J4_J3"
          from="J4"
          to="J3"
          priority="1"
          numLanes="2"
          speed="13.9"/>


    <!-- ================================================== -->
    <!-- WEST APPROACHES -->
    <!-- ================================================== -->

    <edge id="W1_J1"
          from="W1"
          to="J1"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J1_W1"
          from="J1"
          to="W1"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="W2_J3"
          from="W2"
          to="J3"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J3_W2"
          from="J3"
          to="W2"
          priority="1"
          numLanes="2"
          speed="13.9"/>


    <!-- ================================================== -->
    <!-- EAST APPROACHES -->
    <!-- ================================================== -->

    <edge id="J2_E1"
          from="J2"
          to="E1"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="E1_J2"
          from="E1"
          to="J2"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="J4_E2"
          from="J4"
          to="E2"
          priority="1"
          numLanes="2"
          speed="13.9"/>

    <edge id="E2_J4"
          from="E2"
          to="J4"
          priority="1"
          numLanes="2"
          speed="13.9"/>

</edges>
"""


# ============================================================
# WRITE FILES
# ============================================================

print("=" * 60)
print("CREATING V3 MULTI-JUNCTION NETWORK")
print("=" * 60)


with open(NODES_FILE, "w") as f:
    f.write(nodes)


with open(EDGES_FILE, "w") as f:
    f.write(edges)


print()
print("Created:")
print("  v3.nod.xml")
print("  v3.edg.xml")


# ============================================================
# RUN NETCONVERT
# ============================================================

print()
print("Running netconvert...")


command = [
    "netconvert",

    "--node-files",
    NODES_FILE,

    "--edge-files",
    EDGES_FILE,

    "--output-file",
    NET_FILE,

    "--junctions.join",
    "false",

    "--tls.guess",
    "true",

    "--tls.default-type",
    "static",

    "--no-turnarounds",
]


result = subprocess.run(
    command,
    capture_output=True,
    text=True
)


print(result.stdout)

if result.stderr:
    print(result.stderr)


if result.returncode != 0:

    print()
    print("NETWORK CREATION FAILED")

    sys.exit(1)


print()
print("=" * 60)
print("V3 NETWORK CREATED SUCCESSFULLY")
print("=" * 60)

print()
print("Network:")
print(NET_FILE)

print()
print("Traffic-light junctions:")
print("  J1")
print("  J2")
print("  J3")
print("  J4")
