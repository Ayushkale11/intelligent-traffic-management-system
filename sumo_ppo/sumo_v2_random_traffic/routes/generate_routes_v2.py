import os
import random
import xml.etree.ElementTree as ET


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "routes",
    "v2.rou.xml"
)

SIMULATION_END = 600


# ============================================================
# TRAFFIC SCENARIOS
# ============================================================

SCENARIOS = {

    "low": {
        "vehicles": 220,
        "weights": [0.25, 0.25, 0.25, 0.25],
        "seed": 101,
    },

    "medium": {
        "vehicles": 360,
        "weights": [0.25, 0.25, 0.25, 0.25],
        "seed": 202,
    },

    "high": {
        "vehicles": 500,
        "weights": [0.25, 0.25, 0.25, 0.25],
        "seed": 303,
    },

    "ns_heavy": {
        "vehicles": 400,
        "weights": [0.40, 0.40, 0.10, 0.10],
        "seed": 404,
    },

    "ew_heavy": {
        "vehicles": 400,
        "weights": [0.10, 0.10, 0.40, 0.40],
        "seed": 505,
    },

    "random": {
        "vehicles": 450,
        "weights": [0.32, 0.18, 0.27, 0.23],
        "seed": 606,
    },
}


# ============================================================
# ACTUAL V1/V2 NETWORK ROUTES
# ============================================================

ROUTES = {

    "N_S": [
        "N_J",
        "J_S"
    ],

    "S_N": [
        "S_J",
        "J_N"
    ],

    "E_W": [
        "E_J",
        "J_W"
    ],

    "W_E": [
        "W_J",
        "J_E"
    ],
}


# ============================================================
# GENERATE SCENARIO
# ============================================================

def generate_scenario(name):

    scenario = SCENARIOS[name]

    random.seed(
        scenario["seed"]
    )

    root = ET.Element("routes")

    # --------------------------------------------------------
    # Vehicle type
    # --------------------------------------------------------

    ET.SubElement(
        root,
        "vType",
        {
            "id": "car",
            "accel": "2.6",
            "decel": "4.5",
            "sigma": "0.5",
            "length": "5.0",
            "minGap": "2.5",
            "maxSpeed": "13.9",
        },
    )

    # --------------------------------------------------------
    # Routes
    # --------------------------------------------------------

    for route_id, edges in ROUTES.items():

        ET.SubElement(
            root,
            "route",
            {
                "id": route_id,
                "edges": " ".join(edges),
            },
        )

    route_names = list(
        ROUTES.keys()
    )

    # --------------------------------------------------------
    # CREATE VEHICLES FIRST
    # --------------------------------------------------------

    vehicles = []

    for i in range(
        scenario["vehicles"]
    ):

        route_id = random.choices(
            route_names,
            weights=scenario["weights"],
            k=1,
        )[0]

        depart = random.uniform(
            0,
            SIMULATION_END - 1
        )

        vehicles.append(
            {
                "id": f"{name}_{i}",
                "type": "car",
                "route": route_id,
                "depart": depart,
            }
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # SORT BY DEPARTURE TIME
    # --------------------------------------------------------

    vehicles.sort(
        key=lambda vehicle:
        vehicle["depart"]
    )

    # --------------------------------------------------------
    # WRITE VEHICLES
    # --------------------------------------------------------

    for vehicle in vehicles:

        ET.SubElement(
            root,
            "vehicle",
            {
                "id": vehicle["id"],
                "type": vehicle["type"],
                "route": vehicle["route"],
                "depart": f"{vehicle['depart']:.2f}",
                "departLane": "best",
                "departSpeed": "max",
            },
        )

    # --------------------------------------------------------
    # WRITE XML
    # --------------------------------------------------------

    tree = ET.ElementTree(root)

    ET.indent(
        tree,
        space="    "
    )

    tree.write(
        OUTPUT_FILE,
        encoding="UTF-8",
        xml_declaration=True,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("V2 TRAFFIC SCENARIO GENERATED")
    print("=" * 60)

    print(
        f"Scenario : {name}"
    )

    print(
        f"Vehicles : {scenario['vehicles']}"
    )

    print(
        f"Seed     : {scenario['seed']}"
    )

    print()
    print("Route distribution:")

    for route, weight in zip(
        route_names,
        scenario["weights"]
    ):

        print(
            f"  {route:<5} : "
            f"{weight * 100:.1f}%"
        )

    print()
    print(
        f"Saved to:\n{OUTPUT_FILE}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SUMO V2 RANDOM TRAFFIC GENERATOR")
    print("=" * 60)

    names = list(
        SCENARIOS.keys()
    )

    for i, name in enumerate(
        names,
        start=1
    ):

        print(
            f"{i}. {name}"
        )

    print()

    choice = input(
        "Select scenario [1-6]: "
    ).strip()

    try:

        index = int(choice) - 1

        if index < 0 or index >= len(names):
            raise ValueError

        name = names[index]

    except ValueError:

        print(
            "Invalid scenario."
        )

        return

    generate_scenario(
        name
    )


if __name__ == "__main__":
    main()
