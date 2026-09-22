import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import traci


class SumoTrafficEnvV3(gym.Env):

    metadata = {"render_modes": ["human"]}

    JUNCTIONS = ["J1", "J2", "J3", "J4"]

    # =========================================================
    # VERIFIED V3 NETWORK EDGES
    # =========================================================

    JUNCTION_EDGES = {

        "J1": {
            "north": ["N1_J1"],
            "south": ["S1_J1"],
            "east":  ["J2_J1"],
            "west":  ["W1_J1"],
        },

        "J2": {
            "north": ["N2_J2"],
            "south": ["S2_J2"],
            "east":  ["E1_J2"],
            "west":  ["J1_J2"],
        },

        "J3": {
            "north": ["J4_J3"],
            "south": ["S1_J3"],
            "east":  ["J3_J4"],
            "west":  ["W2_J3"],
        },

        "J4": {
            "north": ["E2_J4"],
            "south": ["S2_J4"],
            "east":  ["J4_J3"],
            "west":  ["J3_J4"],
        },
    }

    # =========================================================
    # INIT
    # =========================================================

    def __init__(
        self,
        project_dir=None,
        gui=False,
        max_steps=120,
        step_length=1.0,
    ):

        super().__init__()

        if project_dir is None:
            project_dir = os.path.dirname(
                os.path.abspath(__file__)
            )

        self.project_dir = project_dir

        self.gui = gui

        self.max_steps = int(max_steps)

        self.step_length = float(step_length)

        self.sumocfg = os.path.join(
            self.project_dir,
            "v3.sumocfg"
        )

        if not os.path.exists(self.sumocfg):

            raise FileNotFoundError(
                f"SUMO configuration not found:\n"
                f"{self.sumocfg}"
            )

        # -----------------------------------------------------
        # 4 junctions × 8 values
        #
        # 4 queue values
        # 4 vehicle values
        #
        # = 32
        # -----------------------------------------------------

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(32,),
            dtype=np.float32,
        )

        # One phase decision for each junction

        self.action_space = spaces.MultiDiscrete(
            [2, 2, 2, 2]
        )

        self.step_count = 0

        self.total_throughput = 0

        self.previous_vehicle_ids = set()

        self.switch_counts = {
            junction: 0
            for junction in self.JUNCTIONS
        }

        self.current_phases = {
            junction: 0
            for junction in self.JUNCTIONS
        }

    # =========================================================
    # CLOSE CONNECTION
    # =========================================================

    def _close_connection(self):

        try:

            if traci.isLoaded():

                traci.close()

        except Exception:

            pass

        time.sleep(0.2)

    # =========================================================
    # START SUMO
    # =========================================================

    def _start_sumo(self):

        self._close_connection()

        binary = (
            "sumo-gui"
            if self.gui
            else "sumo"
        )

        command = [

            binary,

            "-c",
            self.sumocfg,

            "--step-length",
            str(self.step_length),

            "--no-step-log",
            "true",

            "--duration-log.disable",
            "true",

            "--time-to-teleport",
            "-1",
        ]

        print(
            "[SUMO] Starting V3 episode"
        )

        print(
            "[SUMO] Using TraCI-managed port"
        )

        try:

            # -------------------------------------------------
            # IMPORTANT
            #
            # DO NOT put --remote-port in command.
            #
            # TraCI creates the port itself.
            # -------------------------------------------------

            traci.start(
                command,
                numRetries=20,
                label="v3"
            )

            # -------------------------------------------------
            # Switch to the V3 connection explicitly.
            # -------------------------------------------------

            traci.switch("v3")

            print(
                "[SUMO] Connected successfully"
            )

        except Exception as e:

            self._close_connection()

            raise RuntimeError(
                f"Could not establish "
                f"SUMO/TraCI connection:\n{e}"
            )

    # =========================================================
    # QUEUE
    # =========================================================

    def _get_queue(self, edge):

        try:

            return int(
                traci.edge.getLastStepHaltingNumber(
                    edge
                )
            )

        except Exception:

            return 0

    # =========================================================
    # VEHICLES ON EDGE
    # =========================================================

    def _get_edge_vehicle_count(self, edge):

        try:

            return len(
                traci.edge.getLastStepVehicleIDs(
                    edge
                )
            )

        except Exception:

            return 0

    # =========================================================
    # JUNCTION OBSERVATION
    # =========================================================

    def _get_junction_observation(
        self,
        junction
    ):

        edges = self.JUNCTION_EDGES[junction]

        observation = []

        directions = [
            "north",
            "south",
            "east",
            "west",
        ]

        # -----------------------------------------------------
        # QUEUES
        # -----------------------------------------------------

        for direction in directions:

            total = 0

            for edge in edges[direction]:

                total += self._get_queue(edge)

            observation.append(total)

        # -----------------------------------------------------
        # VEHICLES
        # -----------------------------------------------------

        for direction in directions:

            total = 0

            for edge in edges[direction]:

                total += self._get_edge_vehicle_count(
                    edge
                )

            observation.append(total)

        return observation

    # =========================================================
    # COMPLETE OBSERVATION
    # =========================================================

    def _get_observation(self):

        observation = []

        for junction in self.JUNCTIONS:

            observation.extend(
                self._get_junction_observation(
                    junction
                )
            )

        return np.asarray(
            observation,
            dtype=np.float32
        )

    # =========================================================
    # TOTAL WAITING
    # =========================================================

    def _get_total_waiting(self):

        total = 0

        for junction in self.JUNCTIONS:

            edges = self.JUNCTION_EDGES[junction]

            for direction in edges:

                for edge in edges[direction]:

                    total += self._get_queue(
                        edge
                    )

        return int(total)

    # =========================================================
    # TOTAL VEHICLES
    # =========================================================

    def _get_total_vehicles(self):

        vehicle_ids = set()

        for junction in self.JUNCTIONS:

            edges = self.JUNCTION_EDGES[junction]

            for direction in edges:

                for edge in edges[direction]:

                    try:

                        ids = (
                            traci.edge
                            .getLastStepVehicleIDs(
                                edge
                            )
                        )

                        vehicle_ids.update(ids)

                    except Exception:

                        pass

        return len(vehicle_ids)

    # =========================================================
    # THROUGHPUT
    # =========================================================

    def _update_throughput(self):

        try:

            current_ids = set(
                traci.vehicle.getIDList()
            )

            exited = (
                self.previous_vehicle_ids
                - current_ids
            )

            self.total_throughput += len(
                exited
            )

            self.previous_vehicle_ids = (
                current_ids
            )

        except Exception:

            pass

    # =========================================================
    # SET JUNCTION PHASE
    # =========================================================

    def _set_phase(
        self,
        junction,
        phase
    ):

        phase = int(phase)

        try:

            current_phase = (
                traci.trafficlight.getPhase(
                    junction
                )
            )

            if current_phase != phase:

                traci.trafficlight.setPhase(
                    junction,
                    phase
                )

                self.switch_counts[
                    junction
                ] += 1

            self.current_phases[
                junction
            ] = phase

        except Exception as e:

            print(
                f"[WARNING] Could not set "
                f"{junction} phase: {e}"
            )

    # =========================================================
    # APPLY ACTION
    # =========================================================

    def _apply_action(self, action):

        action = np.asarray(
            action
        ).flatten()

        if len(action) != 4:

            raise ValueError(
                "Expected action "
                "[J1, J2, J3, J4]"
            )

        for i, junction in enumerate(
            self.JUNCTIONS
        ):

            self._set_phase(
                junction,
                action[i]
            )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        self._close_connection()

        self.step_count = 0

        self.total_throughput = 0

        self.previous_vehicle_ids = set()

        self.switch_counts = {
            junction: 0
            for junction in self.JUNCTIONS
        }

        self.current_phases = {
            junction: 0
            for junction in self.JUNCTIONS
        }

        self._start_sumo()

        # Initial simulation step

        traci.simulationStep()

        try:

            self.previous_vehicle_ids = set(
                traci.vehicle.getIDList()
            )

        except Exception:

            self.previous_vehicle_ids = set()

        observation = self._get_observation()

        info = {

            "step": 0,

            "total_waiting":
                self._get_total_waiting(),

            "total_queue":
                self._get_total_waiting(),

	    "total_vehicles":
                self._get_total_vehicles(),

            "throughput":
                self.total_throughput,

        }

        return observation, info

    # =========================================================
    # STEP
    # =========================================================

    def step(self, action):

        if not traci.isLoaded():

            raise RuntimeError(
                "SUMO/TraCI is not connected."
            )

        # -----------------------------------------------------
        # Apply four-junction action
        # -----------------------------------------------------

        self._apply_action(action)

        # -----------------------------------------------------
        # Advance SUMO
        # -----------------------------------------------------

        traci.simulationStep()

        self.step_count += 1

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        waiting = self._get_total_waiting()

        vehicles = self._get_total_vehicles()

        self._update_throughput()

        # -----------------------------------------------------
        # Reward
        #
        # Primary objective:
        # minimize waiting vehicles
        # -----------------------------------------------------

        switches = sum(
            self.switch_counts.values()
        )

        reward = (
            -float(waiting)
            - 0.01 * switches
        )

        # -----------------------------------------------------
        # SUMO termination
        # -----------------------------------------------------

        terminated = False

        try:

            if (
                traci.simulation
                .getMinExpectedNumber()
                <= 0
            ):

                terminated = True

        except Exception:

            pass

        # -----------------------------------------------------
        # Time limit
        # -----------------------------------------------------

        truncated = (
            self.step_count
            >= self.max_steps
        )

        observation = self._get_observation()

        info = {

            "step":
                self.step_count,

            "total_waiting":
                waiting,

            "total_vehicles":
                vehicles,

            "total_queue":
                waiting,

            "throughput":
                self.total_throughput,

            "switches":
                int(switches),

            "J1_phase":
                self.current_phases["J1"],

            "J2_phase":
                self.current_phases["J2"],

            "J3_phase":
                self.current_phases["J3"],

            "J4_phase":
                self.current_phases["J4"],
        }

        if terminated or truncated:

            self._close_connection()

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info
        )

    # =========================================================
    # RENDER
    # =========================================================

    def render(self):

        pass

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self._close_connection()
