import os
import time
import random

import gymnasium as gym
import numpy as np
import traci
from gymnasium import spaces


class SumoTrafficEnvV3(gym.Env):

    """
    V3 Multi-Junction SUMO Traffic Environment

    Four traffic-light junctions:
        J1, J2, J3, J4

    Action:
        0 = North/South
        1 = East/West

    The PPO agent controls all junctions simultaneously.
    """

    metadata = {
        "render_modes": ["human"]
    }

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        gui=False,
        max_steps=120,
        seed=None
    ):

        super().__init__()

        # -----------------------------------------------------
        # Paths
        # -----------------------------------------------------

        self.project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.sumocfg = os.path.join(
            self.project_dir,
            "v3.sumocfg"
        )

        self.gui = gui

        self.sumo_binary = (
            "sumo-gui"
            if gui
            else "sumo"
        )

        # -----------------------------------------------------
        # Junctions
        # -----------------------------------------------------

        self.tls_ids = [
            "J1",
            "J2",
            "J3",
            "J4"
        ]

        self.num_junctions = len(
            self.tls_ids
        )

        # -----------------------------------------------------
        # Simulation
        # -----------------------------------------------------

        self.max_steps = int(
            max_steps
        )

        self.step_length = 1.0

        # -----------------------------------------------------
        # Signal parameters
        # -----------------------------------------------------

        self.min_green_time = 10.0

        self.max_green_time = 60.0

        self.yellow_time = 3.0

        # -----------------------------------------------------
        # Action space
        # -----------------------------------------------------

        self.action_space = spaces.MultiDiscrete(
            [2] * self.num_junctions
        )

        # -----------------------------------------------------
        # Observation
        #
        # Five values per junction:
        #
        # 0 NS vehicles
        # 1 EW vehicles
        # 2 NS waiting
        # 3 EW waiting
        # 4 total vehicles
        # -----------------------------------------------------

        self.features_per_junction = 5

        observation_size = (
            self.num_junctions *
            self.features_per_junction
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1000.0,
            shape=(observation_size,),
            dtype=np.float32
        )

        # -----------------------------------------------------
        # TraCI
        # -----------------------------------------------------

        self.connection = None

        self.port = None

        # -----------------------------------------------------
        # Runtime state
        # -----------------------------------------------------

        self.episode_step = 0

        self.simulation_time = 0.0

        self.random_seed = seed

        # -----------------------------------------------------
        # Signal state
        # -----------------------------------------------------

        self.current_direction = {
            tls: 0
            for tls in self.tls_ids
        }

        self.green_duration = {
            tls: 0.0
            for tls in self.tls_ids
        }

        self.last_phase = {
            tls: 0
            for tls in self.tls_ids
        }

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        self.total_reward = 0.0

        self.total_waiting = 0.0

        self.total_queue = 0.0

        self.total_vehicle_count = 0.0

        self.total_throughput = 0

        self.total_switches = 0

        self.metric_steps = 0

        self.previous_waiting = 0.0

        self._previous_vehicle_count = 0

    # =========================================================
    # START SUMO
    # =========================================================

    def _start_sumo(self):

        """
        Start SUMO and establish TraCI connection.

        IMPORTANT:
        The port is included exactly once in the command.
        We do not use waitBetweenRetries because the installed
        TraCI version does not support that argument.
        """

        # -----------------------------------------------------
        # Close previous SUMO
        # -----------------------------------------------------

        self._close_sumo()

        time.sleep(0.2)

        # -----------------------------------------------------
        # Check configuration
        # -----------------------------------------------------

        if not os.path.exists(
            self.sumocfg
        ):

            raise FileNotFoundError(
                "SUMO configuration not found:\n"
                f"{self.sumocfg}"
            )

        # -----------------------------------------------------
        # Retry startup
        # -----------------------------------------------------

        max_attempts = 10

        last_error = None

        for attempt in range(
            1,
            max_attempts + 1
        ):

            # -------------------------------------------------
            # Random port
            # -------------------------------------------------

            port = random.randint(
                20000,
                50000
            )

            # -------------------------------------------------
            # SUMO command
            # -------------------------------------------------

            command = [

                self.sumo_binary,

                "-c",
                self.sumocfg,

                "--remote-port",
                str(port),

                "--no-step-log",
                "true",

                "--duration-log.disable",
                "true",

                "--time-to-teleport",
                "-1"
            ]

            # -------------------------------------------------
            # Optional seed
            # -------------------------------------------------

            if self.random_seed is not None:

                command.extend([
                    "--seed",
                    str(
                        int(self.random_seed)
                        + self.episode_step
                        + attempt
                    )
                ])

            print(
                "[SUMO] Starting V3 episode "
                f"(attempt {attempt}/{max_attempts}, "
                f"port {port})"
            )

            try:

                # -------------------------------------------------
                # IMPORTANT
                #
                # Do NOT add:
                #
                # waitBetweenRetries
                #
                # Do NOT add:
                #
                # port=port
                #
                # The port is already inside command.
                # -------------------------------------------------

                traci.start(
                    command,
                    numRetries=10
                )

                time.sleep(0.1)

                # -------------------------------------------------
                # Verify
                # -------------------------------------------------

                if traci.isLoaded():

                    self.port = port

                    try:
                        self.connection = (
                            traci.getConnection()
                        )
                    except Exception:
                        self.connection = None

                    print(
                        "[SUMO] Connected successfully "
                        f"on port {port}"
                    )

                    return

                raise RuntimeError(
                    "SUMO started but TraCI "
                    "connection was not loaded."
                )

            except Exception as exc:

                last_error = exc

                print(
                    "[SUMO] Connection attempt failed: "
                    f"{exc}"
                )

                try:

                    if traci.isLoaded():
                        traci.close()

                except Exception:
                    pass

                self.connection = None

                self.port = None

                time.sleep(0.5)

        # -----------------------------------------------------
        # Failed
        # -----------------------------------------------------

        raise RuntimeError(
            "Could not establish SUMO/TraCI connection "
            f"after {max_attempts} attempts.\n"
            f"Last error: {last_error}"
        )

    # =========================================================
    # CLOSE SUMO
    # =========================================================

    def _close_sumo(self):

        try:

            if traci.isLoaded():
                traci.close()

        except Exception:
            pass

        self.connection = None

        self.port = None

    # =========================================================
    # CONNECTION CHECK
    # =========================================================

    def _is_connected(self):

        try:

            traci.getConnection()

            return True

        except Exception:

            return False

    # =========================================================
    # GET JUNCTION CONTROLLED EDGES
    # =========================================================

    def _get_junction_edges(
        self,
        tls_id
    ):

        try:

            controlled_links = (
                traci.trafficlight
                .getControlledLinks(
                    tls_id
                )
            )

        except Exception:

            return []

        edges = set()

        for group in controlled_links:

            if not group:
                continue

            for link in group:

                if not link:
                    continue

                try:

                    incoming_edge = (
                        link[0]
                    )

                    edges.add(
                        incoming_edge
                    )

                except Exception:

                    continue

        return list(edges)

    # =========================================================
    # JUNCTION STATISTICS
    # =========================================================

    def _get_junction_stats(
        self,
        tls_id
    ):

        edges = self._get_junction_edges(
            tls_id
        )

        ns_vehicles = 0

        ew_vehicles = 0

        ns_waiting = 0

        ew_waiting = 0

        total_vehicles = 0

        for edge_id in edges:

            try:

                vehicle_ids = (
                    traci.edge
                    .getLastStepVehicleIDs(
                        edge_id
                    )
                )

                vehicle_count = len(
                    vehicle_ids
                )

                waiting = (
                    traci.edge
                    .getLastStepHaltingNumber(
                        edge_id
                    )
                )

                # -------------------------------------------------
                # Determine orientation from edge geometry
                # -------------------------------------------------

                shape = (
                    traci.edge
                    .getShape(
                        edge_id
                    )
                )

                if len(shape) >= 2:

                    x1, y1 = shape[0]

                    x2, y2 = shape[-1]

                    dx = abs(
                        x2 - x1
                    )

                    dy = abs(
                        y2 - y1
                    )

                    if dy >= dx:

                        ns_vehicles += (
                            vehicle_count
                        )

                        ns_waiting += (
                            waiting
                        )

                    else:

                        ew_vehicles += (
                            vehicle_count
                        )

                        ew_waiting += (
                            waiting
                        )

                else:

                    ns_vehicles += (
                        vehicle_count
                    )

                    ns_waiting += (
                        waiting
                    )

                total_vehicles += (
                    vehicle_count
                )

            except Exception:

                continue

        return {

            "ns_vehicles":
                int(ns_vehicles),

            "ew_vehicles":
                int(ew_vehicles),

            "ns_waiting":
                int(ns_waiting),

            "ew_waiting":
                int(ew_waiting),

            "total_vehicles":
                int(total_vehicles)
        }

    # =========================================================
    # ALL JUNCTION STATISTICS
    # =========================================================

    def _get_all_stats(self):

        stats = {}

        total_waiting = 0

        total_vehicles = 0

        for tls_id in self.tls_ids:

            junction_stats = (
                self._get_junction_stats(
                    tls_id
                )
            )

            stats[tls_id] = (
                junction_stats
            )

            total_waiting += (

                junction_stats[
                    "ns_waiting"
                ]

                +

                junction_stats[
                    "ew_waiting"
                ]
            )

            total_vehicles += (
                junction_stats[
                    "total_vehicles"
                ]
            )

        # Queue is approximated using
        # stopped/waiting vehicles.

        total_queue = total_waiting

        return (
            stats,
            int(total_waiting),
            int(total_vehicles),
            int(total_queue)
        )

    # =========================================================
    # OBSERVATION
    # =========================================================

    def _get_observation(self):

        stats, _, _, _ = (
            self._get_all_stats()
        )

        observation = []

        for tls_id in self.tls_ids:

            data = stats[tls_id]

            observation.extend([

                float(
                    data["ns_vehicles"]
                ),

                float(
                    data["ew_vehicles"]
                ),

                float(
                    data["ns_waiting"]
                ),

                float(
                    data["ew_waiting"]
                ),

                float(
                    data["total_vehicles"]
                )
            ])

        return np.asarray(
            observation,
            dtype=np.float32
        )

    # =========================================================
    # HOLD CURRENT GREEN
    # =========================================================

    def _hold_phase(
        self,
        tls_id
    ):

        try:

            traci.trafficlight.setPhaseDuration(
                tls_id,
                100000.0
            )

        except Exception:

            pass

    # =========================================================
    # SET JUNCTION ACTION
    # =========================================================

    def _set_junction_action(
        self,
        tls_id,
        action
    ):

        action = int(action)

        if action not in (
            0,
            1
        ):

            raise ValueError(
                f"Invalid action {action} "
                f"for {tls_id}"
            )

        current_direction = (
            self.current_direction[
                tls_id
            ]
        )

        # -----------------------------------------------------
        # Same direction
        # -----------------------------------------------------

        if action == current_direction:

            self._hold_phase(
                tls_id
            )

            return False

        # -----------------------------------------------------
        # Minimum green
        # -----------------------------------------------------

        if (
            self.green_duration[
                tls_id
            ]
            <
            self.min_green_time
        ):

            self._hold_phase(
                tls_id
            )

            return False

        # -----------------------------------------------------
        # Get signal program
        # -----------------------------------------------------

        try:

            logics = (
                traci.trafficlight
                .getAllProgramLogics(
                    tls_id
                )
            )

            if not logics:

                return False

            phases = (
                logics[0].phases
            )

            if not phases:

                return False

            number_of_phases = len(
                phases
            )

            # -------------------------------------------------
            # Typical SUMO 4-phase program:
            #
            # phase 0 = NS green
            # phase 1 = NS yellow
            # phase 2 = EW green
            # phase 3 = EW yellow
            # -------------------------------------------------

            if number_of_phases >= 4:

                target_phase = (
                    0
                    if action == 0
                    else 2
                )

            else:

                target_phase = (
                    action
                    % number_of_phases
                )

            traci.trafficlight.setPhase(
                tls_id,
                target_phase
            )

            traci.trafficlight.setPhaseDuration(
                tls_id,
                100000.0
            )

        except Exception as exc:

            print(
                f"[SIGNAL] Could not change "
                f"{tls_id}: {exc}"
            )

            return False

        self.current_direction[
            tls_id
        ] = action

        self.green_duration[
            tls_id
        ] = 0.0

        self.total_switches += 1

        return True

    # =========================================================
    # UPDATE SIGNAL TIMERS
    # =========================================================

    def _update_signal_timers(self):

        for tls_id in self.tls_ids:

            try:

                phase = (
                    traci.trafficlight
                    .getPhase(
                        tls_id
                    )
                )

            except Exception:

                continue

            if (
                phase
                !=
                self.last_phase[
                    tls_id
                ]
            ):

                if phase in (
                    0,
                    2
                ):

                    self.green_duration[
                        tls_id
                    ] = 0.0

                self.last_phase[
                    tls_id
                ] = phase

            if phase in (
                0,
                2
            ):

                self.green_duration[
                    tls_id
                ] += self.step_length

    # =========================================================
    # REWARD
    # =========================================================

    def _calculate_reward(
        self,
        waiting,
        queue,
        throughput
    ):

        reward = (

            -(0.08 * waiting)

            -(0.12 * queue)

            +(0.05 * throughput)
        )

        return float(
            np.clip(
                reward,
                -100.0,
                100.0
            )
        )

    # =========================================================
    # RESET
    # =========================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(
            seed=seed
        )

        if seed is not None:

            self.random_seed = seed

        # -----------------------------------------------------
        # Start SUMO
        # -----------------------------------------------------

        self._start_sumo()

        # -----------------------------------------------------
        # Reset episode state
        # -----------------------------------------------------

        self.episode_step = 0

        self.simulation_time = 0.0

        # -----------------------------------------------------
        # Reset metrics
        # -----------------------------------------------------

        self.total_reward = 0.0

        self.total_waiting = 0.0

        self.total_queue = 0.0

        self.total_vehicle_count = 0.0

        self.total_throughput = 0

        self.total_switches = 0

        self.metric_steps = 0

        self.previous_waiting = 0.0

        self._previous_vehicle_count = 0

        # -----------------------------------------------------
        # Reset junction states
        # -----------------------------------------------------

        for tls_id in self.tls_ids:

            self.current_direction[
                tls_id
            ] = 0

            self.green_duration[
                tls_id
            ] = 0.0

            self.last_phase[
                tls_id
            ] = 0

            try:

                traci.trafficlight.setPhase(
                    tls_id,
                    0
                )

                traci.trafficlight.setPhaseDuration(
                    tls_id,
                    100000.0
                )

            except Exception as exc:

                print(
                    f"[SUMO] Warning initializing "
                    f"{tls_id}: {exc}"
                )

        # -----------------------------------------------------
        # Advance simulation
        # -----------------------------------------------------

        traci.simulationStep()

        self.simulation_time = (
            traci.simulation.getTime()
        )

        self._update_signal_timers()

        # -----------------------------------------------------
        # Initial statistics
        # -----------------------------------------------------

        (
            stats,
            waiting,
            vehicles,
            queue
        ) = self._get_all_stats()

        self.previous_waiting = (
            float(waiting)
        )

        self._previous_vehicle_count = (
            int(vehicles)
        )

        self.total_waiting += (
            waiting
        )

        self.total_queue += (
            queue
        )

        self.total_vehicle_count += (
            vehicles
        )

        self.metric_steps += 1

        # -----------------------------------------------------
        # Observation
        # -----------------------------------------------------

        observation = (
            self._get_observation()
        )

        info = {

            "junctions":
                stats,

            "total_waiting":
                waiting,

            "total_vehicles":
                vehicles,

            "total_queue":
                queue,

            "throughput":
                self.total_throughput,

            "switches":
                self.total_switches,

            "step":
                self.episode_step
        }

        return (
            observation,
            info
        )

    # =========================================================
    # STEP
    # =========================================================

    def step(
        self,
        action
    ):

        action = np.asarray(
            action,
            dtype=np.int64
        ).flatten()

        # -----------------------------------------------------
        # Validate action
        # -----------------------------------------------------

        if len(action) != (
            self.num_junctions
        ):

            raise ValueError(
                f"Expected "
                f"{self.num_junctions} actions, "
                f"got {len(action)}"
            )

        if not self._is_connected():

            raise RuntimeError(
                "TraCI is not connected. "
                "Call reset() first."
            )

        # -----------------------------------------------------
        # Apply actions
        # -----------------------------------------------------

        switched = 0

        for index, tls_id in enumerate(
            self.tls_ids
        ):

            if self._set_junction_action(
                tls_id,
                action[index]
            ):

                switched += 1

        # -----------------------------------------------------
        # Advance SUMO
        # -----------------------------------------------------

        traci.simulationStep()

        self.simulation_time = (
            traci.simulation.getTime()
        )

        self.episode_step += 1

        # -----------------------------------------------------
        # Update timers
        # -----------------------------------------------------

        self._update_signal_timers()

        # -----------------------------------------------------
        # Traffic statistics
        # -----------------------------------------------------

        (
            stats,
            waiting,
            vehicles,
            queue
        ) = self._get_all_stats()

        # -----------------------------------------------------
        # Throughput
        # -----------------------------------------------------

        previous_vehicles = (
            self._previous_vehicle_count
        )

        exited = max(
            0,
            int(previous_vehicles)
            -
            int(vehicles)
        )

        self.total_throughput += (
            int(exited)
        )

        self._previous_vehicle_count = (
            int(vehicles)
        )

        # -----------------------------------------------------
        # Reward
        # -----------------------------------------------------

        reward = (
            self._calculate_reward(
                waiting,
                queue,
                exited
            )
        )

        self.total_reward += reward

        # -----------------------------------------------------
        # Metrics
        # -----------------------------------------------------

        self.total_waiting += (
            waiting
        )

        self.total_queue += (
            queue
        )

        self.total_vehicle_count += (
            vehicles
        )

        self.metric_steps += 1

        self.previous_waiting = (
            float(waiting)
        )

        # -----------------------------------------------------
        # Observation
        # -----------------------------------------------------

        observation = (
            self._get_observation()
        )

        # -----------------------------------------------------
        # Termination
        # -----------------------------------------------------

        terminated = False

        truncated = (
            self.episode_step
            >=
            self.max_steps
        )

        # -----------------------------------------------------
        # Info
        # -----------------------------------------------------

        info = {

            "junctions":
                stats,

            "total_waiting":
                waiting,

            "total_vehicles":
                vehicles,

            "total_queue":
                queue,

            "throughput":
                self.total_throughput,

            "switches":
                self.total_switches,

            "step":
                self.episode_step,

            "switched":
                switched,

            "simulation_time":
                self.simulation_time
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self):

        self._close_sumo()

    # =========================================================
    # RENDER
    # =========================================================

    def render(self):

        if not self._is_connected():

            return

        try:

            print(
                f"Time: "
                f"{self.simulation_time:.1f} | "
                f"Waiting: "
                f"{self.previous_waiting:.0f} | "
                f"Throughput: "
                f"{self.total_throughput} | "
                f"Switches: "
                f"{self.total_switches}"
            )

        except Exception:

            pass
