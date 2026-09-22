import os
import gymnasium as gym
import numpy as np
import traci


class SumoTrafficEnv(gym.Env):
    """
    SUMO V1 - Single Junction Traffic RL Environment

    Junction:
        J

    Actions:
        0 = North/South Green
        1 = East/West Green

    The agent decides WHEN to switch.
    SUMO handles:
        - Vehicle movement
        - Car-following
        - Lane changing
        - Yellow transition
        - Route completion
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        use_gui=False,
        max_episode_steps=120,
        sim_steps_per_action=5,
    ):
        super().__init__()

        # =========================================================
        # PATH
        # =========================================================

        self.project_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.sumocfg = os.path.join(
            self.project_dir,
            "v1.sumocfg"
        )

        # =========================================================
        # SUMO
        # =========================================================

        self.use_gui = use_gui

        if self.use_gui:
            self.sumo_binary = "sumo-gui"
        else:
            self.sumo_binary = "sumo"

        # =========================================================
        # SIMULATION SETTINGS
        # =========================================================

        self.max_episode_steps = max_episode_steps

        # Number of SUMO seconds simulated per RL action.
        self.sim_steps_per_action = sim_steps_per_action

        # =========================================================
        # TRAFFIC LIGHT
        # =========================================================

        self.tls_id = "J"

        # SUMO phases:
        #
        # Phase 0 = NS Green
        # Phase 1 = NS Yellow
        # Phase 2 = EW Green
        # Phase 3 = EW Yellow

        self.NS_GREEN = 0
        self.NS_YELLOW = 1

        self.EW_GREEN = 2
        self.EW_YELLOW = 3

        # =========================================================
        # INCOMING EDGES
        # =========================================================

        self.incoming_edges = [
            "N_J",
            "S_J",
            "E_J",
            "W_J",
        ]

        # =========================================================
        # ACTION SPACE
        # =========================================================

        # 0 = NS
        # 1 = EW

        self.action_space = gym.spaces.Discrete(2)

        # =========================================================
        # OBSERVATION SPACE
        # =========================================================

        # Observation:
        #
        # [0]  North vehicle count
        # [1]  South vehicle count
        # [2]  East vehicle count
        # [3]  West vehicle count
        #
        # [4]  North waiting
        # [5]  South waiting
        # [6]  East waiting
        # [7]  West waiting
        #
        # [8]  Current green direction
        # [9]  Current green duration
        #
        # Everything normalized to [0, 1].

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(10,),
            dtype=np.float32,
        )

        # =========================================================
        # RL STATE
        # =========================================================

        self.episode_step = 0

        self.current_phase = self.NS_GREEN

        # 0 = NS
        # 1 = EW
        self.green_direction = 0

        self.green_duration = 0.0

        self.previous_waiting = 0.0
        self.previous_vehicles = 0.0

        # =========================================================
        # METRICS
        # =========================================================

        self.total_throughput = 0

        self.switch_count = 0

        self.total_waiting_time = 0.0

        self.total_vehicle_count = 0.0

        self.total_queue = 0.0

        self.metric_steps = 0

        # =========================================================
        # TIME
        # =========================================================

        self.last_sim_time = 0.0

        # =========================================================
        # SIGNAL CONTROL LIMITS
        # =========================================================

        # Minimum amount of green time before switching.

        self.min_green_time = 10.0

        # Maximum green duration.

        self.max_green_time = 60.0

    # =============================================================
    # TRACI CONNECTION CHECK
    # =============================================================

    def _is_traci_connected(self):
        """
        Safely determine whether a TraCI connection exists.

        Some TraCI versions do not provide traci.isConnected(),
        so we use getConnection() instead.
        """

        try:
            traci.getConnection()
            return True

        except Exception:
            return False

    # =============================================================
    # START SUMO
    # =============================================================

    def _start_sumo(self):

        # Close an existing connection if necessary.

        if self._is_traci_connected():

            try:
                traci.close()

            except Exception:
                pass

        command = [
            self.sumo_binary,
            "-c",
            self.sumocfg,

            "--no-step-log",
            "true",

            "--duration-log.disable",
            "true",

            # Never teleport vehicles.
            "--time-to-teleport",
            "-1",
        ]

        traci.start(command)

    # =============================================================
    # GET DIRECTION FROM PHASE
    # =============================================================

    def _get_direction_from_phase(self, phase):

        if phase in [
            self.NS_GREEN,
            self.NS_YELLOW,
        ]:

            return 0

        if phase in [
            self.EW_GREEN,
            self.EW_YELLOW,
        ]:

            return 1

        return self.green_direction

    # =============================================================
    # GET TRAFFIC STATISTICS
    # =============================================================

    def _get_traffic_stats(self):

        vehicle_counts = []

        waiting_counts = []

        for edge in self.incoming_edges:

            try:

                vehicles = traci.edge.getLastStepVehicleNumber(
                    edge
                )

                waiting = traci.edge.getLastStepHaltingNumber(
                    edge
                )

            except Exception:

                vehicles = 0

                waiting = 0

            vehicle_counts.append(
                int(vehicles)
            )

            waiting_counts.append(
                int(waiting)
            )

        total_vehicles = sum(
            vehicle_counts
        )

        total_waiting = sum(
            waiting_counts
        )

        return (
            vehicle_counts,
            waiting_counts,
            total_vehicles,
            total_waiting,
        )

    # =============================================================
    # GET OBSERVATION
    # =============================================================

    def _get_observation(self):

        vehicle_counts, waiting_counts, _, _ = (
            self._get_traffic_stats()
        )

        # ---------------------------------------------------------
        # Vehicle normalization
        # ---------------------------------------------------------

        normalized_vehicles = [
            min(
                vehicle / 50.0,
                1.0
            )
            for vehicle in vehicle_counts
        ]

        # ---------------------------------------------------------
        # Waiting normalization
        # ---------------------------------------------------------

        normalized_waiting = [
            min(
                waiting / 30.0,
                1.0
            )
            for waiting in waiting_counts
        ]

        # ---------------------------------------------------------
        # Green duration normalization
        # ---------------------------------------------------------

        normalized_green_duration = min(
            self.green_duration
            / self.max_green_time,
            1.0
        )

        observation = np.array(
            normalized_vehicles
            + normalized_waiting
            + [
                float(self.green_direction),
                normalized_green_duration,
            ],
            dtype=np.float32,
        )

        return observation

    # =============================================================
    # UPDATE SIGNAL STATE
    # =============================================================

    def _update_signal_state(self):

        try:

            phase = traci.trafficlight.getPhase(
                self.tls_id
            )

        except Exception:

            return

        previous_phase = self.current_phase

        self.current_phase = phase

        self.green_direction = (
            self._get_direction_from_phase(
                phase
            )
        )

        # ---------------------------------------------------------
        # Simulation time
        # ---------------------------------------------------------

        try:

            current_time = (
                traci.simulation.getTime()
            )

        except Exception:

            return

        delta_time = (
            current_time
            - self.last_sim_time
        )

        if delta_time < 0:

            delta_time = 0

        # ---------------------------------------------------------
        # Phase changed
        # ---------------------------------------------------------

        if phase != previous_phase:

            # Reset timer when entering a new green phase.

            if phase in [
                self.NS_GREEN,
                self.EW_GREEN,
            ]:

                self.green_duration = 0.0

        # ---------------------------------------------------------
        # Count green time
        # ---------------------------------------------------------

        if phase in [
            self.NS_GREEN,
            self.EW_GREEN,
        ]:

            self.green_duration += delta_time

        self.last_sim_time = current_time

    # =============================================================
    # HOLD CURRENT GREEN
    # =============================================================

    def _hold_current_green(self):

        """
        Prevent SUMO's default 42-second green duration
        from automatically switching the signal.

        This lets the RL agent decide when to switch.
        """

        phase = self.current_phase

        if phase not in [
            self.NS_GREEN,
            self.EW_GREEN,
        ]:

            return

        try:

            traci.trafficlight.setPhaseDuration(
                self.tls_id,
                100000.0
            )

        except Exception:

            pass

    # =============================================================
    # REQUEST ACTION
    # =============================================================

    def _request_action(self, action):

        action = int(action)

        current_direction = (
            self.green_direction
        )

        # ---------------------------------------------------------
        # Same direction
        # ---------------------------------------------------------

        if action == current_direction:

            self._hold_current_green()

            return False

        # ---------------------------------------------------------
        # Currently yellow
        # ---------------------------------------------------------

        if self.current_phase in [
            self.NS_YELLOW,
            self.EW_YELLOW,
        ]:

            return False

        # ---------------------------------------------------------
        # Minimum green duration
        # ---------------------------------------------------------

        if self.green_duration < self.min_green_time:

            self._hold_current_green()

            return False

        # =========================================================
        # NS -> EW
        # =========================================================

        if (
            current_direction == 0
            and action == 1
        ):

            # Enter NS yellow.

            traci.trafficlight.setPhase(
                self.tls_id,
                self.NS_YELLOW
            )

            # Yellow = 3 seconds.

            traci.trafficlight.setPhaseDuration(
                self.tls_id,
                3.0
            )

            self.switch_count += 1

            return True

        # =========================================================
        # EW -> NS
        # =========================================================

        if (
            current_direction == 1
            and action == 0
        ):

            # Enter EW yellow.

            traci.trafficlight.setPhase(
                self.tls_id,
                self.EW_YELLOW
            )

            # Yellow = 3 seconds.

            traci.trafficlight.setPhaseDuration(
                self.tls_id,
                3.0
            )

            self.switch_count += 1

            return True

        return False

    # =============================================================
    # RESET
    # =============================================================

    def reset(
        self,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        # ---------------------------------------------------------
        # Start SUMO
        # ---------------------------------------------------------

        self._start_sumo()

        # ---------------------------------------------------------
        # Reset RL state
        # ---------------------------------------------------------

        self.episode_step = 0

        self.current_phase = (
            self.NS_GREEN
        )

        self.green_direction = 0

        self.green_duration = 0.0

        self.previous_waiting = 0.0

        self.previous_vehicles = 0.0

        # ---------------------------------------------------------
        # Reset metrics
        # ---------------------------------------------------------

        self.total_throughput = 0

        self.switch_count = 0

        self.total_waiting_time = 0.0

        self.total_vehicle_count = 0.0

        self.total_queue = 0.0

        self.metric_steps = 0

        # ---------------------------------------------------------
        # Reset simulation time
        # ---------------------------------------------------------

        self.last_sim_time = 0.0

        # ---------------------------------------------------------
        # Force NS green
        # ---------------------------------------------------------

        traci.trafficlight.setPhase(
            self.tls_id,
            self.NS_GREEN
        )

        traci.trafficlight.setPhaseDuration(
            self.tls_id,
            100000.0
        )

        # ---------------------------------------------------------
        # Advance SUMO
        # ---------------------------------------------------------

        traci.simulationStep()

        self._update_signal_state()

        # ---------------------------------------------------------
        # Initial traffic statistics
        # ---------------------------------------------------------

        (
            _,
            _,
            total_vehicles,
            total_waiting,
        ) = self._get_traffic_stats()

        self.previous_waiting = (
            total_waiting
        )

        self.previous_vehicles = (
            total_vehicles
        )

        # ---------------------------------------------------------
        # Observation
        # ---------------------------------------------------------

        observation = (
            self._get_observation()
        )

        info = {

            "total_vehicles":
                total_vehicles,

            "total_waiting":
                total_waiting,

            "throughput":
                self.total_throughput,

            "switches":
                self.switch_count,

            "phase":
                self.current_phase,

            "green_direction":
                self.green_direction,

            "green_duration":
                self.green_duration,
        }

        return observation, info

    # =============================================================
    # STEP
    # =============================================================

    def step(self, action):

        self.episode_step += 1

        # ---------------------------------------------------------
        # Agent requests signal action
        # ---------------------------------------------------------

        switched = (
            self._request_action(
                action
            )
        )

        # ---------------------------------------------------------
        # Run SUMO
        # ---------------------------------------------------------

        arrived_this_step = 0

        for _ in range(
            self.sim_steps_per_action
        ):

            if not self._is_traci_connected():

                break

            try:

                traci.simulationStep()

            except Exception:

                break

            # -----------------------------------------------------
            # Vehicles that reached destination
            # -----------------------------------------------------

            try:

                arrived = (
                    traci.simulation
                    .getArrivedNumber()
                )

                arrived_this_step += (
                    arrived
                )

            except Exception:

                pass

            # -----------------------------------------------------
            # Update signal
            # -----------------------------------------------------

            self._update_signal_state()

        # ---------------------------------------------------------
        # Throughput
        # ---------------------------------------------------------

        self.total_throughput += (
            arrived_this_step
        )

        # ---------------------------------------------------------
        # Current traffic
        # ---------------------------------------------------------

        (
            _,
            _,
            total_vehicles,
            total_waiting,
        ) = self._get_traffic_stats()

        # =========================================================
        # METRICS
        # =========================================================

        self.total_waiting_time += (
            total_waiting
            * self.sim_steps_per_action
        )

        self.total_vehicle_count += (
            total_vehicles
        )

        self.total_queue += (
            total_waiting
        )

        self.metric_steps += 1

        # =========================================================
        # REWARD
        # =========================================================

        waiting_change = (
            self.previous_waiting
            - total_waiting
        )

        waiting_reward = (
            waiting_change / 5.0
        )

        queue_penalty = (
            total_waiting / 20.0
        )

        switch_penalty = (
            0.5
            if switched
            else 0.0
        )

        reward = (
            waiting_reward
            - queue_penalty
            - switch_penalty
        )

        reward = float(
            np.clip(
                reward,
                -10.0,
                10.0
            )
        )

        # ---------------------------------------------------------
        # Update previous state
        # ---------------------------------------------------------

        self.previous_waiting = (
            total_waiting
        )

        self.previous_vehicles = (
            total_vehicles
        )

        # =========================================================
        # TERMINATION
        # =========================================================

        terminated = False

        truncated = (
            self.episode_step
            >= self.max_episode_steps
        )

        if self._is_traci_connected():

            try:

                if (
                    traci.simulation
                    .getMinExpectedNumber()
                    <= 0
                ):

                    terminated = True

            except Exception:

                pass

        # =========================================================
        # OBSERVATION
        # =========================================================

        observation = (
            self._get_observation()
        )

        # =========================================================
        # INFO
        # =========================================================

        simulation_time = 0.0

        if self._is_traci_connected():

            try:

                simulation_time = (
                    traci.simulation
                    .getTime()
                )

            except Exception:

                pass

        info = {

            "total_vehicles":
                total_vehicles,

            "total_waiting":
                total_waiting,

            "throughput":
                self.total_throughput,

            "switched":
                switched,

            "switches":
                self.switch_count,

            "phase":
                self.current_phase,

            "green_direction":
                self.green_direction,

            "green_duration":
                self.green_duration,

            "simulation_time":
                simulation_time,

        }

        # =========================================================
        # CLOSE SUMO
        # =========================================================

        if terminated or truncated:

            self.close()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # =============================================================
    # CLOSE
    # =============================================================

    def close(self):

        if self._is_traci_connected():

            try:

                traci.close()

            except Exception:

                pass
