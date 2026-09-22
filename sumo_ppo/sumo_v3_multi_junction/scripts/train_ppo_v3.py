import os
import sys
import signal
import traceback

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# ---------------------------------------------------------
# Make project directory importable
# ---------------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_DIR)

from traffic_env_v3 import SumoTrafficEnvV3


# =========================================================
# PATHS
# =========================================================

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

MODEL_DIR = os.path.join(
    RESULTS_DIR,
    "models"
)

CHECKPOINT_DIR = os.path.join(
    RESULTS_DIR,
    "checkpoints"
)

TENSORBOARD_DIR = os.path.join(
    RESULTS_DIR,
    "tensorboard"
)


# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    TENSORBOARD_DIR,
    exist_ok=True
)


# =========================================================
# TRAINING SETTINGS
# =========================================================

TOTAL_TIMESTEPS = 200_000

MAX_STEPS_PER_EPISODE = 120


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print("=" * 60)
    print("PPO V3 MULTI-JUNCTION TRAINING")
    print("=" * 60)

    print()
    print("Project directory:")
    print(PROJECT_DIR)

    print()
    print("Total timesteps:")
    print(TOTAL_TIMESTEPS)

    print()
    print("Junctions:")
    print("J1, J2, J3, J4")

    print()
    print("Action space:")
    print("[J1, J2, J3, J4]")

    print()
    print("Observation size:")
    print("32")

    # -----------------------------------------------------
    # Create environment
    # -----------------------------------------------------

    print()
    print("Creating V3 environment...")

    env = SumoTrafficEnvV3(
        project_dir=PROJECT_DIR,
        gui=False,
        max_steps=MAX_STEPS_PER_EPISODE,
        step_length=1.0,
    )

    env = Monitor(env)

    print("Environment created successfully.")

    print()
    print("Observation space:")
    print(env.observation_space)

    print()
    print("Action space:")
    print(env.action_space)

    # -----------------------------------------------------
    # Check environment
    # -----------------------------------------------------

    print()
    print("Running initial environment check...")

    observation, info = env.reset()

    print(
        f"Initial observation shape: "
        f"{observation.shape}"
    )

    print(
        f"Initial vehicles: "
        f"{info.get('total_vehicles', 0)}"
    )

    print(
        f"Initial waiting: "
        f"{info.get('total_waiting', 0)}"
    )

    # -----------------------------------------------------
    # PPO checkpoint callback
    # -----------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_v3",
    )

    # -----------------------------------------------------
    # Create PPO
    # -----------------------------------------------------

    print()
    print("Creating PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,

        tensorboard_log=TENSORBOARD_DIR,

        device="auto",
    )

    print()
    print("=" * 60)
    print("STARTING PPO V3 TRAINING")
    print("=" * 60)
    print()

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    try:

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        # -------------------------------------------------
        # Save final model
        # -------------------------------------------------

        final_model_path = os.path.join(
            MODEL_DIR,
            "ppo_v3"
        )

        model.save(
            final_model_path
        )

        print()
        print("=" * 60)
        print("PPO V3 TRAINING COMPLETE")
        print("=" * 60)

        print()
        print("Final model:")
        print(
            final_model_path + ".zip"
        )

        print()
        print("Checkpoints:")
        print(CHECKPOINT_DIR)

        print()
        print("TensorBoard:")
        print(TENSORBOARD_DIR)

    except KeyboardInterrupt:

        print()
        print("=" * 60)
        print("TRAINING INTERRUPTED")
        print("=" * 60)

        interrupted_path = os.path.join(
            MODEL_DIR,
            "ppo_v3_interrupted"
        )

        model.save(
            interrupted_path
        )

        print()
        print("Interrupted model saved:")
        print(
            interrupted_path + ".zip"
        )

    except Exception as e:

        print()
        print("=" * 60)
        print("PPO V3 TRAINING FAILED")
        print("=" * 60)

        print()
        print("Error:")
        print(type(e).__name__)

        print()
        print("Message:")
        print(str(e))

        print()
        traceback.print_exc()

        raise

    finally:

        try:
            env.close()
        except Exception:
            pass


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
