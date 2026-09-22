import os
import sys

# ============================================================
# PATH SETUP
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_DIR)


# ============================================================
# IMPORTS
# ============================================================

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from traffic_env_v2 import SumoTrafficEnv


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

TOTAL_TIMESTEPS = 120_000

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "models"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "checkpoints"
)

TENSORBOARD_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "tensorboard"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

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


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

def make_environment():

    env = SumoTrafficEnv()

    env = Monitor(
        env,
        filename=os.path.join(
            PROJECT_DIR,
            "results",
            "monitor_v2.csv"
        )
    )

    return env


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SUMO V2 PPO TRAINING")
    print("=" * 60)

    print()
    print("Project directory:")
    print(PROJECT_DIR)

    # --------------------------------------------------------
    # CREATE ENVIRONMENT
    # --------------------------------------------------------

    print()
    print("Creating V2 environment...")

    env = make_environment()

    print("Environment created successfully.")

    # --------------------------------------------------------
    # ENVIRONMENT CHECK
    # --------------------------------------------------------

    print()
    print("Checking environment...")

    observation, info = env.reset()

    print(
        "Observation shape:",
        observation.shape
    )

    print(
        "Initial phase:",
        info.get("phase", "N/A")
    )

    print(
        "Initial vehicles:",
        info.get(
            "total_vehicles",
            "N/A"
        )
    )

    print(
        "Initial waiting:",
        info.get(
            "total_waiting",
            "N/A"
        )
    )

    print()
    print("Environment check passed.")

    # --------------------------------------------------------
    # CREATE PPO MODEL
    # --------------------------------------------------------

    print()
    print("Creating PPO model...")

    model = PPO(
        "MlpPolicy",
        env,

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

        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],
                vf=[128, 128]
            )
        ),

        verbose=1,

        tensorboard_log=TENSORBOARD_DIR,

        device="auto"
    )

    print()
    print("PPO model created.")

    # --------------------------------------------------------
    # CHECKPOINT CALLBACK
    # --------------------------------------------------------

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_v2"
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("STARTING PPO V2 TRAINING")
    print("=" * 60)

    print()
    print(
        f"Training for {TOTAL_TIMESTEPS:,} timesteps..."
    )

    try:

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            reset_num_timesteps=True
        )

    except KeyboardInterrupt:

        print()
        print("=" * 60)
        print("TRAINING INTERRUPTED")
        print("=" * 60)

        interrupted_path = os.path.join(
            MODEL_DIR,
            "ppo_v2_interrupted"
        )

        model.save(
            interrupted_path
        )

        print()
        print(
            "Interrupted model saved:"
        )

        print(
            interrupted_path + ".zip"
        )

        env.close()

        return

    except Exception as e:

        print()
        print("=" * 60)
        print("TRAINING FAILED")
        print("=" * 60)

        print()
        print(
            "Error:",
            type(e).__name__
        )

        print(
            "Message:",
            e
        )

        env.close()

        raise

    # --------------------------------------------------------
    # SAVE FINAL MODEL
    # --------------------------------------------------------

    final_model_path = os.path.join(
        MODEL_DIR,
        "ppo_v2"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # CLOSE ENVIRONMENT
    # --------------------------------------------------------

    env.close()

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("PPO V2 TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        "Final model:"
    )

    print(
        final_model_path + ".zip"
    )

    print()
    print(
        "Checkpoints:"
    )

    print(
        CHECKPOINT_DIR
    )

    print()
    print(
        "TensorBoard logs:"
    )

    print(
        TENSORBOARD_DIR
    )

    print()
    print("=" * 60)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
