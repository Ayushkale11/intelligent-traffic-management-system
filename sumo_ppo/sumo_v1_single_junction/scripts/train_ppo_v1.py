import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


# ================================================================
# PATH
# ================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        ".."
    )
)

sys.path.insert(
    0,
    PROJECT_DIR
)

from traffic_env_v1 import SumoTrafficEnv


# ================================================================
# DIRECTORIES
# ================================================================

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ================================================================
# MAIN
# ================================================================

def main():

    print()
    print("=" * 60)
    print("SUMO V1 PPO TRAINING")
    print("=" * 60)

    # ------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------

    env = SumoTrafficEnv(
        use_gui=False,
        max_episode_steps=120,
        sim_steps_per_action=5,
    )

    env = Monitor(env)

    # ------------------------------------------------------------
    # Environment validation
    # ------------------------------------------------------------

    print()
    print("Checking environment...")

    check_env(
        env.unwrapped,
        warn=True
    )

    print("Environment check complete.")

    # ------------------------------------------------------------
    # PPO
    # ------------------------------------------------------------

    print()
    print("Creating PPO model...")

    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,

        n_steps=120,

        batch_size=60,

        n_epochs=10,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        vf_coef=0.5,

        max_grad_norm=0.5,

        verbose=1,

        tensorboard_log=os.path.join(
            RESULT_DIR,
            "tensorboard"
        ),

        device="auto",

        seed=42,
    )

    # ------------------------------------------------------------
    # Training
    # ------------------------------------------------------------

    total_timesteps = 120_000

    print()
    print(
        f"Training for {total_timesteps:,} timesteps..."
    )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "ppo_v1"
    )

    model.save(
        model_path
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print()
    print(
        "Model saved to:"
    )

    print(
        model_path + ".zip"
    )

    print()
    print("=" * 60)

    env.close()


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()
