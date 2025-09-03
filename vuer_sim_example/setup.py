from setuptools import find_packages, setup

setup(
    name='vuer-deployment-sim',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    description='Sim2Sim deployment system for humanoid robots using Vuer simulator with Unitree SDK integration',
    url="https://github.com/your-username/vuer-deployment-sim",  # Update this with your actual repository URL
    python_requires=">=3.8",
    install_requires=[
        "vuer[all]",
        "pinocchio>=3.2.0",
        "mujoco",
        "pyyaml",
        "scipy",
        "onnxruntime",
        "pynput", 
        "sshkeyboard",
        "termcolor",
        "loguru",
        "loop_rate_limiters",
        "pygame",
        "meshcat",
        "hydra-core>=1.2.0",
        "numpy==1.23.5",
        "rich",
        "ipdb",
        "matplotlib",
        "wandb",
        "plotly",
        "tqdm",
        "tensorboard",
        "onnx",
        "opencv-python",
        "joblib",
        "easydict",
        "lxml",
        "numpy-stl",
        "open3d"
    ]
)