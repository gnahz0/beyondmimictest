from dotvar import auto_load  # noqa

from vuer_sim_example import tasks

# things I want to test: action space dimension, observation space, camera keys.


def test_action_dim():
    env = tasks.make("Pick_place-v1")
    assert env.action_space.shape[0] == 3 + 6 + 1, f"Expected action dimension: 10, but got: {env.action_space.shape[0]}"


def test_state_dim():
    env = tasks.make("Pick_place-v1")
    obs = env.reset()

    state = obs["state"]
    assert state.shape[0] == 3 + 6, f"Expected state shape: (10,), but got: {state.shape}"


def test_reset():
    env = tasks.make("Pick_place-v1")
    obs = env.reset()


def test_camera_keys():
    env = tasks.make("Pick_place-v1")
    obs = env.reset()
    for key in [
        "wrist/rgb",
        "left/rgb",
        "front/rgb",
    ]:
        assert key in obs, f"Expected camera key: {key}, but not found in {env.camera_keys}"
