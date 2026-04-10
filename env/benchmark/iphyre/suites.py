import os
import json


TASK_SUITE_PATH = {
    "Iphyre-HandDesign-v0": "../iphyre/test_toy20250110/20250525/output_hand_test",
    "Iphyre-HandDesign-v1": "../iphyre_hand_design_dataset",
    "Iphyre-ProceduralShift-v0": "../iphyre/test_toy20250110/20250602/output_eval_shift",
    "Iphyre-ProceduralRotate-v0": "../iphyre/test_toy20250110/20250602/output_eval_rotate",
    "Iphyre-VLMGeneratedShift-v0": "../iphyre/test_toy20250110/20250427/output_shift",
    "Iphyre-VLMGeneratedRotate-v0": "../iphyre/test_toy20250110/20250427/output_rotate",
}

TEST_HARD_LIMIT = 100


def load_test_suite(test_suite_name):
    """Return (env_names, env_task_configs) for a named Iphyre test suite."""
    print("Load test suite: ", test_suite_name)
    tasks_path = TASK_SUITE_PATH[test_suite_name]
    task_dirs = [d for d in os.listdir(tasks_path) if not d.startswith(".")]

    env_names = []
    env_task_configs = []

    for task_dir in task_dirs:
        config_path = os.path.join(tasks_path, task_dir, "config.json")
        config = json.load(open(config_path))
        task_config = config["config"]
        task_name = task_dir

        if "VLM" in test_suite_name:
            if config["success_rate"] > 0.0:
                env_names.append(task_name)
                env_task_configs.append(task_config)
        else:
            env_names.append(task_name)
            env_task_configs.append(task_config)

        if len(env_names) >= TEST_HARD_LIMIT:
            break

    return env_names, env_task_configs
