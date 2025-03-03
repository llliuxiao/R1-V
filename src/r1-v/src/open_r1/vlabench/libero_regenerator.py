# Transform VLA Bench Dataset to the format that can be load directly into R1-V

import os
from datasets import Dataset, Features, Value, Image, DatasetDict
import json
import tensorflow_datasets as tfds
import PIL


def data_regenerator(dataset_dir, dataset_name):
    dataset, info = tfds.load(
        "libero_10_no_noops", data_dir=dataset_dir, with_info=True
    )
    with open(
        os.path.join(dataset_dir, dataset_name, "chain_of_thought_h4_gripper.json"), "r"
    ) as f:
        gripper_infos = json.load(f)
    for episode in dataset["train"]:
        episode_id = episode["episode_metadata"]["episode_id"].numpy().decode("utf-8")
        gripper_info = gripper_infos[episode_id]["0"]["reasoning"]
        for idx, step in enumerate(episode["steps"]):
            image = PIL.Image.fromarray(step["observation"]["image"].numpy())
            instruction = step["language_instruction"].numpy().decode("utf-8")
            gripper_action = gripper_info[str(idx)]["move"]
            plan = gripper_info[str(idx)]["plan"]
            subtask = gripper_info[str(idx)]["subtask"]
            subtask_reason = gripper_info[str(idx)]["subtask_reason"]
            move_reason = gripper_info[str(idx)]["move_reason"]
            yield {
                "dataset_name": dataset_name,
                "episode_id": episode_id,
                "image": image,
                "instruction": instruction,
                "move": gripper_action,
                "plan": plan,
                "subtask": subtask,
                "subtask_reason": subtask_reason,
                "move_reason": move_reason,
            }


features = Features(
    {
        "dataset_name": Value("string"),
        "episode_id": Value("string"),
        "image": Image(),
        "instruction": Value("string"),
        "move": Value("string"),
        "plan": Value("string"),
        "subtask": Value("string"),
        "subtask_reason": Value("string"),
        "move_reason": Value("string"),
    }
)
dataset_dir = "/data/zzb/tensorflow_datasets"
dataset_name = "libero_10_no_noops"
dataset = DatasetDict(
    {
        "train": Dataset.from_generator(
            lambda: data_regenerator(dataset_dir, dataset_name), features=features
        )
    }
)
dataset.save_to_disk("/data/lx/vlabench_dataset/libero_r1v")
print(dataset["train"].features)
