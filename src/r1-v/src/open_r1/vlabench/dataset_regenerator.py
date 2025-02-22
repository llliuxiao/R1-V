# Transform VLA Bench Dataset to the format that can be load directly into R1-V

import os
from datasets import Dataset, Features, Value, Image, DatasetDict
import json
import glob


def data_regenerator(root_dir):
    for ability_path in glob.glob(root_dir + "/*/"):
        ability = os.path.basename(os.path.normpath(ability_path))
        for task_path in glob.glob(ability_path + "/*/"):
            task = os.path.basename(os.path.normpath(task_path))
            for example_path in glob.glob(task_path + "/*/"):
                example = os.path.basename(os.path.normpath(example_path))
                image_path = os.path.join(example_path, "input/input.png")
                image_gt_path = os.path.join(example_path, "input/input_gt.png")
                instruction_path = os.path.join(example_path, "input/instruction.txt")
                output_seq_path = os.path.join(
                    example_path, "output/operation_sequence.json"
                )
                if not (os.path.exists(image_path) and os.path.exists(image_gt_path)):
                    print(f"{image_path} or {image_gt_path} not exists")
                    continue

                if not (
                    os.path.getsize(image_path) > 0
                    and os.path.getsize(image_gt_path) > 0
                ):
                    print(f"{image_path} or {image_gt_path} is empty")
                    continue

                instruction_text = ""
                if (
                    os.path.exists(instruction_path)
                    and os.path.getsize(instruction_path) > 0
                ):
                    with open(instruction_path, "r") as f:
                        instruction_text = f.read().strip()
                else:
                    print(f"{instruction_path} not exists")

                output_seq = ""
                if (
                    os.path.exists(output_seq_path)
                    and os.path.getsize(output_seq_path) > 0
                ):
                    with open(output_seq_path, "r") as f:
                        output_seq = json.load(f)
                        output_seq = json.dumps(output_seq)
                else:
                    print(f"{output_seq_path} not exists or empty")

                yield {
                    "ability": ability,
                    "task": task,
                    "example_id": example,
                    "image": image_path,
                    "image_gt": image_gt_path,
                    "instruction": instruction_text,
                    "output": output_seq,
                }


features = Features(
    {
        "ability": Value("string"),
        "task": Value("string"),
        "example_id": Value("string"),
        "image": Image(),
        "image_gt": Image(),
        "instruction": Value("string"),
        "output": Value("string"),
    }
)

data_dir = "/data/lx/vlabench_dataset/eval_vlm_v0"
dataset = DatasetDict(
    {
        "train": Dataset.from_generator(
            lambda: data_regenerator(data_dir), features=features
        )
    }
)
dataset.save_to_disk("/data/lx/vlabench_dataset/eval_vlm_r1v")
print(dataset["train"].features)
