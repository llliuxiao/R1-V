from pathlib import Path
import os

data_path = "/data/lx/vlabench_dataset/eval_vlm_v0"


def load_single_io_path(task_type, task_name, example_num):
    input_pic_path = os.path.join(
        data_path,
        task_type,
        task_name,
        example_num,
        "input/input.png",
    )
    input_pic_gt_path = os.path.join(
        data_path,
        task_type,
        task_name,
        example_num,
        "input/input_gt.png",
    )
    input_instruction_path = os.path.join(
        data_path,
        task_type,
        task_name,
        example_num,
        "input/instruction.txt",
    )
    gt_operation_sequence_path = os.path.join(
        data_path,
        task_type,
        task_name,
        example_num,
        "output/operation_sequence.json",
    )
    return (
        input_pic_path,
        input_pic_gt_path,
        input_instruction_path,
        gt_operation_sequence_path,
    )


def load_usable_datapoints():
    datapoints = []
    task_types = [d.name for d in Path(data_path).iterdir() if d.is_dir()]
    for task_type in task_types:
        tasks = [d.name for d in Path(data_path, task_type).iterdir() if d.is_dir()]
        for task in tasks:
            examples = [
                d.name for d in Path(data_path, task_type, task).iterdir() if d.is_dir()
            ]
            for example in examples:
                io_paths = load_single_io_path(task_type, task, example)
                exist_flags = [os.path.exists(p) for p in io_paths]
                if all(exist_flags):
                    not_empty_flags = [os.path.getsize(p) > 0 for p in io_paths]
                    if all(not_empty_flags):
                        datapoints.append(io_paths)
    return datapoints


if __name__ == "__main__":
    import random
    import json

    datapoints = load_usable_datapoints()
    example = random.choice(datapoints)
    with open(example[3], "r") as f:
        a = json.load(f)
        print(a.keys())
