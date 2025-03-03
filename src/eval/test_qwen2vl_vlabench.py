import os
import random
import warnings
import json
import torch
from qwen_vl_utils import process_vision_info
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from colorama import Fore, Style, init
from open_r1.vlabench import *
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


warnings.filterwarnings("ignore")
init(autoreset=True)

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class VLMEvaluator:
    def __init__(self, model_name, data_path, save_path):
        self.data_path = data_path
        assert os.path.exists(data_path), "Data path does not exist"
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.all_task_list = os.listdir(data_path)
        self.pre_prompt = get_prompt()
        self.model_name = model_name

        self.vlm = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=f"cuda",
        )
        self.vlm_processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        self.datapoints = []
        task_types = [d.name for d in Path(self.data_path).iterdir() if d.is_dir()]
        for task_type in task_types:
            tasks = [
                d.name for d in Path(self.data_path, task_type).iterdir() if d.is_dir()
            ]
            for task in tasks:
                examples = [
                    d.name
                    for d in Path(self.data_path, task_type, task).iterdir()
                    if d.is_dir()
                ]
                for example in examples:
                    io_paths = self.load_single_io_path(task_type, task, example)
                    exist_flags = [os.path.exists(p) for p in io_paths]
                    if all(exist_flags):
                        not_empty_flags = [os.path.getsize(p) > 0 for p in io_paths]
                        if all(not_empty_flags):
                            self.datapoints.append((task_type, task, example))

    def build_input(self, task_type, task_name, example_num, few_shot_num=0):
        prepared_input = {}
        prepared_input["pre_prompt"] = self.pre_prompt

        if few_shot_num != 0:
            prepared_input["shot_input_pic"] = {}
            prepared_input["shot_input_pic_gt"] = {}
            prepared_input["shot_input_instruction"] = {}
            prepared_input["shot_output"] = {}

        for i in range(few_shot_num):
            shot_example = random.choice(self.datapoints)
            while shot_example == (task_type, task_name, example_num):
                shot_example = random.choice(self.datapoints)
            shot_task_type, shot_task_name, shot_example_num = shot_example
            shot_input_pic, shot_input_pic_gt, shot_input_instruction = (
                self.load_single_input(shot_task_type, shot_task_name, shot_example_num)
            )
            shot_output = self.load_single_output(
                shot_task_type, shot_task_name, shot_example_num
            )

            prepared_input["shot_input_pic"][str(i)] = shot_input_pic
            prepared_input["shot_input_pic_gt"][str(i)] = shot_input_pic_gt
            prepared_input["shot_input_instruction"][str(i)] = shot_input_instruction
            prepared_input["shot_output"][str(i)] = shot_output

        input_pic, input_pic_gt, input_instruction = self.load_single_input(
            task_type, task_name, example_num
        )
        prepared_input["input_pic"] = input_pic
        prepared_input["input_pic_gt"] = input_pic_gt
        prepared_input["input_instruction"] = input_instruction

        return prepared_input

    def load_single_io_path(self, task_type, task_name, example_num):
        input_pic_path = os.path.join(
            self.data_path,
            task_type,
            task_name,
            example_num,
            "input/input.png",
        )
        input_pic_gt_path = os.path.join(
            self.data_path,
            task_type,
            task_name,
            example_num,
            "input/input_gt.png",
        )
        input_instruction_path = os.path.join(
            self.data_path,
            task_type,
            task_name,
            example_num,
            "input/instruction.txt",
        )
        gt_operation_sequence_path = os.path.join(
            self.data_path,
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

    def load_single_input(self, task_type, task_name, example_num):
        input_pic_path = os.path.join(
            self.data_path,
            task_type,
            task_name,
            example_num,
            "input/input.png",
        )
        input_pic_gt_path = os.path.join(
            self.data_path,
            task_type,
            task_name,
            example_num,
            "input/input_gt.png",
        )
        input_instruction_path = os.path.join(
            self.data_path,
            task_type,
            task_name,
            example_num,
            "input/instruction.txt",
        )

        input_pic = input_pic_path
        input_pic_gt = input_pic_gt_path
        input_instruction = input_instruction = open(
            input_instruction_path, "r", encoding="utf-8"
        ).read()

        return input_pic, input_pic_gt, input_instruction

    def load_single_output(self, task_type, task_name, example_num):
        gt_operation_sequence_path = os.path.join(
            self.data_path,
            task_type,
            task_name,
            example_num,
            "output/operation_sequence.json",
        )
        with open(gt_operation_sequence_path) as f:
            gt_operation_sequence = json.load(f)
        return gt_operation_sequence

    def build_prompt_with_tilist(self, ti_list):
        content = []
        for ti in ti_list:
            if ti[0] == "text":
                content.append({"type": "text", "text": ti[1]})
            elif ti[0] == "image":
                content.append({"type": "image", "image": ti[1]})
        return content

    def get_single_anwer(self, ti_list):
        content = self.build_prompt_with_tilist(ti_list)
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
        text = self.vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.vlm_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.vlm.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def get_ti_list(self, input_dict):
        ti_list = []
        ti_list.append(["text", input_dict["pre_prompt"]])
        if "shot_input_pic" in input_dict:
            for shot_num in input_dict["shot_input_pic"]:
                shot_input_pic = Image.open(input_dict["shot_input_pic"][shot_num])
                shot_input_pic_gt = Image.open(
                    input_dict["shot_input_pic_gt"][shot_num]
                )
                ti_list.append(["text", "Example " + shot_num + " input picture:"])
                ti_list.append(["image", shot_input_pic])
                ti_list.append(
                    [
                        "text",
                        "Example " + shot_num + " input picture with numbered tags",
                    ]
                )
                ti_list.append(["image", shot_input_pic_gt.resize(shot_input_pic.size)])
                ti_list.append(
                    ["text", "Example " + shot_num + " language instruction:"]
                )
                ti_list.append(["text", input_dict["shot_input_instruction"][shot_num]])
                ti_list.append(
                    ["text", "Example " + shot_num + " output skill sequence"]
                )
                ti_list.append(
                    ["text", json.dumps(input_dict["shot_output"][shot_num])]
                )
        input_dic = Image.open(input_dict["input_pic"])
        input_dic_gt = Image.open(input_dict["input_pic_gt"])
        ti_list.append(["text", "Input picture"])
        ti_list.append(["image", input_dic])
        ti_list.append(["text", "Input picture with numbered tags"])
        ti_list.append(["image", input_dic_gt.resize(input_dic.size)])
        ti_list.append(["text", "Language instruction:"])
        ti_list.append(["text", input_dict["input_instruction"]])
        ti_list.append(["text", "Please give the output skill sequence"])
        return ti_list

    def get_score(self, task_type, task_name, example_num, model_output):
        standard_output = self.load_single_output(task_type, task_name, example_num)
        standard_output = json.dumps(standard_output)
        score = get_accurancy_score([standard_output], model_output, eval=True)
        return score[0]

    def evaluate(self, few_shot_num=0):
        print(Fore.YELLOW + Style.BRIGHT + "\n\nworking on ", end="")
        print(Fore.BLUE + Style.BRIGHT + self.model_name)
        model_output, score_output = {}, {}
        for task_type, task, example in tqdm(self.datapoints):
            few_shot_dict = self.build_input(task_type, task, example, few_shot_num)
            model_prompt = self.get_ti_list(few_shot_dict)
            answer = self.get_single_anwer(model_prompt)
            if task_type not in model_output:
                model_output[task_type] = {}
                score_output[task_type] = {}
            if task not in model_output[task_type]:
                model_output[task_type][task] = {}
                score_output[task_type][task] = {}
            model_output[task_type][task][example] = answer
            score_output[task_type][task][example] = self.get_score(
                task_type, task, example, answer
            )
        for task_type in score_output.keys():
            task_type_scores = []
            for task in score_output[task_type].keys():
                for example in score_output[task_type][task].keys():
                    task_type_scores.append(score_output[task_type][task][example])
            average = sum(task_type_scores) / len(task_type_scores)
            score_output[task_type]["average"] = average
            print(f"Task type {task_type} average score: {average}")

        with open(
            os.path.join(self.save_path, f"{self.model_name[5:]}_model_output.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(model_output, f)
        with open(
            os.path.join(self.save_path, f"{self.model_name[5:]}_score_output.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(score_output, f)


if __name__ == "__main__":
    evaluator = VLMEvaluator(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        data_path="/data/lx/vlabench_dataset/eval_vlm_v0",
        save_path=os.path.join("/data/lx/vlabench_dataset/eval_output"),
    )
    evaluator.evaluate(few_shot_num=1)
