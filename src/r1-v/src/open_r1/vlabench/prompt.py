import pathlib

dir = pathlib.Path(__file__).parent.absolute()


def get_prompt():
    with open(f"{dir}/vlm_prompt_CoT.txt") as f:
        file = f.read()
    return file
