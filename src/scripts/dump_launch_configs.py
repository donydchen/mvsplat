from pathlib import Path

import yaml
from colorama import Fore

if __name__ == "__main__":
    # Go to the repo directory.
    x = Path.cwd()
    while not (x / ".git").exists():
        x = x.parent

    # Hackily load JSON with comments and trailing commas.
    with (x / ".vscode/launch.json").open("r") as f:
        launch_with_comments = f.readlines()
    launch = [
        line for line in launch_with_comments if not line.strip().startswith("//")
    ]
    launch = "".join(launch)
    launch = yaml.safe_load(launch)

    for cfg in launch["configurations"]:
        print(f"{Fore.CYAN}{cfg['name']}{Fore.RESET}")

        arg_str = " ".join(cfg.get("args", []))
        if "env" in cfg:
            env_str = " ".join([f"{key}={value}" for key, value in cfg["env"].items()])
        else:
            env_str = ""

        command = f"{env_str} python3 -m {cfg['module']} {arg_str}".strip()
        print(f"{command}\n")
