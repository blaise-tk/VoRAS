import os
import subprocess
import sys
import shlex

python = sys.executable
git = os.environ.get("GIT", "git")
index_url = os.environ.get("INDEX_URL", "")
stored_commit_hash = None
skip_install = False


def run(command, desc=None, errdesc=None, custom_env=None):
    if desc:
        print(desc)

    try:
        output = subprocess.check_output(
            command, stderr=subprocess.PIPE, shell=True, env=custom_env
        )
    except subprocess.CalledProcessError as e:
        message = f"""{errdesc or 'Error running command'}.
Command: {command}
Error code: {e.returncode}
stdout: {e.output.decode(encoding="utf8", errors="ignore") if e.output else '<empty>'}
stderr: {e.stderr.decode(encoding="utf8", errors="ignore") if e.stderr else '<empty>'}
"""
        raise RuntimeError(message)

    return output.decode(encoding="utf8", errors="ignore")


def is_installed(package):
    try:
        subprocess.check_output(
            [python, "-c", f"import {package}"], stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        return False
    return True


def commit_hash():
    global stored_commit_hash

    if stored_commit_hash is not None:
        return stored_commit_hash

    try:
        stored_commit_hash = (
            subprocess.check_output(f"{git} rev-parse HEAD", shell=True)
            .strip()
            .decode()
        )
    except Exception:
        stored_commit_hash = "<none>"

    return stored_commit_hash


def prepare_environment():
    commit = commit_hash()
    print(f"Python {sys.version}")
    print(f"Commit hash: {commit}")

    if "--skip-install" in sys.argv:
        sys.argv.remove("--skip-install")
        return

    if (
        "--reinstall-torch" in sys.argv
        or not is_installed("torch")
        or not is_installed("torchvision")
    ):
        torch_command = os.environ.get(
            "TORCH_COMMAND",
            "pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118",
        )
        run(
            f'"{python}" -m {torch_command}',
            "Installing torch and torchvision",
            "Couldn't install torch",
        )

    if "--ngrok" in sys.argv and not is_installed("pyngrok"):
        run(
            f'"{python}" -m pip install pyngrok',
            "Installing pyngrok",
            "Couldn't install pyngrok",
        )

    run(
        f'"{python}" -m pip install -r requirements.txt',
        "Installing requirements",
        "Couldn't install requirements",
    )


def start():
    os.environ["PATH"] = (
        os.path.join(os.path.dirname(__file__), "bin")
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    subprocess.run([python, "webui.py", *sys.argv[1:]])


if __name__ == "__main__":
    commandline_args = os.environ.get("COMMANDLINE_ARGS", "")
    sys.argv += shlex.split(commandline_args)
    prepare_environment()
    start()
