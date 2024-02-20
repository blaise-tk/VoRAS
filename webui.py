import os

from modules import cmd_opts, ui

_list_dir = os.listdir


def listdir4mac(path):
    return [file for file in _list_dir(path) if not file.startswith(".")]


os.listdir = listdir4mac


def webui():
    app = ui.create_ui()
    app.queue(64)
    app = app.launch(
        server_name=cmd_opts.opts.host,
        server_port=cmd_opts.opts.port,
        share=cmd_opts.opts.share,
        inbrowser=True,
    )


if __name__ == "__main__":
    webui()
