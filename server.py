import io
import json
import os
import traceback
from typing import *

import soundfile as sf
from flask import Flask, make_response, request, send_file
from scipy.io.wavfile import write

from modules.server.model import VoiceServerModel

model: Optional[VoiceServerModel] = None
app = Flask(__name__)


@app.route("/ping")
def ping():
    return make_response("server is alive", 200)


@app.route("/upload_model", methods=["POST"])
def upload_model():
    """
    input:
        json:
            rvc_model_file: str
                specify rvc model's absolute path (.pt, .pth)
            faiss_index_file: Optional[str]
                specify faiss index'S absolute path (.index)
    """
    global model
    if request.method == "POST":
        rvc_model_file = request.json["rvc_model_file"]
        try:
            model = VoiceServerModel(rvc_model_file)
            return make_response("model is load", 200)
        except:
            traceback.print_exc()
            return make_response("model load error", 400)
    else:
        return make_response("use post method", 400)


@app.route("/convert_sound", methods=["POST"])
def convert_sound():
    """
    input:
        params: json
            speaker_id: int
                default: 0
        input_wav: wav file

    output:
        wavfile
    """
    global model
    if model is None:
        return make_response("please upload model", 400)
    if request.method == "POST":
        input_buffer = io.BytesIO(request.files["input_wav"].stream.read())
        audio, sr = sf.read(input_buffer)

        req_json = json.load(io.BytesIO(request.files["params"].stream.read()))
        sid = int(req_json.get("speaker_id", 0))

        out_audio = model(audio, sr, sid)
        output_buffer = io.BytesIO()
        write(output_buffer, rate=model.tgt_sr, data=out_audio)
        output_buffer.seek(0)
        response = make_response(send_file(output_buffer, mimetype="audio/wav"), 200)
        return response
    else:
        return make_response("use post method", 400)


if __name__ == "__main__":
    app.run()
