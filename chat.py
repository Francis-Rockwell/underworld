import openai
import json
import requests
import torch
import time
import os
from vits import utils, models, app
from vits.models import SynthesizerTrn
from scipy.io.wavfile import write

api_key = "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"
openai.api_key = api_key
gpt_model = "gpt-3.5-turbo"


def chat_with_gpt(message):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    data = {
        "model": gpt_model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0.8,
        "stream": True,
    }
    response = requests.post(url, headers=headers, json=data, stream=True)
    total_msg = ""
    for chunk in response.iter_lines():
        response_data = chunk.decode("utf-8").strip()
        if not response_data:
            continue
        try:
            if response_data.endswith("data: [DONE]"):
                break
            data_list = response_data.split("data: ")
            if len(data_list) > 2:
                json_data = json.loads(data_list[2])
            else:
                json_data = json.loads(response_data.split("data: ")[1])
            if "content" in json_data["choices"][0]["delta"]:
                msg = json_data["choices"][0]["delta"]["content"]
                total_msg += msg
        except:
            print("json load error:", response_data)
    return total_msg


def vits_chat(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace("\n", " ").replace("\r", "").replace(" ", "")
    limitation = os.getenv("SYSTEM") == "spaces"
    if len(text) > 100 and limitation:
        return f"输入文字过长！{len(text)}>100", None, None
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    stn_tst, clean_text = app.get_text(text, hps_ms)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        speaker_id = torch.LongTensor([speaker_id]).to(device)
        audio = (
            net_g_ms.infer(
                x_tst,
                x_tst_lengths,
                sid=speaker_id,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )

    return (22050, audio)


if __name__ == "__main__":
    device = torch.device("cpu")
    hps_ms = utils.get_hparams_from_file(r"./vits/model/config.json")
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model,
    )
    _ = net_g_ms.eval().to(device)
    speakers = hps_ms.speakers
    model, optimizer, learning_rate, epochs = utils.load_checkpoint(
        r"./vits/model/G_953000.pth", net_g_ms, None
    )
    while True:
        user_input = input("You:")
        chat_response = chat_with_gpt(user_input + " 请用50字以内的回答。")
        print("ChatGPT:{}".format(chat_response))
        audio = vits_chat("你妈死了", 0, 2, 0.6, 0.668, 1.2)
        write("output_audio.wav", audio[0], audio[1])
