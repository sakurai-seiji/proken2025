from pydub import AudioSegment
from pathlib import Path
import os
import re

input_dir = "/home/sakurai/Chiba3Party/input_wav"
output_dir = "/home/sakurai/Chiba3Party/input_wav_2"

os.makedirs(output_dir, exist_ok=True)

pattern = re.compile(r"(\d+)\.wav$")

for wav_path in Path(input_dir).glob("*.wav"):
    m = pattern.search(wav_path.name)
    if m is None:
        continue

    idx = int(m.group(1))
    if idx < 582:
        continue

    sound = AudioSegment.from_wav(wav_path)

    # 16kHz / mono に統一
    sound = sound.set_frame_rate(16000).set_channels(1)

    out_path = Path(output_dir) / wav_path.name
    sound.export(out_path, format="wav")

    print(f"converted: {wav_path.name}")
