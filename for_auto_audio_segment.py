import os
import csv
import soundfile as sf
import re

def extract_prev_utterance_before_aizuchi(
    wav_path = "/home/sakurai/Chiba3Party/Version1.0/data/T020/T020_025/T020_025_IC0A.wav",
    csj_path = "/home/sakurai/Chiba3Party/Version1.0/data/T020/T020_025/T020_025-luu.csv",
    out_dir = "/home/sakurai/Chiba3Party/input_wav_2",
    aizuchi_pattern = re.compile(r"(あー|ふーん|へえ|えっ|はい|うーん|ええ+|えー+|え)"),
    start_index=1,
    pad_sec = 0.0
):
    
    os.makedirs(out_dir, exist_ok=True)

    # WAV 読み込み
    wav, sr = sf.read(wav_path)
    if wav.ndim > 1:
        wav = wav[:, 0]  # mono

    # CSJ を全行読み込む（1行前を見るため）
    with open(csj_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))

    count = start_index

    for i in range(1, len(rows)):
        row = rows[i]
        prev_row = rows[i - 1]

        if len(row) < 4 or len(prev_row) < 3:
            continue

        text = row[-1]

        # ----- 相槌チェック -----
        if not aizuchi_pattern.search(text):
            continue

        # ----- 1つ前の発話の時間 -----
        start_time = float(prev_row[1])
        end_time = float(prev_row[2])

        # padding
        start_time = max(0.0, start_time - pad_sec)
        end_time = end_time + pad_sec

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)

        if end_sample <= start_sample:
            continue

        segment = wav[start_sample:end_sample]

        out_path = os.path.join(out_dir, f"{count:04d}.wav")
        sf.write(out_path, segment, sr)

        count += 1

    print(f"Extracted {count - start_index} files to {out_dir}")
