import json
import soundfile as sf
from pathlib import Path

def create_manifest(audio_dir, text_file, output_manifest):
    audio_dir = Path(audio_dir)

    output_manifest = Path(output_manifest) 
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with open(text_file, encoding="utf-8") as f, \
         open(output_manifest, "w", encoding="utf-8") as out:

        i = 1
        for line in f:
            start, end, filler = line.strip().split(' ')
            wav_path = f"/home/sakurai/Chiba3Party/input_wav/input_data_{i}.wav"

            info = sf.info(str(wav_path))
            item = {
                "audio_filepath": str(wav_path),
                "filler": filler,
                "duration": info.duration,
                "sample_rate": info.samplerate,
            }
            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            i +=1

if __name__ == "__main__":
    create_manifest(
        audio_dir="/home/sakurai/Chiba3party/input_wav",
        text_file="/home/sakurai/Chiba3Party/Trans_copy/chiba_2.txt",
        output_manifest="data/train_manifest_2.json"
    )
