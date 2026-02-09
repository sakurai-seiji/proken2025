from pydub import AudioSegment

# ファイルの読み込み
text_file = "/home/sakurai/Chiba3Party/Version1.0/data/K003/K003_006/K003_006-luu.csv"
sound = AudioSegment.from_wav("/home/sakurai/Chiba3Party/Version1.0/data/K003/K003_006/K003_006_IC0A.wav")

# 5000ms～10000ms(5～10秒)を抽出
# sound1 = sound[6771.8:7836.7]

# 抽出した部分を出力
# sound1.export(f"/home/sakurai/Chiba3Party/input_dataset/chiba1232_{i}.wav", format="wav")


i=732

with open(text_file) as t:
    for line in t:
        _, start, end, filler = line.strip().split(',')
        print(start, end)
        splitted_sound = sound[float(start)*1000:float(end)*1000]#秒に変換
        splitted_sound.export(f"/home/sakurai/Chiba3Party/input_wav/input_data_{i}.wav", format="wav")
        i+=1
    
