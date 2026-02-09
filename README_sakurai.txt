１．使うモデルとvocabの指定
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_1_huggingface.py


２．マニフェスト作成
テストデータ用
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_2_create_test_manifest.py

学習データ用
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_2_create_train_manifest.py

検証データ用
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_2_create_valid_manifest.py


３．データローダー（HuBERT + BiLSTM + 全結合）
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_CE_3_dataset_dataloader.py


４．モデル構造の指定
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_CE_4_hubert_bilstm.py


５．学習ループ
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_CE_5_train_loop.py


６．評価
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_CE_6_test.py


７．collate（可変長音声・ラベルのバッチ化処理）
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/practice_CE_collate_fn.py


８．その他
csvを元にwavファイルを分割
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/for_audio_segment.py


学習データ内の相槌の種類とそれぞれのデータ数を出力
/home/sakurai/pro/gakushuu_rennshuu/speechbrain/for_filler_counts.py


※補足
HuBERTは凍結して使用
相槌分類は音響情報のみに基づく
クラス不均衡に対して class weight を使用