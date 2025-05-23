This version of the ReadMe.txt is specifically for the Wav2Vec and the Hubert ASR model.

This is a table that represents the arguments, default, and the description of each parameter required.

| Argument        | Description                                         | Default    |
| --------------- | --------------------------------------------------- | ---------- |
| `--model`       | Choose transcription model: `wav2vec2` or `hubert`  | `wav2vec2` |
| `--threshold`   | Audio volume threshold for detecting voice          | `0.3`      |
| `--silence`     | Duration of silence (in seconds) to end a recording | `2`        |
| `--max_time` | Max session time in seconds                         | `300`      |

Name of the file is Hubert+Wav2Vec.py

These are sample commands that you can input:

	python Hubert+Wav2Vec.py (this simple uses the default settings)
	
	python Hubert+Wav2Vec.py --model wav2vec --silence 3 (this sets the model to wav2vec and change the silence duration from 2 to 3)

