# noisyfier


Noisyfier adds noise from directories or files to audio file.

Noisyfier supports only `.flac` and `.wav` audio files.

### prepare

run `./download_bg_noise.sh`

### runing 

`python add_noise.py --input test_audio/speech.flac -n noise-example.yaml --output out.flac`

Noise files or directories are described in `.yaml` format like `noise-example.yaml`.

(Note: noisyfier doesn't take audio files from subdirectories) 



