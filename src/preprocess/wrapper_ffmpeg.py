import os, subprocess


def loud_norm(path_to_input, sampling_rate):

    if path_to_input.endswith('wav') or path_to_input.endswith('WAV'):
        path_to_output = path_to_input[:-4] + '_norm.' + path_to_input.split('.')[-1]
        # path_to_output = path_to_input.split('.')[0] + '_norm.' + path_to_input.split('.')[1]

        subprocess.run([
            'ffmpeg',
            '-i',
            path_to_input,
            '-af',
            'loudnorm=tp=-1.0',
            '-ar',
            str(sampling_rate),
            path_to_output
        ])
        print(f'Normalized file is saved: {path_to_output}')
        return path_to_output
    else:
        print(f'Skip: {path_to_input}')
        return path_to_input


def silence_remove(path_to_input, sampling_rate):

    if path_to_input.endswith('wav') or path_to_input.endswith('WAV'):
        path_to_output = path_to_input[:-4] + '_vad.' + path_to_input.split('.')[-1]
        #path_to_output = path_to_input.split('.')[0] + '_vad.' + path_to_input.split('.')[1]

        subprocess.call([
            'ffmpeg',
            '-i',
            path_to_input,
            '-ac',
            '1',
            '-af',
            'silenceremove=stop_periods=-1:stop_duration=0.05:stop_threshold=-30dB',
            '-c:a',
            'pcm_s16le',
            '-ar',
            str(sampling_rate),
            path_to_output
        ])
        print(f'Continuous speech file is saved: {path_to_output}')
        return path_to_output
    else:
        print(f'Skip: {path_to_input}')
        return path_to_input


def convert_audio(path_to_input, sampling_rate):

    if path_to_input.endswith('wav') or path_to_input.endswith('WAV'):
        path_to_output = path_to_input[:-4] + '_ffmpeg.' + path_to_input.split('.')[-1]

        subprocess.call([
            'ffmpeg',
            '-i',
            path_to_input,
            '-c:a',
            'pcm_s16le',
            '-ar',
            str(sampling_rate),
            path_to_output
        ])
        print(f'Converted file is saved: {path_to_output}')
        return path_to_output
    else:
        print(f'Skip: {path_to_input}')
        return path_to_input


def preprocess(dir_root, sr, loudnorm, vad):

    for cur_file in os.listdir(dir_root):
        path_to_cur_file = os.path.join(dir_root, cur_file)
        if os.path.isdir(path_to_cur_file):
            preprocess(path_to_cur_file, sr, loudnorm, vad)
        print(f'Read file: {path_to_cur_file}')
        path_to_cur = convert_audio(path_to_cur_file, sr)
        if loudnorm:
            path_to_cur = loud_norm(path_to_cur, sr)
        if vad:
            path_to_cur = silence_remove(path_to_cur, sr)
