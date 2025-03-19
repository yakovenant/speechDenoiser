import os
import random
import torch
import torchaudio
import torchaudio.functional as F


def get_same_len(wave_speech, wave_noise):

    len_speech, len_noise = wave_speech.shape[1], wave_noise.shape[1]

    if len_speech > len_noise:
        wave_noise = torch.cat((wave_noise, wave_noise), dim=1)
        return get_same_len(wave_speech, wave_noise)

    elif len_speech < len_noise:
        len_diff = len_noise - len_speech
        return wave_noise[:, len_diff:]

    return wave_noise


def get_mix(path_to_speech, dir_noise, snr_db):

    print(f'Read speech file: {path_to_speech}')
    wave_speech, sr_speech = torchaudio.load(path_to_speech)

    noise_file_name = random.choice(os.listdir(dir_noise))
    path_to_noise = os.path.join(dir_noise, noise_file_name)
    print(f'Add noise from file: {path_to_noise}')
    wave_noise, sr_noise = torchaudio.load(path_to_noise)
    assert sr_speech == sr_noise
    wave_noise = get_same_len(wave_speech, wave_noise)

    mixed_waves = F.add_noise(wave_speech, wave_noise, torch.Tensor(snr_db))
    wave_mix = mixed_waves[0][None, :]

    return wave_mix


def add_noise_to_speech(dir_speech, dir_noise, snrs, sr):

    for snr_level in snrs:
        dir_mix = dir_speech + '_' + str(snr_level) + 'dB'
        print(f'\nSave output to: {dir_mix}\n')
        for speech_file_name in os.listdir(dir_speech):
            path_to_speech = os.path.join(dir_speech, speech_file_name)
            if os.path.isdir(path_to_speech):
                add_noise_to_speech(path_to_speech, dir_noise, snrs, sr)
            if not speech_file_name.split('.')[0].endswith('norm'):
                continue
            mix_file_name = speech_file_name.split('.')[0] + '_' + str(snr_level) + 'dBsnr.' + speech_file_name.split('.')[-1]
            path_to_mix = os.path.join(dir_mix, mix_file_name)
            os.makedirs(os.path.dirname(path_to_mix), exist_ok=True)
            wave_mix = get_mix(path_to_speech, dir_noise, [snr_level])
            torchaudio.save(path_to_mix, wave_mix, sample_rate=sr, format='wav', backend='ffmpeg')
            print(f'Saved to: {path_to_mix}\n')
