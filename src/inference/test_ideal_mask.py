import os
import yaml
import pickle
import librosa
import numpy as np
import warnings
from scipy.signal.windows import hann
from matplotlib import pyplot as plt
from scipy.io import wavfile


def read_yaml(path):
    with open(path) as f:
        return list(yaml.safe_load_all(f))[0]


def read_pickle(path):
    with open(path, 'br') as handle:
        return pickle.load(handle)


def load_audio(audio_path, fs):
    Sn, f_Sn = librosa.load(audio_path, sr=None, mono=True)
    # Sampling frequency check
    if f_Sn > fs:
        Sn = librosa.resample(Sn, orig_sr=f_Sn, target_sr=fs)
    elif f_Sn < fs:
        raise ValueError('The input audio has a low sample rate!')
    return Sn, f_Sn


def get_stft(samples, params):
    # STFT window
    window_function = np.sqrt(hann(M=params['n_fft'], sym=False))
    # Compute STFT
    complex_spectra = librosa.stft(
        y=samples,
        n_fft=params['n_fft'],
        hop_length=params['hop_length'],
        win_length=params['n_fft'],
        window=window_function)
    # Remove specified bins (e.g. dc, nyquist etc.)
    complex_spectra = np.delete(complex_spectra, params['skip_bins'], axis=0)
    # Nullify first and last bins
    complex_spectra[[-1, 0], :] = 0
    # Get amplitudes and phases
    phase_spectra = np.angle(complex_spectra[:, :]).astype(np.float32)
    amplitude_spectra = np.abs(complex_spectra[:, :]).astype(np.float32)
    # Plot spectrogram
    if 0:
        plot_spectrogram(amplitude_spectra)
    return amplitude_spectra, phase_spectra


def plot_spectrogram(stft):
    S_db = librosa.amplitude_to_db(stft, ref=np.max)
    plt.figure()
    librosa.display.specshow(S_db)
    plt.colorbar()
    plt.show()


def get_features(params_feat, audio_dir=None, audio_name=None):
    print('\nFeature extraction...')
    total_amplitudes = []
    total_phases = []
    audio_list = []

    if audio_dir is None and audio_name is None:
        print('Run testing mode')
        audio_dir = params_feat['root_audio_test']
    if audio_name:
        if type(audio_name) is str:  # single file processing
            lstdir = [audio_name]
        elif type(audio_name) is list:  # list of files processing
            lstdir = audio_name
        else:
            raise Exception("Wrong lstdr type.")
    else:  # full dir processing
        lstdir = os.listdir(audio_dir)
        # Init bin packer
    for wav_file in lstdir:
        if audio_dir is None:
            audio_path = wav_file
        else:
            audio_path = os.path.join(audio_dir, wav_file)
        audio_list.append(audio_path)
        # Load audio
        print('\nRead audio file: ', wav_file)
        Sn, _ = load_audio(audio_path, params_feat['sample_rate'])
        # Get spectra
        amplitudes, phases = get_stft(Sn, params_feat)

        total_amplitudes.append(amplitudes)
        total_phases.append(phases)

    print('Feature extraction is done.')
    return total_amplitudes, total_phases, audio_list


def apply_gain_mask(out_stft, ref_stft):
    gain_mask = out_stft / ref_stft
    gain_mask = np.nan_to_num(gain_mask, nan=0, posinf=0, neginf=0)
    for f in range(gain_mask.shape[1]):
        for b in range(gain_mask.shape[0]):
            if gain_mask[b, f] < 0.5:
                gain_mask[b, f] = 0  # 0.1
    return gain_mask * ref_stft


def restore_missing_bins(X, skip_bins):
    Y = np.zeros(shape=(X.shape[0], X.shape[1])).astype(np.float32)
    missing_bins = X[skip_bins, :]
    for i in skip_bins:
        Y = np.insert(X, i, missing_bins, axis=0)
    return Y


def restore_complex(amplitudes, phase, skip_bins):
    amplitudes = restore_missing_bins(amplitudes, skip_bins)
    phase = restore_missing_bins(phase, skip_bins)
    return amplitudes * np.exp(1j * phase)


def get_output_audio(stft, params, audio_list):
    # Define window
    window_function = np.sqrt(hann(M=params['n_fft'], sym=False))
    # Inverse STFT
    audios = []
    for i in range(len(stft)):
        # Librosa expects matrix of shape=[1 + n_fft/2, time]
        y = librosa.istft(
            stft_matrix=stft[i],
            hop_length=params['hop_length'],
            win_length=params['n_fft'],
            window=window_function)
        if audio_list is not None:
            audio_file = audio_list[i]
            S, f_Sn = load_audio(audio_file, params['sample_rate'])
            if len(y) < len(S):
                warnings.warn('Duration of the input and output signal is different')
                y = np.pad(y, pad_width=(0, S.shape[0] - y.shape[0]))

        audios.append(y)

    return audios


def main(dir_params, dir_inputs):
    # DEFINE PATHS
    path_param_feat = os.path.join(dir_params, 'config_features.yml')
    path_param_model = os.path.join(dir_params, 'config_train.yml')
    path_norm_factors = os.path.join(dir_params, 'norm_factors.pickle')
    # READ CONFIGS
    params_feat = read_yaml(path_param_feat)
    params_model = read_yaml(path_param_model)
    params_model['norm_factors'] = read_pickle(path_norm_factors)
    # IO DIRS
    path_inputs_noisy = os.path.join(dir_inputs, 'inputs')
    path_inputs_clean = os.path.join(dir_inputs, 'clean')
    dir_outs = os.path.join(os.getcwd(), 'test_ideal_mask')
    if not os.path.isdir(dir_outs):
        os.makedirs(dir_outs)
    # EXTRACT FEATURES
    noisy_amplitudes, noisy_phases, inputs_list = get_features(
        params_feat=params_feat,
        audio_dir=path_inputs_noisy)
    clean_amplitudes, clean_phases, _ = get_features(
        params_feat=params_feat,
        audio_dir=path_inputs_clean)
    # Loop through the files
    output = []
    for i in range(len(inputs_list)):
        output.append(apply_gain_mask(clean_amplitudes[i], noisy_amplitudes[i]))
        # Add missing bins and phases
        output[i] = restore_complex(
            amplitudes=output[i],
            phase=noisy_phases[i],
            skip_bins=params_feat['skip_bins'])
    print('Get de-noised audio output')
    audios = get_output_audio(
        stft=output,
        params=params_feat,
        audio_list=inputs_list)
    for i in range(len(audios)):
        wav_name = os.path.join(dir_outs, 'ideal_mask__' + inputs_list[i].split('/')[-1])
        # SAVE AUDIOS
        wavfile.write(filename=wav_name, rate=params_feat['sample_rate'], data=audios[i])
    print('\nDone!')


if __name__ == '__main__':
    main(
        dir_params='/home/iakovenant/PycharmProjects/Iguana-feon/models/32fp_16000hz_512fft_128hop_64eg_16seg_log_norm_2_S8hN2h_mix/20230315-223457',
        dir_inputs='/home/iakovenant/datasets/audio/test/ideal_mask',
    )
