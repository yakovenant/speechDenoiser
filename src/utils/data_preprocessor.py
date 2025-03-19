from os.path import join
from warnings import warn
from pathlib import Path
from soundfile import read, write
from src.processing.get_features import srCheck, removeSilence


def wave_processor(path_input, target_sr):
    input_signal, input_sr = read(str(path_input))

    if len(input_signal.shape) > 1:
        warn('Input WAV file is stereo and will be converted to mono.')
        input_signal_mono = input_signal[:, 0]
    else:
        input_signal_mono = input_signal

    input_signal_mono_resampled, _ = srCheck(input_signal_mono, input_sr, target_sr)
    input_signal_mono_resampled_nosilence = removeSilence(input_signal_mono_resampled)

    return input_signal_mono_resampled_nosilence


def main(path_input: str, path_output=None, target_sr=16000):

    path_input = Path(path_input).resolve()
    assert path_input.exists()

    if path_input.is_file() and path_input.suffix == '.wav':

        signal_output = wave_processor(path_input, target_sr)

        if path_output is None:
            path_output = join(str(path_input.parent), 'output.wav')

        print('Writing output to:', str(path_output))
        write(path_output, signal_output, target_sr)

    elif path_input.is_dir():

        path_input_files = list(path_input.glob('*.wav'))
        for ii, path_to_input_file in enumerate(path_input_files):
            print(f"Processing file [{ii+1}/{len(path_input_files)}]")
            print(path_to_input_file)
            signal_output = wave_processor(path_to_input_file, target_sr)

            path_output = join(str(path_input), str(path_to_input_file.stem) + '_processed.wav')
            write(path_output, signal_output, target_sr)

    else:
        raise ValueError("Invalid input path.")


if __name__ == '__main__':
    print("\nRun audio data preprocessor...")
    main(
        path_input=f"/home/iakovenant/datasets/audio/custom/2023-06-04MTC-DATAFEST_1"
    )
    print("\n...done.")
