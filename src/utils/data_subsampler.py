from os.path import join
from pathlib import Path
from soundfile import read, write


def main(path_input: str, duration_samples=160000):

    path_input = Path(path_input).resolve()
    assert path_input.exists()
    path_output = Path(join(path_input.parent, 'subsampled'))
    path_output.mkdir(exist_ok=True)

    path_input_files = list(path_input.glob('*.wav'))
    for n_subsample, path_to_input_file in enumerate(path_input_files):
        print(f"Processing file [{n_subsample+1}/{len(path_input_files)}]")
        print(path_to_input_file)
        input_signal, sr = read(str(path_to_input_file))

        path_current_subsample = join(str(path_output), str(path_to_input_file.stem))
        Path(path_current_subsample).mkdir(exist_ok=True)
        idx_start = 0
        idx_stop = duration_samples
        i = 1
        while idx_start < len(input_signal):
            path_current_output = join(str(path_current_subsample), str(i) + '.wav')
            current_signal_clip = input_signal[idx_start:idx_stop]
            write(path_current_output, current_signal_clip, sr)
            idx_start += duration_samples
            idx_stop += duration_samples
            i += 1


if __name__ == '__main__':
    print("\nRun audio data preprocessor...")
    main(
        path_input=f"/home/iakovenant/datasets/audio/custom/2023-06-04MTC-DATAFEST_1/processed"
    )
    print("\n...done.")
