import os, argparse, time
import torch
import torchaudio.functional as F
from torchaudio import load, save
from torchaudio.pipelines import CONVTASNET_BASE_LIBRI2MIX
from pathlib import Path


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


def separate_sources(model, mix, device=None):

    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    if mix.ndim != 3:
        mix = mix[None, :, :]

    with torch.no_grad():
        out = model.forward(mix.to(device))

    return out[0]


def main(args):

    # Check input path
    inPath = Path(args.dir_input).resolve()
    assert inPath.exists()

    # Init model
    print(f"Load model: {args.model_name}\n")

    model = CONVTASNET_BASE_LIBRI2MIX.get_model()
    model.to(device)

    print(f"Model sample rate: {CONVTASNET_BASE_LIBRI2MIX.sample_rate}")

    if inPath.is_file() and inPath.suffix == '.wav':
        raise ValueError("Invalid input path.")  # todo?

    elif inPath.is_dir():

        if args.dir_output:

            # full provided path
            outdir = Path(args.dir_output).resolve()
        else:
            raise ValueError("Invalid model name variable.")  # todo?

        print(f'\nWriting output to:{str(outdir)}')
        outdir.mkdir(parents=True, exist_ok=True)

        fpaths = list(inPath.glob('*.WAV'))
        for ii, path in enumerate(fpaths):

            print(f"\nProcessing file [{ii+1}/{len(fpaths)}]")
            print(path)

            outpath = str(outdir / path.name)
            if os.path.isfile(outpath):
                print(f'Skip existing output: {outpath}')
                continue
            outpath = outpath[:outpath.rfind('.WAV')] + '_' + args.model_name +'.wav'

            input_wave, input_sr = load(str(path))
            input_wave = input_wave.to(device)

            if input_sr != CONVTASNET_BASE_LIBRI2MIX.sample_rate:
                input_wave = F.resample(input_wave, input_sr, CONVTASNET_BASE_LIBRI2MIX.sample_rate)

            start_time = time.time()

            source_waves = separate_sources(model, input_wave, device)[0]
            source_waves = source_waves[None,:]

            source_min, _ = torch.min(source_waves, dim=1, keepdim=True)
            source_max, _ = torch.max(source_waves, dim=1, keepdim=True)
            source_waves = (source_waves - source_min) / (source_max - source_min)

            elapsed_time = time.time() - start_time

            if input_sr != CONVTASNET_BASE_LIBRI2MIX.sample_rate:
                denoised_audio = F.resample(torch.mean(source_waves, dim=0).unsqueeze(0).cpu(), CONVTASNET_BASE_LIBRI2MIX.sample_rate, input_sr)
            else:
                denoised_audio = torch.mean(source_waves, dim=0).unsqueeze(0).cpu()

            save(outpath, denoised_audio, input_sr)

            print("Audio: '{}', length: {:.2f} s:".format(path, len(input_wave) / 1000))
            print("\tdenoised audio    '{}'".format(outpath))
            print('\tprocessing time   {:.2f} s'.format(elapsed_time))
            print('\tprocessing speed  {:.1f} RT'.format(len(input_wave) / 1000 / elapsed_time))
    else:
        raise ValueError("Invalid input path.")


if __name__ == "__main__":

    model_name = 'convtasnet'
    snr_dB = 0

    prefix = f'SNR_{str(snr_dB)}dBsnr'
    dir_data = "/media/ssd/TIMIT/DENOISER"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_input", type=str, help="Path to noisy speech",
                        default=f"{dir_data}/{prefix}")
    parser.add_argument("--dir_output", type=str, help="Output directory",
                        default=f"{dir_data}/{prefix}_{model_name}")
    parser.add_argument("--model_name", type=str, help="Model name",
                        default=model_name)
    args = parser.parse_args()
    main(args)
    print('\nTesting finished!\n')
