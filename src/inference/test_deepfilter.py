import os, argparse, time
import torch
from pathlib import Path
from src.denoiser.DeepFilterNet.DeepFilterNet.df.enhance import enhance, init_df, load_audio, save_audio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

def main(args):

    # Check input path
    inPath = Path(args.dir_input).resolve()
    assert inPath.exists()

    # Init model
    print(f"Load model: {args.dir_model}/{args.model_name}\n")
    df_model, df_state, df_name, _ = init_df(model_base_dir=args.dir_model, default_model=args.model_name)
    df_model.to(device)
    assert args.model_name == df_name

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
            outpath = outpath[:outpath.rfind('.WAV')] + '_' + args.model_name +'.wav'

            audio, audio_info = load_audio(str(path), sr=16000)

            start_time = time.time()
            denoised_audio = enhance(model=df_model, df_state=df_state, audio=audio)
            elapsed_time = time.time() - start_time

            save_audio(outpath, denoised_audio, audio_info.sample_rate)

            print("Audio: '{}', length: {:.2f} s:".format(path, len(audio) / 1000))
            print("\tdenoised audio    '{}'".format(outpath))
            print('\tprocessing time   {:.2f} s'.format(elapsed_time))
            print('\tprocessing speed  {:.1f} RT'.format(len(audio) / 1000 / elapsed_time))
    else:
        raise ValueError("Invalid input path.")


if __name__ == "__main__":

    model_name = 'DeepFilterNet3'
    snr_dB = 12

    prefix = f'SNR_{str(snr_dB)}dBsnr'
    dir_data = "/media/ssd/TIMIT/DENOISER"
    dir_model = f"{str(Path(os.getcwd()).parent)}/denoiser/DeepFilterNet/models/{model_name}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_input", type=str, help="Path to noisy speech",
                        default=f"{dir_data}/{prefix}")
    parser.add_argument("--dir_output", type=str, help="Output directory",
                        default=f"{dir_data}/{prefix}_{model_name}")
    parser.add_argument("--dir_model", type=str, help="Model directory",
                        default=dir_model)
    parser.add_argument("--model_name", type=str, help="Model name",
                        default=model_name)
    args = parser.parse_args()
    main(args)
    print('\nTesting finished!\n')
