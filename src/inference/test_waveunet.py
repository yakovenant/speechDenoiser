import os, argparse, time
import torch
import soundfile as sf
import numpy as np
import src.denoiser.WaveUNet.data.utils as data_utils
import src.denoiser.WaveUNet.model.utils as model_utils
from src.denoiser.WaveUNet.test import predict
from src.denoiser.WaveUNet.model.waveunet import Waveunet
from pathlib import Path
from librosa import resample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


def load_model(checkpoint, args, device):
    num_features = [args.features * i for i in range(1, args.levels+1)] if args.feature_growth == "add" else \
        [args.features * 2 ** i for i in range(0, args.levels)]
    target_outputs = int(args.output_size * args.sr)

    model = Waveunet(
        args.channels,
        num_features,
        args.channels,  # ?
        args.instruments,
        kernel_size=args.kernel_size,
        target_output_size=target_outputs,
        depth=args.depth,
        strides=args.strides,
        conv_type=args.conv_type,
        res=args.res,
        separate=args.separate
    )

    if device.type != "cpu":
        model = model_utils.DataParallel(model)
        print("Move model to GPU")
        model.to(device)
        cuda=True
    else:
        cuda=False
    state = model_utils.load_model(model, None, checkpoint, cuda)
    print('Step', state['step'])
    return model


def run_model(model, input_file, args, device):

    # Load mix in original SR
    input_wave, input_sr = data_utils.load(input_file, sr=None, mono=True)
    mix_channels = input_wave.shape[0]
    mix_len = input_wave.shape[1]

    # Adapt mixture channels to required input channels
    if args.channels == 1:
        input_wave = np.mean(input_wave, axis=0, keepdims=True)
    else:
        if mix_channels == 1:  # Duplicate channels if input is mono but model is stereo
            input_wave = np.tile(input_wave, [args.channels, 1])
        else:
            assert (mix_channels == args.channels)

    sr_target = args.sr
    # Resample to model SR
    if input_sr != sr_target:
        input_wave = resample(input_wave, orig_sr=input_sr, target_sr=sr_target)

    output_wave = predict(input_wave, model)

    return output_wave, input_sr


def main(args):

    # Check input path
    inPath = Path(args.dir_input).resolve()
    assert inPath.exists()

    # Init model
    print(f"Load model: {args.model_name}\n")
    checkpoint_file = os.path.join(args.dir_model, 'model')
    model = load_model(checkpoint_file, args, device)
    print(f"Loading model checkpoint: {str(checkpoint_file)}")

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

            start_time = time.time()
            source_waves, output_sr = run_model(model, str(path), args, device)
            output_wave = source_waves['vocals']
            elapsed_time = time.time() - start_time

            if args.sr != output_sr:
                x = np.mean(output_wave, axis=0)
                denoised_audio = resample(x, orig_sr=args.sr, target_sr=output_sr)
            else:
                denoised_audio = torch.mean(source_waves, dim=0).unsqueeze(0).numpy().cpu()

            sf.write(outpath, denoised_audio, output_sr)

            print("\tdenoised audio    '{}'".format(outpath))
            print('\tprocessing time   {:.2f} s'.format(elapsed_time))
    else:
        raise ValueError("Invalid input path.")


if __name__ == "__main__":

    model_name = 'WaveUNet'
    snr_dB = 0

    prefix = f'SNR_{str(snr_dB)}dBsnr'
    dir_data = "/media/ssd/TIMIT/DENOISER"
    dir_model = os.path.join(os.path.dirname(os.getcwd()), 'denoiser', model_name, 'checkpoints', 'pretrained')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_input", type=str, help="Path to noisy speech",
                        default=f"{dir_data}/{prefix}")
    parser.add_argument("--dir_output", type=str, help="Output directory",
                        default=f"{dir_data}/{prefix}_{model_name}")
    parser.add_argument("--dir_model", type=str, help="Model directory",
                        default=dir_model)
    parser.add_argument("--model_name", type=str, help="Model name",
                        default=model_name)

    parser.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"],
                        help="List of instruments to separate (default: \"bass drums other vocals\")")
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA (default: False)')
    parser.add_argument('--features', type=int, default=32,
                        help="Number of feature channels per layer")
    parser.add_argument('--load_model', type=str, default='checkpoints/waveunet/model',
                        help="Reload a previously trained model")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    parser.add_argument('--levels', type=int, default=6,
                        help="Number od DS/US blocks")
    parser.add_argument('--depth', type=int, default=1,
                        help="Number of convs per block")
    parser.add_argument('--sr', type=int, default=44100,
                        help="Sampling rate")
    parser.add_argument('--channels', type=int, default=2,
                        help="Number of input audio channels")
    parser.add_argument('--kernel_size', type=int, default=5,
                        help="Filter width of kernels. Has to be an odd number")
    parser.add_argument('--output_size', type=float, default=2.0,
                        help="Output duration")
    parser.add_argument('--strides', type=int, default=4,
                        help="Strides in Waveunet")
    parser.add_argument('--conv_type', type=str, default='gn',
                        help="Type of convolution (normal, MN-normalised, GN-normalised: normal/bn/gn)")
    parser.add_argument('--res', type=str, default='fixed',
                        help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    parser.add_argument('--separate', type=int, default=1,
                        help="Train separate model for each ource (1) or only one (0)")
    parser.add_argument('--feature_growth', type=str, default='double',
                        help="How the features in each layer should grow, either (add) the initial number of features each time, or multiply by 2 (double)")

    args = parser.parse_args()
    main(args)
    print('\nTesting finished!\n')
