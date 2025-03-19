import os
import tensorflow as tf
from argparse import ArgumentParser
from soundfile import read, write
#from scipy.io.wavfile import read, write
from os.path import dirname, join
from pathlib import Path
from src.denoiser.cnnoise_wrapper import CNNoiseWrapper


def main(args):
    # check input path
    inPath = Path(args.input).resolve()
    assert inPath.exists()

    # args.model = join(args.model, "saved_model.pb")
    model_name = args.model.split(".")[0]

    # Create the denoiser object
    denoiser_obj = CNNoiseWrapper(modelfile=args.model, fs=args.fs)

    if inPath.is_file() and inPath.suffix == '.wav':
        # input is single .wav file
        #sigIn, fs = read(str(inPath))
        sigIn = tf.io.read_file(str(inPath))
        sigIn, fs = tf.audio.decode_wav(sigIn, desired_channels=1, desired_samples=16000)
        if len(sigIn.shape) > 1:
            sigIn = sigIn[:, 0]

        outFs, outSig = denoiser_obj(sigIn, fs)

        outname = './{:s}_{:s}.wav'.format(inPath.stem, model_name)
        if args.output:
            # write in given dir
            outdir = Path(args.output)
            outdir.mkdir(exist_ok=True)
            outpath = outdir.joinpath(outname)
        else:
            # write in current work dir
            outpath = Path(outname)

        print('Writing output to:', str(outpath))
        write(outpath.resolve(), outSig, fs)

    elif inPath.is_dir():
        # input is directory
        if args.output:
            # full provided path
            outdir = Path(args.output).resolve()
        else:
            outdir = Path(join('/', *dirname(__file__).split('/')[:-2], model_name, inPath.name))

        outdir.mkdir(parents=True, exist_ok=True)
        print('Writing output to:', str(outdir))

        fpaths = list(inPath.glob('*.WAV'))

        for ii, path in enumerate(fpaths):
            print(f"Processing file [{ii+1}/{len(fpaths)}]")
            print(path)
            outpath = outdir / (model_name.split("/")[-1] + '__' + path.name)
            outpath = str(outpath)
            if os.path.isfile(outpath):
                print(f'Skip existing output: {outpath}')
                continue
            sigIn, fs = read(str(path))
            #fs, sigIn = read(str(path))
            #sigIn = tf.io.read_file(str(path))
            #sigIn, fs = tf.audio.decode_wav(sigIn, desired_channels=1, desired_samples=16000)
            if len(sigIn.shape) > 1:
                sigIn = sigIn[:, 0]
            outFs, outSig = denoiser_obj(sigIn, fs)
            write(outpath, outSig, fs)
    else:
        raise ValueError("Invalid input path.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Path to noisy speech wav file or directory.",
        #default="/home/iakovenant/datasets/audio/test/webinar-speech"
        default="/media/ssd/TIMIT/DENOISER/SNR_12dBsnr"
        #default="/home/iakovenant/datasets/audio/test/tmp"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Optional output directory.",
        default="/media/ssd/TIMIT/DENOISER/SNR_12dBsnr_cnnoise",
        required=False)
    parser.add_argument(
        "-fs",
        type=int,
        help="Sampling rate of the input audio",
        default=16000,
        choices=[16000, 48000],
        required=False)
    parser.add_argument(
        "-nn", "--model",
        type=str,
        help="Path to the TF model directory.",
        default="/home/iakovenant/PycharmProjects/denoiser/models/tf/20231004")
    main(parser.parse_args())
    print('Testing finished!')
