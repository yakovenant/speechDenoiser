import argparse
import soundfile as sf
from pathlib import Path
from src.denoiser.nsnet2_wrapper import nsnet2_wrapper


def main(args):
    # check input path
    inPath = Path(args.input).resolve()
    assert inPath.exists()

    # Create the denoiser object
    denoiser_obj = nsnet2_wrapper(fs=args.fs)

    # get model name
    model_name = Path(args.model).stem

    if inPath.is_file() and inPath.suffix == '.wav':

        ## input is single .wav file

        sigIn, fs = sf.read(str(inPath))
        if len(sigIn.shape) > 1:
            sigIn = sigIn[:,0]

        outSig = denoiser_obj(sigIn, fs)

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
        sf.write(outpath.resolve(), outSig, fs)

    elif inPath.is_dir():

        ## input is a directory

        if args.output:
            # full provided path
            outdir = Path(args.output).resolve()
        else:
            outdir = inPath.parent.joinpath(model_name).resolve()

        outdir.mkdir(parents=True, exist_ok=True)
        print(f'Writing output to:{str(outdir)}\n')

        #fpaths = list(inPath.glob('*.wav'))
        fpaths = list(inPath.glob('*.WAV'))

        for ii, path in enumerate(fpaths):
            print(f"Processing file [{ii+1}/{len(fpaths)}]")
            print(path)

            sigIn, fs = sf.read(str(path))

            if len(sigIn.shape) > 1:
                sigIn = sigIn[:,0]

            outSig = denoiser_obj(sigIn, fs)
            outpath = outdir / path.name
            outpath = str(outpath)
            sf.write(outpath, outSig, fs)

    else:
        raise ValueError("Invalid input path.")


if __name__ == "__main__":

    prefix = 'SNR_12dBsnr'

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to noisy speech wav file or directory.",
                        #default="/home/iakovenant/datasets/audio/test/webinar"
                        default=f"/media/ssd/TIMIT/DENOISER/{prefix}")
    parser.add_argument("-o", "--output", type=str, help="Optional output directory.", required=False,
                        default=f"/media/ssd/TIMIT/DENOISER/{prefix}_nsnet2")
    parser.add_argument("-fs", type=int, help="Sampling rate of the input audio",
                        default=16000, choices=[16000, 48000])
    parser.add_argument("-m", "--model", type=str, help="Model name",
                        default="nsnet2-20ms-baseline.onnx")
    args = parser.parse_args()
    main(args)
    print('\nTesting finished!\n')
