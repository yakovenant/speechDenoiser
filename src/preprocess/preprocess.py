import argparse
from wrapper_ffmpeg import preprocess
from wrapper_speech_mix import add_noise_to_speech


sr_target = 16000
snr_levels = [48, 24, 12, 6, 3, 0]
dir_speech = '/media/ssd/TIMIT/TEST/DR8'
dir_noise = '/media/ssd/KARMA/noise-pack1-all-locs'


def main(args):

    if 0:
        print('\nPreprocess noise...')
        preprocess(args.dir_noise, args.sr_target, loudnorm=True, vad=False)
    if 0:
        print('\nPreprocess speech...')
        preprocess(args.dir_speech, args.sr_target, loudnorm=True, vad=False)
    if 0:
        print('\nAdd noise...')
        add_noise_to_speech(args.dir_speech, args.dir_noise, args.snrs, args.sr_target)
    if 1:
        print('\nGenerate denoiser data')
        get_denoiser_data(dir_root, dir_output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_target", default=sr_target, type=int)
    parser.add_argument("--snrs", default=snr_levels, type=int or list)
    parser.add_argument("--dir_speech", default=dir_speech, type=str)
    parser.add_argument("--dir_noise", default=dir_noise, type=str)
    args = parser.parse_args()

    main(args)

    print('\nDONE!\n')
