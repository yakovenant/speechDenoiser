import os
import speechmetrics as sm
import soundfile as sf
import numpy as np
#from pyloudnorm.normalize import peak


def main(dir_test, dir_targ, metric, prefix):

    print('\nTrying RELATIVE metrics:')
    method = f'relative.{metric}'
    metrics = sm.load(method)
    res = []
    for cur_test_name in os.listdir(dir_test):

        cur_test_abs_path = os.path.normpath(os.path.join(dir_test, cur_test_name))
        print('Computing scores for ', cur_test_abs_path)
        cur_test_wave, cur_test_sr = sf.read(cur_test_abs_path)
        # peak normalize audio to -1 dB
        #cur_test_wave = peak(cur_test_wave, -1.0)

        if dir_targ is None:
            cur_targ_wave = np.random.uniform(low=-0.00001, high=0.00001, size=(len(cur_test_wave),))
        else:
            cur_targ_abs_path = os.path.join(dir_targ, cur_test_name.split('.')[0][:-len(prefix)-1]+'.'+cur_test_name.split('.')[1])
            if not os.path.isfile(cur_targ_abs_path):
                cur_targ_abs_path = cur_targ_abs_path[:-4] + '.WAV'
                if not os.path.isfile(cur_targ_abs_path):
                    print(f'\nSKIP absent file: {cur_targ_abs_path} :(\n')
                    continue
            cur_targ_wave, cur_targ_sr = sf.read(cur_targ_abs_path)
            assert cur_targ_sr == cur_test_sr
            #assert len(cur_targ_wave) == len(cur_test_wave)

        scores = metrics(cur_targ_wave, cur_test_wave, rate=cur_test_sr)
        res.append(scores[metric][0])
    avg_score = sum(res) / len(res)
    print(f'\nAverage {metric}: {avg_score}')


if __name__ == '__main__':

    model_name = 'convtasnet'
    snr_dB = 0
    metric = 'pesq'  # pesq, stoi
    dir_root = '/media/ssd/TIMIT/DENOISER'

    prefix = f'{str(snr_dB)}dBsnr_{model_name}'
    dir_test = f'{dir_root}/SNR_{prefix}'
    dir_targ = f'{dir_root}/target'

    main(dir_test, dir_targ, metric, prefix)
    print('\nDONE!\n')
