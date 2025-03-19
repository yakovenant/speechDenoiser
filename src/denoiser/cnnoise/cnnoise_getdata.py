import os
import tensorflow as tf
import pickle

from random import shuffle
from soundfile import read, write
from sklearn.model_selection import train_test_split
from src.processing import get_features
from src.denoiser.cnnoise_train import get_config_feats


def noise_mix(clean, noise, min_amp, max_amp):  # todo?
    """
    Mix clean and noise audio signals.
    :param clean:       clean speech audio samples.
    :param noise:       noise audio samples.
    :param min_amp:     min ratio of noise power to speech power.
    :param max_amp:     max ratio of noise power to speech power.
    :return:            mixed audio samples.
    """

    # noise level is rand from min_amp to max_amp
    noise_amp = np.random.uniform(min_amp, max_amp)
    # if length of noise audio samples <  speech audio samples => duplicate noise
    noise = noise.repeat(1, clean.shape[1] // noise.shape[1] + 2)
    # get random start index for noise clip from noise audio samples
    idx_start = np.random.randint(0, noise.shape[1] - clean.shape[1] + 1)
    noise_clip = noise[:, idx_start:idx_start + clean.shape[1]]
    # add noise mix
    noise_mix = clean.abs().max() / noise_clip.abs().max() * noise_amp
    return (clean + noise_clip * noise_mix) / (1 + noise_amp)


def save_pickle(path, name, params):
    name = name + '.pickle'
    with open(os.path.join(path, name), 'bw') as handle:
        pickle.dump(params, handle)  # protocol=4


def check_dir(feat_dir):
    if not os.path.isdir(feat_dir):
        os.makedirs(feat_dir)

    if not os.path.isdir(os.path.join(feat_dir, 'train')):
        os.makedirs(os.path.join(feat_dir, 'train'))

    if not os.path.isdir(os.path.join(feat_dir, 'valid')):
        os.makedirs(os.path.join(feat_dir, 'valid'))


def init_data_dict():
    """Dictionary for training data. Contains predictors and targets."""
    data = {
        0: [],  # predictors
        1: [],  # targets
    }
    return data


def extract_features_and_split_batches(batch_size_max_gb, param_feats, list_of_audio_files, dir_save):
    """
    :arg:
        batch_size_max:
        param_feats:
        list_of_audio_files:
        dir_save:
    :return:
    """

    # max features size in memory
    batch_size_max_bytes = float(batch_size_max_gb)  # float(batch_size_max_gb) * 1e+9 / 2

    # initialise dict to save batches
    data_batch = init_data_dict()
    X = {
        'features': [],
        'xspectra': [],
        'segments': []
    }
    y = {
        'features': [],
        'xspectra': [],
        'segments': []
    }

    #batch_xspectra_X = []
    #batch_features_X = []
    batch_segments_X = []

    #batch_xspectra_y = []
    #batch_features_y = []
    batch_segments_y = []

    batch_number = 0
    idx_of_first_audio_file_in_batch = 0

    for audio_file in list_of_audio_files:
        print(audio_file)
        # LOAD AUDIO
        inSig, inFs = read(audio_file)
        inSig, fs = get_features.srCheck(inSig, inFs, param_feats['fs'])
        if 0:
            if audio_file.split('/')[-2] == 'clean':
                inSig = get_features.normalizeVolume(inSig)
                inSig = get_features.removeSilence(inSig)
                write(os.path.join(dir_save, audio_file.split('/')[-1]), inSig, fs)
            continue
        else:
            if audio_file.split('/')[-2] == 'clean':
                inSig = get_features.removeSilence(inSig)
            inSig = get_features.normalizeVolume(inSig)
        # AUGMENT CLEAN SPEECH WITH NOISE todo?
        # FEATURE EXTRACTION
        inSpecs_X = get_features.sig2spec(inSig, param_feats)
        inFeats_X = get_features.spec2feats(inSpecs_X, param_feats)
        inSegs_X, inSegs_y = get_features.shapeSegs(
            X=inFeats_X,
            y=inFeats_X,
            num_fr_seg=param_feats['nframes'])
        assert inSegs_X.shape[0] == inSegs_y.shape[0]
        # COMPOSE BATCH
        #batch_xspectra_X.append(inSpecs_X)
        #batch_features_X.append(inFeats_X)
        batch_segments_X.append(inSegs_X)
        batch_segments_y.append(inSegs_y)
        # CHECK TOTAL SIZE OF DATA SUBSET
        batch_size_bytes = batch_segments_X.__sizeof__() + batch_segments_y.__sizeof__()
            #inSpecs_X.__sizeof__() + \
            #inFeats_X.__sizeof__() + \
            #inSegs_X.__sizeof__()
        print('\nCurrent batch size: ' + str(batch_size_bytes) + '\n')  # str(round(batch_size_bytes / 1e+9, 5))
        if (batch_size_bytes > batch_size_max_bytes) or (audio_file == list_of_audio_files[-1]):
            # X['features'] = np.concatenate(batch_features_X, axis=1)
            # X['amplitudes'] = np.concatenate(batch_amplitudes_X, axis=1)
            # X['phases'] = np.concatenate(batch_phases_X, axis=1)
            X['segments'] = tf.concat(batch_segments_X, axis=0)
            # print('Features shape: ', X['features'].shape)
            # print('Amplitudes shape: ', X['amplitudes'].shape)
            # print('Phases shape: ', X['phases'].shape)
            print('Predictor segments shape: ', X['segments'].shape)
            # Replace targets for noise with zeros
            list_of_audio_files_in_batch = list_of_audio_files[
                                           idx_of_first_audio_file_in_batch:list_of_audio_files.index(audio_file) + 1]
            for audio_file_in_batch in list_of_audio_files_in_batch:
                audio_type = audio_file_in_batch.split('/')[-2]
                '''if audio_type == 'clean':
                    batch_amplitudes_y.append(batch_amplitudes_X[list_of_audio_files_in_batch.index(audio_file_in_batch)])
                    batch_phases_y.append(batch_phases_X[list_of_audio_files_in_batch.index(audio_file_in_batch)])
                    batch_features_y.append(batch_features_X[list_of_audio_files_in_batch.index(audio_file_in_batch)])
                    batch_segments_y.append(
                        dn.format_data(
                            np.expand_dims(batch_segments_X[list_of_audio_files_in_batch.index(audio_file_in_batch)][:, 0, :, 0].copy(), axis=1),
                            params_feat['data_format']
                        )
                    )'''
                if audio_type == 'noise':  # tf.zeros((specsize, N_frames, M), dtype=tf.dtypes.complex64)
                    #batch_xspectra_y.append(
                    #    tf.zeros(batch_xspectra_X[list_of_audio_files_in_batch.index(audio_file_in_batch)].shape))
                    #batch_features_y.append(
                    #    tf.zeros(batch_features_X[list_of_audio_files_in_batch.index(audio_file_in_batch)].shape))
                    '''batch_segments_y.append(
                        dn.format_data(
                            np.expand_dims(np.zeros(
                                shape=(
                                    batch_features_X[list_of_audio_files_in_batch.index(audio_file_in_batch)].shape[1],
                                    batch_features_X[list_of_audio_files_in_batch.index(audio_file_in_batch)].shape[0]
                                )
                            ).astype(np.float32), axis=1), params_feat['data_format']
                        )
                    )'''
                    batch_segments_y[list_of_audio_files_in_batch.index(audio_file_in_batch)] *= 0
                    batch_segments_y[list_of_audio_files_in_batch.index(audio_file_in_batch)] -= 12
                else:  # todo?
                    batch_segments_y[list_of_audio_files_in_batch.index(audio_file_in_batch)] += 6
            # y['features'] = np.concatenate(batch_features_y, axis=1)
            # y['amplitudes'] = np.concatenate(batch_amplitudes_y, axis=1)
            # y['phases'] = np.concatenate(batch_phases_y, axis=1)
            y['segments'] = tf.concat(batch_segments_y, axis=0)
            # print('Features shape: ', y['features'].shape)
            # print('Amplitudes shape: ', y['amplitudes'].shape)
            # print('Phases shape: ', y['phases'].shape)
            print('Target segments shape: ', y['segments'].shape)

            assert X['segments'].shape[0] == y['segments'].shape[0]

            # Save batch
            data_batch[0].append(X['segments'])  # predictors
            data_batch[1].append(y['segments'])  # targets
            save_pickle(dir_save, ('Xy_' + str(batch_number)), data_batch)
            print('\nTraining batch ' + str(batch_number) + ' successfully saved!\n\n.')

            # Reset batch dict
            data_batch = init_data_dict()

            # Reset locals
            idx_of_first_audio_file_in_batch = list_of_audio_files.index(list_of_audio_files_in_batch[-1]) + 1
            batch_number += 1
            list_of_audio_files_in_batch.clear()
            # predictors
            #batch_xspectra_X.clear()
            #batch_features_X.clear()
            batch_segments_X.clear()
            # targets
            #batch_xspectra_y.clear()
            #batch_features_y.clear()
            batch_segments_y.clear()
    print('\nDone!')
    return X['segments'].shape[2]


def get_features_name(params):
    """Parsing full name of features"""
    # nseg = params['segdur'] * params['fs']
    name_feats = \
        str(int(params['float'])) + 'fp_' + \
        str(int(params['fs'])) + 'hz_' + \
        str(int(params['nfft'])) + 'nfft_' + \
        str(int(params['hoplen'])) + 'nhop_' + \
        params["feattype"]
    name_params = '_' + str(int(params['nframes'])) + 'F' + '_' + str(int(params['nfeats'])) + 'B'
    if 0:
        name_params = name_params + '_S8hN2h'  # additional info about data, change it manual!
    return name_feats + name_params


def main(param_feats, name_features, dir_audio_train):
    # Setting up the main paths
    root_project = os.path.join('/', *os.path.dirname(__file__).split('/')[:-2])
    root_features = os.path.join(root_project, 'features')  # path to save features
    # Define path to training audio files
    dir_audio_train_clean = os.path.join(dir_audio_train, 'clean')
    dir_audio_train_noise = os.path.join(dir_audio_train, 'noise')
    # Generate list of all audio inputs
    list_audio_total = []
    for _audio in os.listdir(dir_audio_train_clean):
        list_audio_total.append(os.path.join(dir_audio_train_clean + '/' + _audio))
    for _audio in os.listdir(dir_audio_train_noise):
        list_audio_total.append(os.path.join(dir_audio_train_noise + '/' + _audio))
    shuffle(list_audio_total)

    # Separate training and testing subsets of audio inputs
    list_audio_train, list_audio_valid = train_test_split(list_audio_total, test_size=0.05, random_state=1)

    # Set saving dirs
    dir_save_features = os.path.join(root_features, name_features, dir_audio_train.split('/')[-1])
    if not os.path.isdir(dir_save_features):
        os.makedirs(dir_save_features)
    check_dir(dir_save_features)
    dir_save_train = os.path.join(dir_save_features, 'train')
    dir_save_valid = os.path.join(dir_save_features, 'valid')
    print(f'Features dir: {dir_save_features}')

    # Training data processing
    print('\nRun training data generation and saving')
    n_features = extract_features_and_split_batches(
        batch_size_max_gb=2345,
        param_feats=param_feats,
        list_of_audio_files=list_audio_train,
        dir_save=dir_save_train)
    # Validation data processing
    print('\nRun validation data generation and saving')
    n_features = extract_features_and_split_batches(
        batch_size_max_gb=2345,  # 0.0001
        param_feats=param_feats,
        list_of_audio_files=list_audio_valid,
        dir_save=dir_save_valid)


if __name__ == '__main__':
    # Specify and check params
    param_feats = get_config_feats()
    # Parsing input segment characteristics into the name of dataset
    name_features = get_features_name(param_feats)
    print(f'\nFeatures name: {name_features}')
    # Run data generation and saving
    main(
        param_feats=param_feats,      # config for feature extraction
        name_features=name_features,  # name of features
        dir_audio_train="/home/ayakovenko/datasets/16kHzEN8hRU5hN2h"
        # dir_audio_train="/home/ayakovenko/datasets/dnsc_ru"
        #dir_audio_train="/home/iakovenant/datasets/audio/custom/16kHzEN8hRU5hN2h"
        # dir_audio_train="/home/iakovenant/datasets/audio/test/tmp"
        # dir_audio_train="/home/iakovenant/datasets/audio/custom/dnsc_ru"
    )
    print('\nTraining and validation data successful generated!')
