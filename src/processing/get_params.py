import tensorflow as tf


def training_cnnoise_new(features_name, dataset_name):  # smaller FFT size + segment length (memory)
    cfg = {
        'name_feats': features_name,
        'name_datas': dataset_name,
        'name_model': "cnnoise",
        'init_lr': 8e-8,
        'lr_schedule': None,  # exponential_decay (for SGD), None
        'minibatch_size': 16,  # 1 32 64 128
        'num_epochs': 50,
        'optimizer': "adam",
        'loss': "mse",  # mse, mae, custom, custom_rmse
        'data_format': "channels_last"
    }
    return cfg


def training_cnnoise_230906(features_name, dataset_name):
    cfg = {
        'name_feats': features_name,
        'name_datas': dataset_name,
        'name_model': "cnnoise",
        'init_lr': 1e-6,
        'lr_schedule': None,
        'minibatch_size': 64,
        'num_epochs': 100,
        'optimizer': "adam",
        'loss': "mse",
        'data_format': "channels_last"
    }
    return cfg


def ms2samples(cfg, fs):
    cfg['winlen'] = cfg['windur'] * fs
    cfg['hoplen'] = tf.constant(int(cfg['winlen'] * cfg["hopfrac"]))
    cfg['winlen'] = tf.constant(int(cfg['winlen']))

    cfg['seglen'] = tf.constant(int(cfg['segdur'] * fs))

    cfg['nfft'] = cfg['winlen']
    cfg['nfeats'] = tf.constant(int(cfg['nfft'] / 2 + 1))  # num spectral feature bins

    cfg['nframes'] = tf.constant(int((cfg['seglen'] + cfg['winlen'] - cfg['hoplen']) / cfg['hoplen'])) - 1  # num frames in seg
    #cfg['nframes'] = tf.constant(int((cfg['seglen'] + cfg['winlen'] - cfg['hoplen']) / cfg['hoplen'] / 2))  # num frames in seg

    return cfg


def features_cnnoise_20230906(fs=16000):
    cfg = {
        'fs': tf.constant(fs),
        'segdur': tf.constant(0.3),
        'windur': tf.constant(0.02),
        'hopfrac': tf.constant(0.5),
        'mingain': tf.constant(-80),  # -80, -40, -25, -12
        'float': tf.constant(32),
        'feattype': 'LogPow',
    }
    cfg_samples = ms2samples(cfg, fs)
    return cfg_samples


def features_cnnoise_20230923(fs=16000):  # smaller FFT size + smaller segment length (memory)
    cfg = {
        'fs': tf.constant(fs),
        'segdur': tf.constant(0.12),
        'windur': tf.constant(0.006),
        'hopfrac': tf.constant(0.5),
        'mingain': tf.constant(-80),  # -80, -40, -25, -12
        'float': tf.constant(32),
        #'feattype': 'LogPow',
        'feattype': 'MagSpec',
    }
    cfg_samples = ms2samples(cfg, fs)
    return cfg_samples


def features_cnnoise_new(fs=16000):  # smaller FFT size + smaller segment length (memory)
    cfg = {
        'fs': tf.constant(fs),
        'segdur': tf.constant(0.12),
        'windur': tf.constant(0.008),
        'hopfrac': tf.constant(0.5),
        'mingain': tf.constant(-80),  # -80, -40, -25, -12
        'float': tf.constant(32),
        #'feattype': 'LogPow',
        'feattype': 'MagSpec',
    }
    cfg_samples = ms2samples(cfg, fs)
    return cfg_samples


if __name__ == '__main__':
    print('\nThis is GET PARAMS module.\n')
