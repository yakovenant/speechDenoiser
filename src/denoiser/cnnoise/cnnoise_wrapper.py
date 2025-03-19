import tensorflow as tf
from os.path import dirname, join
from src.processing import get_features, get_params


class CNNoiseWrapper(object):
    """CNNoise custom enhancer class."""
    def __init__(self, modelfile, fs):
        """Instantiate FullyConvNet given a trained model path."""

        # self.cfg = self._config(fs)
        self.cfg = get_params.features_cnnoise_new(fs)
        #self.cfg = get_params.features_cnnoise_230906(fs)
        self.frameShift = float(self.cfg['windur']) * float(self.cfg["hopfrac"])
        self.fs = int(self.cfg['fs'])
        self.mingain = 10 ** (self.cfg['mingain'] / 20)
        self.N_win = int(float(self.cfg['windur']) * self.fs)
        self.win = tf.math.sqrt(tf.signal.hann_window(self.N_win))
        if 'nfft' in self.cfg:
            self.N_fft = int(self.cfg['nfft'])
        else:
            self.N_fft = self.N_win
        self.N_hop = int(self.N_fft * float(self.cfg["hopfrac"]))
        self.dtype = tf.float32
        modelfile = join('/', *dirname(__file__).split('/')[:-2], 'models', 'tflite', modelfile)

        # load TF Lite model
        """
        self.interpreter = tf.lite.Interpreter(modelfile)
        self.interpreter.allocate_tensors()
        self.details_inp = self.interpreter.get_input_details()[0]
        self.details_out = self.interpreter.get_output_details()[0]
        """

        # load TF model
        self.model = tf.keras.models.load_model(modelfile)

    '''def _config(self, fs):
        cfg = {
            'fs': tf.constant(fs),
            'segdur': tf.constant(0.3),
            'windur': tf.constant(0.02),
            'hopfrac': tf.constant(0.5),
            'mingain': tf.constant(-80),  # -80, -40, -25, -12
            'feattype': 'LogPow',
            'nfft': tf.constant(320),
            'float': tf.constant(32),
        }
        cfg['winlen'] = cfg["windur"] * fs
        cfg['hoplen'] = tf.constant(int(cfg['winlen'] * cfg["hopfrac"]))
        cfg['winlen'] = tf.constant(int(cfg['winlen']))
        cfg['seglen'] = tf.constant(int(cfg['segdur'] * fs))
        cfg['nfeats'] = tf.constant(int(cfg['nfft'] / 2 + 1))  # N of feature bins
        cfg['nframes'] = tf.constant(
            int((cfg['seglen'] + cfg['winlen'] - cfg['hoplen']) / cfg['hoplen']))  # N of frames in segment
        return cfg'''

    '''def enhance_tflite(self, x):
        """Obtain the estimated filter."""
        self.interpreter.set_tensor(self.details_inp['index'], x.astype(self.details_inp["dtype"]))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.details_out['index'])[0]'''

    def denoise(self, x):
        """
        Obtain the estimated filter.
        :arg: x (EagerTensor, float32): input features [segments x frames x features x channels]
        :return: EagerTensor, float32: model output predictions [features x frames]
        """
        # Apply model
        y = self.model.predict(x)  # ndarray [ frames x 1 x features x 1 ]
        # Reduce dims
        return tf.squeeze(y)  # tensor 2d [frames x features]

    def __call__(self, sig_in, fs_in):
        """Enhance a single audio signal."""

        # Check sampling rate and resample
        sig_in, _ = get_features.srCheck(sig_in, fs_in, self.fs)

        # Calculate spectra
        xspec_in = get_features.sig2spec(sig_in, self.cfg)

        # Extract features
        feats_in = get_features.spec2feats(xspec_in, self.cfg)
        # Input segments (segs_in) shape: [batch x time x freq x channel]
        if 0:  # Real-time test
            segs_in = tf.expand_dims(tf.transpose(feats_in), axis=(0, -1))
            out = tf.expand_dims(self.denoise(segs_in), axis=-1)
        else:
            # Segment features for real-time processing simulation
            segs_in, _ = get_features.shapeSegs(  # 4-d tensor [total_frames x segment_frames x features x 1 ]
                X=feats_in,
                y=None,
                num_fr_seg=self.cfg['nframes'])

            # Obtain network output from input segments
            out = self.denoise(segs_in)  # 2-d tensor [ features x total_frames ]

        # Emphasise output
        '''out = get_features.emphasiseOut(
            X=out,
            start_bin=len(out)//2,
            limit=111,
            factor=1
        )'''

        # Get output spectra
        if 0:  # Real-time test
            xspec_out = get_features.masking_float_rt(xspec_in, out, self.mingain)
        else:
            # xspec_out = self.masking(xspec_in, out)  # 2-d tensor of complex spectra [ features x total_frames ]
            xspec_out = get_features.masking_float(xspec_in, out, self.mingain)

        # Go back to time domain
        if 0:  # Real-time test
            sig_out = get_features.spec2sig_rt(xspec_out, self.cfg)
        else:
            sig_out = get_features.spec2sig(xspec_out, self.cfg)

        # Check sampling rate and resample
        sig_out, _ = get_features.srCheck(sig_out, self.fs, fs_in)

        print('Processing is done.\n')
        return fs_in, sig_out


if __name__ == '__main__':
    print('\nThis is CNNoise wrapper class.\n')
