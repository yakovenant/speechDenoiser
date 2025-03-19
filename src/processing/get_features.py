import numpy as np
import tensorflow as tf
from resampy import resample
#from librosa.effects import split


def emphasiseOut(X, start_bin, limit, factor):
    n_bins_gained = len(X) - start_bin
    emphasis_law = np.linspace(factor, limit, n_bins_gained)  # linear growth
    emphaser = np.concatenate((np.ones(shape=start_bin).astype(np.float32), emphasis_law), axis=0)
    emphaser = np.expand_dims(emphaser, axis=1)
    '''
    import matplotlib.pyplot as plt
    plt.plot(emphaser)
    plt.show()
    '''
    return emphaser * X


def removeSilence(samples_in):
    print("Silence removal...")
    clips = split(samples_in, top_db=30)
    samples_out = []
    for c in clips:
        data = samples_in[c[0]:c[1]]
        samples_out.extend(data)
    return np.array(samples_out)


def normalizeVolume(samples_in):
    print("Volume normalization...")
    samples_in /= np.max(samples_in)
    return samples_in


def scaleWaveform(samples_in, max_val, min_val):
    print("Waveform scaling...")
    return (samples_in - np.min(samples_in)) / (np.max(samples_in) - np.min(samples_in)) * (max_val - min_val) + min_val


def convert_to_16_bit_wav(data):
    if data.dtype == np.float32:
        print(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        #data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        print(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint8:
        print(
            "Audio data is not in 16-bit integer format."
            "Trying to convert to 16-bit int format."
        )
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError("Audio data cannot be converted to " "16-bit int format.")
    return data


def checkWaveform(x, sr_input, sr_target):
    ...
    if sr_input == 48000:
        if str(x.dtype) == 'int32':
            #x_int16_khz48 = convert_to_16_bit_wav(x)

            x_float = x.astype(np.float32, order='C') / 2147483647.0
            # y0 = tf.stack([float(val) / pow(2, 15) for val in x]).numpy()

            #y1 = ((x / max(np.max(x), 1)) * 32768).reshape((-1, 3)).mean(axis=1).astype('float32')
            #y2 = y1.astype(np.float32, order='C') / 32768.0

            y = resample(x_float, sr_input, sr_target)

            #y = scaleWaveform(y2, np.max(y0), np.min(y0))

            #x_float32 = x_int16_khz48.astype(np.float32, order='C') / 32768.0
            #y = librosa.resample(x_float32, orig_sr=sr_input, target_sr=sr_target)
        else:
            ...
    elif sr_input == 16000:
        x = x.numpy()
        if str(x.dtype) == 'float32':
            x_resampled = resample(x, sr_input, sr_target)
            y = convert_to_16_bit_wav(x_resampled)
        else:
            ...
    else:
        ...
    return y, sr_target


def srCheck(x, sr_input, sr_target):
    """Check sampling rate and resample."""
    assert sr_input and sr_target in (16000, 48000), "Inconsistent sampling rate!"
    if sr_input != sr_target:
        print("Input and target SR are different, run resampling...")
        if type(x).__module__ == 'numpy':
            y = resample(x, sr_input, sr_target)
            return y, sr_target
        else:
            y = resample(x.numpy(), sr_input, sr_target)
            return y, sr_target
    else:
        return x, sr_input


def formatData(X, data_format='channels_last'):
    """Format data for CPU or GPU inference."""
    if data_format == 'channels_first':
        raise Exception("For CPU model must be in NHWC format!")
        # return np.expand_dims(X, 1)
    elif data_format == 'channels_last':
        return tf.expand_dims(X, 3)
    else:
        raise ValueError('Wrong data format: ', data_format)


def shapeSegs(X, y=None, num_fr_seg=31):
    """Shape input segments for real-time simulation."""
    print('Shaping segments...')
    # X = tf.transpose(X)
    # X = tf.concat([X[:, :num_fr_seg], X], 1)
    X = tf.concat([X[:num_fr_seg, :], X], 0)
    time_steps = X.shape[0] - num_fr_seg
    # num_features = X.shape.as_list()[0]
    # X_segmented = np.zeros((time_steps, num_fr_seg, num_features)).astype(np.float32)
    X_segmented = []
    y_segmented = None

    for step in range(time_steps):
        # Segment preprocessing
        segment = X[step:(step + num_fr_seg), :]
        #segment += 6
        # X_segmented[step, :, :] = tf.transpose(segment, (1, 0))
        # X_segmented.append(tf.transpose(segment, (1, 0)))
        X_segmented.append(segment)
    X_segmented = tf.stack(X_segmented)
    # Format input data
    X_segmented = formatData(X_segmented)
    print('Segmented predictors dims: ', X_segmented.shape)
    print('min:', float(tf.math.reduce_min(X_segmented)), 'max:', float(tf.math.reduce_max(X_segmented)))
    if y is not None:
        y_segmented = tf.expand_dims(y, axis=1)
        y_segmented = formatData(y_segmented)
        print('Segmented targets dims: ', y_segmented.shape)
        print('min:', float(tf.math.reduce_min(y_segmented)), 'max:', float(tf.math.reduce_max(y_segmented)))
    # print('Segmentation is done.')
    return X_segmented, y_segmented


def spec2feats(spec, params):
    """
    Compute spectral features.
    :arg:
        spec (EagerTensor, complex64): spectrogram [features x frames]
        params (dict): parameters of feature extraction
    :return:
        EagerTensor, float32: amplitudes of log power spectra or magnitude spectra
    """
    def _getLog10(x):
        """
        :param: x (EagerTensor, float32): amplitude spectra
        :return: EagerTensor, float32: log power spectra or magnitude spectra
        """
        numerator = tf.math.log(x)  # 2-d tensor [ 161 x N_frames ]
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))  # constant value
        return numerator / denominator
    if params['feattype'] == "MagSpec":
        return tf.abs(spec)
    elif params['feattype'] == "LogPow":
        pmin = tf.math.pow(10., -12)
        powSpec = tf.math.pow(tf.abs(spec), 2)  # 2-d tensor of amplitude spectra features by frames
        return _getLog10(tf.math.maximum(powSpec, pmin))
    else:
        return ValueError('Feature not implemented.')


def sig2spec(y, params, channel=None):
    """
    Converts input audio time-domain signal into the spectrogram.
    :arg:
        y (ndarray, float64): time domain signal [samples x channels]
        params (dict): parameters of STFT
    :return:
        EagerTensor, tf.complex64: spectrogram [features x frames]
    """
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    if channel is not None and (len(y.shape) > 1):
        # use only single channel
        y = y[:, channel]
    return tf_stft(y, params)


def spec2sig(Spec, params):
    """
    Converts output spectrum into the time-domain audio signal.
    :arg:
        Spec (EagerTensor, complex64): output complex spectra [features x frames]
        params (dict): parameters of ISTFT
    :return:
        EagerTensor, float32: output time-domain audio signal [samples x channels]
    """
    return tf_istft(Spec, params)


def tf_stft(x, params):
    """
    Short-time Fourier transform using TensorFlow.
    :arg:
        x (EagerTensor, float32): time domain signal [samples x channels]
        params (dict): parameters of STFT
    :return:
        EagerTensor, complex64: spectrogram [frames x features]
    """
    # get lengths
    Nx = x.shape[0]  # number of samples in audio signal
    # specsize = int(params['nfft'] / 2 + 1)  # number of spectral features
    N_frames = int(tf.math.ceil((Nx + params['winlen'] - params['hoplen']) / params['hoplen']))  # number of frames in audio signal
    Nx = N_frames * params['hoplen']  # padded length of audio signal in samples
    x = tf.concat([x, tf.zeros((Nx - len(x)), dtype=tf.dtypes.float32)], 0)  # 1-d tensor of samples padded with zeros
    # init
    win = tf.math.sqrt(tf.signal.hann_window(params['winlen'], periodic=True, dtype=tf.dtypes.float32))  # 1-d tensor window function with 320 samples length
    win_M = tf.squeeze(tf.einsum('i,j->ij', win, tf.ones(1, dtype=tf.dtypes.float32)))  # 1-d tensor of dot-product window with 320 samples length
    x_frames = tf.signal.frame(x, frame_length=params['winlen'], frame_step=params['hoplen'])  # 2-d tensor of frames [ N_frames - 1 x 320 ]
    # run
    X_spec = []  # list of spectral features by frames
    for i, x_frame in enumerate(x_frames):
        # x_frame = tf.squeeze(tf.slice(x_frames, [i, 0], [1, -1]))
        x_win = tf.math.multiply(win_M, x_frame)  # 1-d tensor of windowed signal with 320 samples
        X = tf.signal.rfft(x_win)  # 1-d tensor of complex spectra with 161 spectral features
        X_spec.append(X)
    return tf.stack(X_spec)  # 2-d tensor of complex spectra features by frames [ N_frames x specsize ]


def tf_istft(X, params):
    """
    Inverse short-time Fourier transform using TensorFlow.
    :arg:
        X (EagerTensor, complex64): output complex spectra [features x frames]
        params (dict): parameters of ISTFT
    :return:
        output (EagerTensor, float32): output time-domain audio signal [samples x channels]
    """
    # get lengths
    # specsize = X.shape[0]  # number of spectral features (N_features) = 161
    '''N_frames = X.shape[1]  # number of frames
    if X.ndim < 3:
        X = X[:, :, tf.newaxis]'''
    M = 1  # X.shape[2]  # mono channel of output audio, M = 1
    N_win = int(params['winlen'])  # number of samples in window = 320
    N_hop = int(params['hoplen'])  # number of samples in window step = 160
    N_fft = int(params['nfft'])  # number of samples in FFT
    # Nx = N_hop * (N_frames - 1) + N_win  # number of samples in output audio
    # init
    win = tf.math.sqrt(tf.signal.hann_window(params['winlen'], dtype=tf.dtypes.float32))  # 1-d tensor of window function with 320 samples length
    win_M = tf.einsum('i,j->ij', win, tf.ones(1, dtype=tf.dtypes.float32))  # 2-d tensor of dot-product window [ 320 x 1 ]
    # run
    output = 0
    #for i in range(0, N_frames):
    #    X_frame = tf.squeeze(X[:, i, :])  # 1-d tensor of audio frame with 161 features
    for i, X_frame in enumerate(X):
        x_win = tf.signal.irfft(X_frame)  # 1-d istft tensor = 320
        x_win = tf.reshape(x_win, [N_fft, M])  # 2-d istft tensor [ 320 x 1 ]
        x_win = win_M * x_win[0:N_win, :]  # 2-d windowed tensor [ 320 x 1 ]
        if i != 0:
            x = x[N_hop:N_win+N_hop, :]  # 2-d tensor [ 320 x 1 ]
            x += x_win  # 2-d tensor [ 320 x 1 ]
            output = tf.concat([output, x[0:N_hop, :]], 0)  # 2-d output audio samples tensor [ 320 x 1 ]
            x = tf.concat([x, tf.zeros((N_hop, M), dtype=tf.dtypes.float32)], 0)  # 2-d tensor [ 480 x 1 ]
        else:
            x = tf.concat([x_win, tf.zeros((N_hop, M), dtype=tf.dtypes.float32)], 0)  # 2-d tensor of padded samples [ 480 x 1 ]
            output = x[0:N_hop, :]  # 2-d output audio samples tensor [ 161 x 1 ]
    if M == 1:
        output = tf.squeeze(output)  # 1-d output audio samples tensor
    return output


def spec2sig_rt(xspec_out, params):
    """
    Real-time simulation of spectra to signal transform.
    :param xspec_out: 2-d complex tensor [ features x frames ]. In web: [ 161 x 1 ]
    :param params: dict of ISTFT parameters
    :return: 1-d output audio samples tensor
    """
    # get constants
    output_samples_total = 0  # output (denoised) audio stream samples
    # output_samples_current = 0  # output prediction samples for current audio segment
    output_samples_previous = 0  # output prediction samples for previous audio segment
    M = 1  # mono channel of output audio
    N_win = int(params['winlen'])  # number of samples in window = 320
    N_hop = int(params['hoplen'])  # number of samples in window step = 160
    N_fft = int(params['nfft'])  # number of samples in FFT
    # init window function
    win = tf.math.sqrt(tf.signal.hann_window(N_win, dtype=tf.dtypes.float32))  # 1-d tensor of window function with 320 samples length
    win_M = tf.einsum('i,j->ij', win, tf.ones(1, dtype=tf.dtypes.float32))  # 2-d tensor of dot-product window [ 320 x 1 ]
    # loop through frames
    for output_frame_idx, output_frame in enumerate(xspec_out):  # output_frame: 1-d complex tensor 161 features length
        if output_frame_idx > 0:  # previous output predictions exist
            output_samples_current = tf_istft_rt(output_frame, M, N_win, N_hop, N_fft, win_M, output_samples_previous)  # 2-d tensor [ 480 x 1 ]
            # output_samples_current_sliced = output_samples_current[0:N_hop, :]
            output_samples_current_sliced = tf.slice(output_samples_current, [0, 0], [N_hop, M])
            output_samples_total = tf.concat([output_samples_total, output_samples_current_sliced], 0)  # 2-d tensor [ N+160 x 1 ]
            output_samples_previous = output_samples_current  # 2-d tensor [ 480 x 1 ]
        else:  # previous output predictions don't exist
            output_samples_current = tf_istft_rt(output_frame, M, N_win, N_hop, N_fft, win_M)  # 2-d tensor [ 480 x 1 ]
            # output_samples_current_sliced = output_samples_current[0:N_hop, :]
            output_samples_current_sliced = tf.slice(output_samples_current, [0, 0], [N_hop, M])
            output_samples_total = output_samples_current_sliced  # 2-d tensor [ 160 x 1 ]
            output_samples_previous = output_samples_current  # 2-d tensor [ 480 x 1 ]
    return tf.squeeze(output_samples_total)


def tf_istft_rt(current_frame, M, N_win, N_hop, N_fft, win_M, previous_samples=None):
    """
    Real-time simulation of istft.
    :return: 1-d output audio samples tensor
    """
    current_frame_irfft = tf.reshape(tf.signal.irfft(current_frame), [N_fft, M])  # 2-d istft tensor [ 320 x 1 ]
    current_samples = tf.math.multiply(win_M, current_frame_irfft)  # 2-d windowed tensor [ 320 x 1 ]
    if previous_samples is not None:
        # previous_samples_sliced = previous_samples[N_hop:N_win+N_hop, :]
        previous_samples_sliced = tf.slice(previous_samples, [N_hop, 0], [N_win, M])
        output_samples_windowed = tf.math.add(current_samples, previous_samples_sliced)  # 2-d tensor [ 320 x 1 ]
        output_samples = tf.concat([output_samples_windowed, tf.zeros((N_hop, M), dtype=tf.dtypes.float32)], 0)  # 2-d tensor [ 480 x 1 ]
    else:
        output_samples = tf.concat([current_samples, tf.zeros((N_hop, M), dtype=tf.dtypes.float32)], 0)  # 2-d tensor [ 480 x 1 ]
    return output_samples


def masking_complex(input_xspec, predicted_aspec, min_gain):
    """
    Apply spectral masking for complex input.
    :arg:
        input_xspec (EagerTensor, complex64): input complex spectra [features x frames]
        predicted_aspec (EagerTensor, float32): predicted amplitude spectra [features x frames]
        min_gain (EagerTensor, float64): minimal gain
    :return:
        EagerTensor, complex64: output complex spectra [features x frames]
    """
    # Limit suppression gain
    gain = tf.clip_by_value(
        predicted_aspec,
        clip_value_min=tf.cast(min_gain, tf.float32),
        clip_value_max=1.0)
    # Apply spectral mask
    return input_xspec * tf.cast(gain, tf.complex64)


def masking_float(input_xspec, predicted_aspec, min_gain):
    """
    Apply spectral masking for input amplitude spectra.
    :arg:
        input_xspec (EagerTensor, complex64): input complex spectra [features x frames]
        predicted_aspec (EagerTensor, float32): predicted amplitude spectra [features x frames]
        min_gain (EagerTensor, float64): minimal gain
    :return:
        EagerTensor, complex64: output complex spectra [features x frames]
    """
    # Limit suppression gain
    gain = tf.clip_by_value(
        predicted_aspec,
        clip_value_min=tf.cast(min_gain, tf.float32),
        clip_value_max=1.0)
    # Get real and imaginary part of complex spectra
    input_aspec = tf.math.real(input_xspec)
    input_phase = tf.math.imag(input_xspec)
    # Apply spectral mask
    return tf.dtypes.complex((tf.math.multiply(input_aspec, gain)), (tf.math.multiply(input_phase, gain)))


def masking_float_rt(input_xspec, predicted_aspec, min_gain):
    """
    Real-time simulation of spectral masking for input amplitude spectra.
    :arg:
        input_xspec (EagerTensor, complex64): input complex spectra [features x frames]
        predicted_aspec (EagerTensor, float32): predicted amplitude spectra [features x frames]
        min_gain (EagerTensor, float64): minimal gain
    :return:
        EagerTensor, complex64: output complex spectra [features x frames]
    """
    # Limit suppression gain
    gain = tf.clip_by_value(
        predicted_aspec,
        clip_value_min=tf.cast(min_gain, tf.float32),
        clip_value_max=1.0)
    output_xspec = []
    for i, input_complex_frame in enumerate(input_xspec):
        # Get real and imaginary part of complex spectra
        input_amplitude_frame = tf.math.real(input_complex_frame)
        input_phase_frame = tf.math.imag(input_complex_frame)
        predicted_gain_frame = tf.squeeze(tf.slice(gain, [i, 0], [1, -1]))
        # Apply spectral mask
        output_amplitude_frame = tf.math.multiply(input_amplitude_frame, predicted_gain_frame)
        output_phase_frame = tf.math.multiply(input_phase_frame, predicted_gain_frame)
        output_complex_frame = tf.dtypes.complex(output_amplitude_frame, output_phase_frame)
        output_xspec.append(output_complex_frame)
    return tf.stack(output_xspec)


if __name__ == '__main__':
    print('\nThis is GET FEATURES module.\n')
