import os
import random
import pickle
import datetime
import warnings
import numpy as np
import soundfile as sf

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

from tqdm import tqdm
from pathlib import Path
# from matplotlib import pyplot as plt
from contextlib import redirect_stdout

from src.processing import get_features, get_params


def prepare_new_dirs(dirs: dict):
    """Creates saving directories if they don't already exist."""

    if not os.path.isdir(dirs['dir_model']):
        os.makedirs(dirs['dir_model'])
    if not os.path.isdir(dirs['dir_model_checks']):
        os.makedirs(dirs['dir_model_checks'])
    if not os.path.isdir(dirs['dir_model_best']):
        os.makedirs(dirs['dir_model_best'])

    if not os.path.isdir(dirs['dir_journal_logs']):
        os.makedirs(dirs['dir_journal_logs'])
    if not os.path.isdir(os.path.join(dirs['dir_journal_logs'], 'pic')):
        os.makedirs(os.path.join(dirs['dir_journal_logs'], 'pic'))
    if not os.path.isdir(os.path.join(dirs['dir_journal_logs'], 'wav')):
        os.makedirs(os.path.join(dirs['dir_journal_logs'], 'wav'))

    if not os.path.isdir(dirs['dir_tb_logs']):
        os.makedirs(dirs['dir_tb_logs'])
    if not os.path.isdir(os.path.join(dirs['dir_tb_logs'], 'train')):
        os.makedirs(os.path.join(dirs['dir_tb_logs'], 'train'))
    if not os.path.isdir(os.path.join(dirs['dir_tb_logs'], 'valid')):
        os.makedirs(os.path.join(dirs['dir_tb_logs'], 'valid'))


def save_model_summary(model, save_dir):
    with open(os.path.join(save_dir, 'model_summary.txt'), 'w') as f:
        with redirect_stdout(f):
            model.summary()


def minibatch_reader(X, Y, minibatch_size):
    """
    Yields mini-batches of (x, y).
    :param X:                features/predictors.
    :param Y:                labels/targets.
    :param minibatch_size:   number of samples in minibatch.
    :return:                 training batch generator.
    """
    num_minibatches = int(tf.math.floor(X.shape[0] / minibatch_size))
    # idxs = list(range(num_minibatches))
    # random.shuffle(idxs)
    for i in range(num_minibatches):
        # for i in idxs:
        current_minibatch = (
            X[i * minibatch_size:min(num_minibatches * minibatch_size, (i + 1) * minibatch_size), :, :, :],
            Y[i * minibatch_size:min(num_minibatches * minibatch_size, (i + 1) * minibatch_size), :, :, :])
        yield current_minibatch


def shuffle(a, b):
    # assert len(a) == len(b)
    # p = np.random.permutation(len(a))
    # return a[p], b[p]
    assert tf.shape(a)[0] == tf.shape(b)[0]
    indices = tf.range(start=0, limit=tf.shape(a)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    a = tf.gather(a, shuffled_indices)
    b = tf.gather(b, shuffled_indices)
    return a, b


def read_pickle(path):
    with open(path, 'br') as handle:
        return pickle.load(handle)


def read_dataset(data_path, partition, batch_index=None):
    """
    Reads a training dataset into dictionaries X and y with keys 'train' and 'valid'.
    :param data_path:    path to folder with data partition.
    :param partition:    data splits to read {'train', 'valid'}.
    :return:
        X (dict):        noisy data (predictors).
        y (dict):        clean data (targets).
    """

    inputs = {}
    amplitudes = {}
    targets = {}
    if batch_index is None:
        # print('Read validation data')
        data = read_pickle(os.path.join(data_path, 'Xy_0.pickle'))
        # use in case of validation or train_debug
        inputs[partition] = data[0][0]  # predictors  # data[0][0]['segments']
        targets[partition] = data[1][0]  # targets  # data[1][0]['segments']
    else:
        print('Reading data...')
        data = read_pickle(os.path.join(data_path, f'Xy_{str(batch_index)}.pickle'))
        inputs[partition] = data[0][0]  # predictors
        targets[partition] = data[1][0]  # targets
    print('Predictors shape: ', inputs[partition].shape)
    print('Targets shape: ', targets[partition].shape)
    return inputs, targets, amplitudes, data


def get_batches(data_path, data_id, data_type, cfg_train):
    """
    Creates a batch reader for data in data_path.
    :param data_path:    path to data files.
    :param data_id:      data file id from [0, 1, 2].
    :param data_type:    'train' or 'test'.
    :param cfg_train:    training config.
    :return:             batch reader (generator object).
    """
    # Read dataset
    input_features, target_features, predictor_amplitudes, data = read_dataset(
        data_path=data_path,
        partition=data_type,
        batch_index=data_id,
    )
    # MiniBatch-learning or online-learning
    if cfg_train['minibatch_size'] == 1:
        print('Shuffling segments disabled for sequential learning process')
    # n_minibatches = input_features[data_type].shape[0]
    else:
        print('Random shuffling segments for batch learning process')
        # if np.isnan(predictor_amplitudes):
        input_features[data_type], target_features[data_type] = shuffle(input_features[data_type], target_features[data_type])
    # Generate mini-batches
    mini_batches = minibatch_reader(
        input_features[data_type],
        # predictor_amplitudes[data_type],
        target_features[data_type],
        cfg_train['minibatch_size'])
    return mini_batches, data


def exponential_decay_schedule(epoch_idx, initial_lr, k=0.1):
    """Updates learning rate."""
    return initial_lr * np.exp(-k * epoch_idx)


def custom_test_step(epoch, batch, model):
    """
    Runs inference on test files and saves output as pics and audios.
    Test features should be already attached to the model together with complex spectra and audio file names.
    :param epoch:     current epoch.
    :param batch:     current batch.
    :param model:     keras model.
    :return:          None
    """
    dt = model.dirs['dt']
    mingain = 10 ** (model.params_feat['mingain'] / 20)
    test_audios = model.test_audio_list
    for i in range(len(test_audios)):
        # Process test predictors
        output = model.predict(model.test_segments[i])
        # limit suppression gain
        out_gain = tf.squeeze(output)
        '''out_gain = tf.clip_by_value(out_gain, clip_value_min=tf.cast(mingain, tf.float32), clip_value_max=1.0)
        # out_gain = np.clip(out_gain, a_min=maingain, a_max=1.0)
        # out_gain = np.squeeze(out_gain)
        # Get output spectra
        out_spec = model.test_xspectra[i] * tf.cast(out_gain, tf.complex64)'''
        out_spec = get_features.masking_float(model.test_xspectra[i], out_gain, mingain)
        # Get output audio
        out_sig = get_features.spec2sig(out_spec, model.params_feat)
        # Save test audio files
        name_file = str(test_audios[i]).split('/')[-1].split('.')[0]
        name_audio = model.params['name_model'] + f'_{dt}_epoch_{epoch}_batch_{batch}.wav'
        audio_path = os.path.join(model.dirs['dir_journal_logs'], 'wav', name_file)
        print('Saving ' + audio_path)
        if not os.path.isdir(audio_path):
            os.makedirs(audio_path)
        sf.write(os.path.join(audio_path, name_audio), out_sig, model.params_feat['fs'])
    # Save test audio figures TODO?
    '''print('Saving test spectrograms...')
    for i in range(len(output_amplitude_spectra)):
        S_db = librosa.amplitude_to_db(output_amplitude_spectra[i], ref=np.max)
        plt.figure()
        librosa.display.specshow(S_db)
        plt.colorbar()
        file_name = model.test_audio_list[i].split('\\')[-1].split('.')[0]
        # db_level = file_name.split('_')[-1]

        pic_name = model.params['model_name'] + f'_{dt}_epoch_{epoch}_batch_{batch}.png'
        pic_path = os.path.join(model.dirs['log_dir_journal'], 'pic', file_name)
        if not os.path.isdir(pic_path):
            os.makedirs(pic_path)
        plt.savefig(os.path.join(pic_path, pic_name))
        plt.close('all')'''


@tf.function
def custom_eval_step(model, batch_inputs, batch_targets, loss_fn):
    """
    Computes validation loss value for current mini-batch.
    :param model:          keras model.
    :param batch_inputs:   ...
    :param batch_targets:  ...
    :param loss_fn:        function to compute the loss.
    :return:               validation loss.
    """
    # Logits for this mini-batch
    logits = model(batch_inputs, training=False)
    return loss_fn(batch_targets, logits)


@tf.function
def custom_train_step(model, batch_inputs, batch_targets, optimizer, loss_fn):
    """
    Runs one step of gradient descent in mini-batches.
    :param model:              keras model.
    :param batch_inputs:       ...
    :param batch_targets:      ...
    :param optimizer:          keras optimizer.
    :param loss_fn:            function to compute the loss.
    :return: loss_val:         loss value.
    """

    # Open a GradientTape to record the operations run	during the forward pass, which enables auto-differentiation.
    with tf.GradientTape(persistent=True) as gt:
        # Forward pass.
        # The operations that the layer applies to its inputs are going to be recorded on the GradientTape.
        gt.watch(model.trainable_weights)
        # Logits for current mini batch
        logits = model(batch_inputs, training=True)
        if tf.reduce_any(tf.math.is_nan(logits)):
            warnings.warn('Logits containing NAN values are replaced with ZEROS!')
            logits = tf.where(tf.math.is_nan(logits), 0., logits)
        elif tf.reduce_any(tf.math.is_inf(logits)):
            warnings.warn('Logits containing INF values are replaced with ONES!')
            logits = tf.where(tf.math.is_inf(logits), 1., logits)
        # Compute the loss value for current mini batch
        loss_val = loss_fn(batch_targets, logits)
    # Use the gradient tape to automatically retrieve the gradients of the trainable variables with respect to the loss.
    grads = gt.gradient(loss_val, model.trainable_weights)
    # Backward pass.
    # Run one step of gradient descent by updating the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_val


def run_training(model):
    """
    Runs training process followed by a validation and a test steps.
    All summaries are saved to tensorboard.
    :param model:   keras model structure.
    :returns:       saving the best trained keras model.
    """
    max_epochs = model.params['num_epochs']
    # Create directories
    prepare_new_dirs(model.dirs)
    # Save model info
    save_model_summary(model, model.dirs['dir_model'])
    # Create tensorboard logs and callbacks
    summary_writer_train = tf.summary.create_file_writer(os.path.join(model.dirs['dir_tb_logs'], 'train'))
    summary_writer_valid = tf.summary.create_file_writer(os.path.join(model.dirs['dir_tb_logs'], 'valid'))
    # Define LR schedule
    if model.params['lr_schedule'] == "exponential_decay":
        print("Using exponential decay learning rate schedule.")
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            model.params['initial_lr'],
            decay_steps=10000,
            decay_rate=0.95,  # 0.86
            staircase=True)
    else:
        print("\nUsing default learning rate schedule.")
        lr_schedule = model.params['init_lr']
    # Define optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        clipnorm=0.001,
    )
    # Define loss
    if model.params['loss'] == 'mse':
        loss_fn = keras.losses.MeanSquaredError()
    elif model.params['loss'] == 'mae':
        loss_fn = keras.losses.MeanAbsoluteError()
    else:
        raise ValueError('Loss function is not defined: ', np.nan)

    # Compile model
    model.compile(optimizer=optimizer, loss=loss_fn)

    # TRAINING LOOP

    lr = model.params['init_lr']
    flag_nan_loss = False
    validation_scores = []
    models = []
    train_loss = 0
    val_loss = 0
    print('\nRun training loop...')
    for epoch in range(max_epochs):
        print('\nEpoch ', epoch, '/', (max_epochs - 1))

        print('\nTRAINING STEP')
        epoch_train_loss = 0  # Loss will be accumulated across all train data files
        # Repeat for each train data file
        for batch_id in range(model.params['num_batches']):
            print('\n... training on batch', batch_id)
            # Read batches
            train_batches, _ = get_batches(
                data_path=model.dirs['dir_data_train'],
                data_id=batch_id,
                data_type='train',
                cfg_train=model.params,
            )
            print('Propagation...')
            # Loss will be accumulated across mini batches
            total_train_loss = 0
            # Iterate over the mini-batches
            for step, (mini_batch_inputs, mini_batch_targets) in tqdm(enumerate(train_batches)):
                # Check for nan data
                assert not tf.math.reduce_any(tf.math.is_nan(mini_batch_inputs))
                assert not tf.math.reduce_any(tf.math.is_nan(mini_batch_targets))
                train_loss = custom_train_step(
                    model=model,
                    batch_inputs=mini_batch_inputs,
                    batch_targets=mini_batch_targets,
                    optimizer=optimizer,
                    loss_fn=loss_fn)
                if tf.math.is_nan(train_loss):
                    warnings.warn('Train loss is NAN! Break training loop.')
                    flag_nan_loss = True
                    break
                # raise ValueError('Train loss is NAN!')
                total_train_loss += train_loss
            if flag_nan_loss:
                break
            total_train_loss = total_train_loss / (step + 1)
            print('Batch loss:', total_train_loss.numpy())
            # Loss is an eager tensor for a single train file
            # Accumulate loss for all train data files
            epoch_train_loss += total_train_loss.numpy()
        '''if flag_nan_loss:
            break'''
        epoch_train_loss = epoch_train_loss / model.params['num_batches']
        print(f'\nEpoch {epoch} training loss: ', epoch_train_loss)

        print('\nVALIDATION STEP\n')
        epoch_val_loss = 0
        # Read batches
        val_batches, val_data = get_batches(
            data_path=model.dirs['dir_data_valid'],
            data_type='valid',
            data_id=None,
            cfg_train=model.params)
        total_val_loss = 0
        for step, (mini_batch_inputs_val, mini_batch_targets_val) in enumerate(val_batches):
            val_loss = custom_eval_step(
                model=model,
                batch_inputs=mini_batch_inputs_val,
                batch_targets=mini_batch_targets_val,
                loss_fn=loss_fn)
            total_val_loss += val_loss
        total_val_loss = total_val_loss / (step + 1)
        # print(f'Batch {batch_id} validation loss: {total_val_loss}')

        epoch_val_loss += total_val_loss.numpy()
        # epoch_val_loss = epoch_val_loss / model.params['num_batches']
        print(f'\nEpoch {epoch} validation loss: ', epoch_val_loss)
        validation_scores.append(epoch_val_loss)

        # Saving summary to tensorboard

        print('\nSaving summary to tensorboard')
        with summary_writer_train.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('learning_rate_custom', lr, step=epoch)
        with summary_writer_valid.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)

        # Saving intermediate model (checkpoints)

        print('\nSaving checkpoint model')
        model_name_epoch = model.params['name_model'] + f'_epoch_{epoch}'
        model.save(os.path.join(model.dirs['dir_model_checks'], model_name_epoch))
        models.append(model)

        # TEST STEP (control) - process external audios without targets
        if (epoch % 2) == 0:
            print('\nTEST STEP')
            custom_test_step(epoch=epoch, batch='full', model=model)
        else:
            print('\nSkip TEST STEP')

    # Save the best model when training loop is complete
    best_epoch_loss = validation_scores.index(np.min(validation_scores))
    print(f'\nMinimum validation loss was obtained at epoch {best_epoch_loss}.')
    best_epoch = best_epoch_loss  # SET THE BEST EPOCH CRITERION
    return models[best_epoch]


def get_model_flops(model, batch_size=None):
    """Calculate floating point operations in the model forward pass."""
    if batch_size is None:
        batch_size = 1
    real_model = tf.function(model).get_concrete_function(
        tf.TensorSpec([batch_size] + model.inputs[0].shape[1:],
                      model.inputs[0].dtype))
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(
        graph=frozen_func.graph,
        run_meta=run_meta,
        cmd='op',
        options=opts)
    return flops.total_float_ops


def get_num_model_params(model, bias=1):
    """Calculate the number of parameters of the convolution layers."""
    num_params_per_layers = []
    for layer in model.layers:
        if layer.name.find("conv") != -1:
            input_channels = int(layer.input.type_spec.shape.dims[3])
            kernel_size = np.prod(layer.kernel_size)
            output_channels = layer.output_shape[3]
            num_params = (input_channels * kernel_size + bias) * output_channels
            num_params_per_layers.append(num_params)
        else:
            continue
    return np.sum(num_params_per_layers)


def get_manual_structure(num_segments: int, num_features: int, name='cnnoise', data_format='channels_last'):
    """Define CNNoise model architecture."""
    layer_repetitions = 1  # 4
    n_filters_1 = 6  # 18
    n_filters_2 = 8  # 27
    n_filters_3 = 4  # 9, num_segments
    kernel_size_0 = [num_segments, 6]  # 18
    kernel_size_1 = [3, 3]  # 6 18
    kernel_size_2 = [3, 3]  # 4 9
    stride = (num_segments, 1)
    curr_conv_num = 1

    # TODO:
    if data_format == 'channels_first':
        input_shape = (1, num_segments, num_features)
        batch_norm_axis = 1
        raise Exception("For CPU model must be in NHWC format!")
    elif data_format == 'channels_last':
        input_shape = (num_segments, num_features, 1)
        batch_norm_axis = -1
    else:
        raise ValueError('Wrong data format: ', data_format)

    model = keras.models.Sequential(name=name)

    model.add(keras.layers.Conv2D(
        input_shape=input_shape,
        filters=n_filters_1,
        kernel_size=kernel_size_0,
        strides=stride,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=0.01),
        bias_regularizer=keras.regularizers.l2(l=0.0001),
        data_format=data_format,
        name=f'conv_{curr_conv_num}')
    )
    model.add(keras.layers.BatchNormalization(
        axis=batch_norm_axis,
        name=f'batchnorm_{curr_conv_num}')
    )
    model.add(keras.layers.Activation('relu'))
    curr_conv_num += 1

    for i in range(layer_repetitions):
        model.add(keras.layers.Conv2D(
            filters=n_filters_2,
            kernel_size=kernel_size_2,
            strides=stride,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l=0.01),
            bias_regularizer=keras.regularizers.l2(l=0.0001),
            data_format=data_format,
            name=f'conv_{curr_conv_num}')
        )
        model.add(keras.layers.BatchNormalization(
            axis=batch_norm_axis,
            name=f'batchnorm_{curr_conv_num}')
        )
        model.add(keras.layers.Activation('relu'))
        curr_conv_num += 1

        model.add(keras.layers.Conv2D(
            filters=n_filters_3,
            kernel_size=kernel_size_1,
            strides=stride,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l=0.01),
            bias_regularizer=keras.regularizers.l2(l=0.0001),
            data_format=data_format,
            name=f'conv_{curr_conv_num}')
        )
        model.add(keras.layers.BatchNormalization(
            axis=batch_norm_axis,
            name=f'batchnorm_{curr_conv_num}')
        )
        model.add(keras.layers.Activation('relu'))
        curr_conv_num += 1

        model.add(keras.layers.Conv2D(
            filters=n_filters_1,
            kernel_size=kernel_size_1,
            strides=stride,
            padding='same',
            kernel_regularizer=keras.regularizers.l2(l=0.01),
            bias_regularizer=keras.regularizers.l2(l=0.0001),
            data_format=data_format,
            name=f'conv_{curr_conv_num}')
        )
        model.add(keras.layers.BatchNormalization(
            axis=batch_norm_axis,
            name=f'batchnorm_{curr_conv_num}')
        )
        model.add(keras.layers.Activation('relu'))
        curr_conv_num += 1

    model.add(keras.layers.Conv2D(
        filters=n_filters_2,
        kernel_size=kernel_size_2,
        strides=stride,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=0.01),
        bias_regularizer=keras.regularizers.l2(l=0.0001),
        data_format=data_format,
        name=f'conv_{curr_conv_num}')
    )
    model.add(keras.layers.BatchNormalization(
        axis=batch_norm_axis,
        name=f'batchnorm_{curr_conv_num}')
    )
    model.add(keras.layers.Activation('relu'))
    curr_conv_num += 1

    model.add(keras.layers.Conv2D(
        filters=n_filters_3,
        kernel_size=kernel_size_1,
        strides=stride,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=0.01),
        bias_regularizer=keras.regularizers.l2(l=0.0001),
        data_format=data_format,
        name=f'conv_{curr_conv_num}')
    )
    model.add(keras.layers.BatchNormalization(
        axis=batch_norm_axis,
        name=f'batchnorm_{curr_conv_num}')
    )
    model.add(keras.layers.Activation('relu'))
    curr_conv_num += 1

    model.add(keras.layers.Conv2D(
        filters=1,
        kernel_size=[1, num_features],
        strides=stride,
        padding='same',
        kernel_regularizer=keras.regularizers.l2(l=0.01),
        bias_regularizer=keras.regularizers.l2(l=0.0001),
        data_format=data_format,
        name=f'conv_{curr_conv_num}')
    )

    #model.add(Dropout(0.2))  #
    # model.add(Activation('sigmoid'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # model.add(Dense(1, activation='linear'))
    return model


def build_model(num_features: int, num_segments: int, data_format: str, model_name: str):
    """Specifies the structure and complexity of the model"""
    # Configure model structure
    model = get_manual_structure(
        name=model_name,
        data_format=data_format,
        num_segments=int(num_segments),
        num_features=int(num_features))
    # Compute the number of parameters within convolution layers
    print(f'Number of convolution layers params: ', get_num_model_params(model))
    # Compute the number of floating point operations (FLOPS)
    flops = get_model_flops(model, batch_size=1)
    print(f'Number of floating point operations: ', flops)
    # print(f"FLOPS: {flops / 10 ** 9:.03} G")
    return model


def define_dirs(root_project, feat_name, data_name):
    root_data = os.path.join(root_project, 'features', feat_name, data_name)
    root_models = os.path.join(root_project, 'models-dev')
    dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_model = os.path.join(root_models, feat_name, data_name, dt)
    dirs = {
        'root_data': root_data,
        'dir_data_train': os.path.join(root_data, 'train'),
        'dir_data_valid': os.path.join(root_data, 'valid'),
        'dir_model': dir_model,
        'dir_model_best': os.path.join(dir_model, 'model'),
        'dir_model_checks': os.path.join(dir_model, 'checkpoints'),
        'dir_journal_logs': os.path.join(dir_model, 'logs', 'journal'),
        'dir_tb_logs': os.path.join(dir_model, 'logs', 'tensorboard'),
        'dir_audio_test': os.path.join(root_project, 'audio-test'),
        'dt': dt
    }
    return dirs


def check_param_feats(params):
    fs = params['fs']
    if type(fs) is int:
        assert fs == 16e3, "Wrong sample rate!"
    elif type(fs) is str:
        fs = int(float(fs))
        params['sample_rate'] = fs
        assert fs == 16e3, "Wrong sample rate!"
    assert type(params['windur']) is float
    assert type(params['hopfrac']) is float
    assert type(params['mingain']) is int
    assert type(params['nfft']) is int
    assert type(params['feattype']) is str
    return params


def check_param_train(params):
    # assert type(params['name_feats']) is str
    assert type(params['name_model']) is str
    assert type(params['optimizer']) is str
    assert type(params['loss']) is str
    if params['loss'].lower() not in ['mse', 'mae', 'custom_rmse', 'custom', 'other', 'experimental']:
        raise ValueError('Loss not recognized: ', params['loss'])
    assert type(params['minibatch_size']) is int
    assert type(params['num_epochs']) is int
    assert type(params['init_lr']) is float
    assert float(params['init_lr']) < 1.0e-01
    return params


def get_config_feats(fs=16000):
    cfg = get_params.features_cnnoise_new(fs)
    return cfg  # check_param_feats(cfg)


def get_config_train(features_name, dataset_name):
    cfg = get_params.training_cnnoise_new(features_name, dataset_name)
    return check_param_train(cfg)


def prepare_training_env(gpus):
    # Set random seeds
    os.environ['PYTHONHASHSEED'] = str(1)
    random.seed(1)
    # np.random.seed(1)
    tf.random.set_seed(1)

    # GPU settings
    # os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    # os.getenv('TF_GPU_ALLOCATOR')
    if gpus:
        gpu_name = tf.test.gpu_device_name()
        out = "\nTensorFlow IS using the GPU" + "\nGPU device found at: {}".format(gpu_name)
        # Memory management for GPU
        tf.config.experimental.set_memory_growth(gpus[0], True)
    else:
        out = "\nTensorFlow IS NOT using the GPU"
    return print(out)


def main(name_feats, name_datas):

    # Specify and check params
    param_train = get_config_train(name_feats, name_datas)
    param_feats = get_config_feats(fs=16000)

    # Managing project structure and directories
    dirs = define_dirs(
        root_project=os.path.join('/', *os.path.dirname(__file__).split('/')[:-2]),
        feat_name=param_train['name_feats'],
        data_name=param_train['name_datas'],
    )
    print('\nFeature name:', param_train['name_feats'])
    print('\nDataset name:', param_train['name_datas'])
    print('Model name:', dirs['dt'])

    # Number of batches to train on
    param_train['num_batches'] = int(len(os.listdir(dirs['dir_data_train'])))

    # Define model
    print('\nDefine model structure... ')
    model = build_model(
        num_features=param_feats['nfeats'],
        num_segments=param_feats['nframes'],
        data_format=param_train['data_format'],
        model_name=param_train['name_model'])

    # Prepare test audio
    model.test_xspectra = {}
    model.test_features = {}
    model.test_segments = {}
    model.test_audio_list = {}
    print('\nPrepare test audio features...')
    for ntest, pathtest in enumerate(Path(dirs['dir_audio_test']).resolve().glob('*.wav')):
        print(pathtest)
        # testFs, testSig = wavfile.read(str(pathtest))
        testSig, testFs = sf.read(str(pathtest))
        testSig, _ = get_features.srCheck(testSig, testFs, param_feats['fs'])
        testSpecs = get_features.sig2spec(testSig, param_feats)
        testFeats = get_features.spec2feats(testSpecs, param_feats)
        testSegs, _ = get_features.shapeSegs(
            X=testFeats,
            y=None,
            num_fr_seg=param_feats['nframes'])
        # shape: [batch x time x freq x channel]
        #testSegs = np.expand_dims(np.expand_dims(np.transpose(testSegs), axis=0), axis=-1)

        # Attach testing data and control params to the model for later use
        model.test_xspectra[ntest] = testSpecs
        model.test_features[ntest] = testFeats
        model.test_segments[ntest] = testSegs
        model.test_audio_list[ntest] = pathtest

    model.dirs = dirs
    model.params = param_train
    model.params_feat = param_feats
    model.dt = dirs['dt']

    # Start training process
    model = run_training(model)
    # Save results and params to the best model directory
    model.save(dirs['dir_model_best'], 'model.model')


if __name__ == '__main__':
    prepare_training_env(tf.config.list_physical_devices('GPU'))
    main(
        name_feats='32fp_16000hz_128nfft_64nhop_MagSpec_30F_65B',
        name_datas='16kHzEN8hRU5hN2h'
        # name_datas='dnsc_ru'
        #name_datas='tmp'
    )
    print('\nTraining process completed!')
