import os, shutil


def get_dataset(dir_root, dir_output, prefix):

    for cur_input in os.listdir(dir_root):

        if os.path.isdir(cur_input):
            cur_dir = os.path.join(dir_root, cur_input)
            get_dataset(cur_dir, dir_output, prefix)

        if not cur_input.split('.')[0].endswith(prefix):
            continue

        if prefix == '' and not cur_input.split('.')[0].endswith('norm'):
            continue

        path_to_input = os.path.join(dir_root, cur_input)

        sub = dir_root.split('/')[-2]
        if prefix.endswith('snr'):
            id = dir_root.split('/')[-1][:-3].split('_')[0]
        else:
            id = dir_root.split('/')[-1].split('_')[0]
        cur_output = sub + '_' + id + '_' + cur_input
        path_to_output = os.path.join(dir_output, cur_output)

        os.makedirs(os.path.dirname(path_to_output), exist_ok=True)
        shutil.copy(path_to_input, path_to_output)


if __name__ == "__main__":

    path_to_root = '/media/ssd/TIMIT/TEST/'
    sub_dir = 'DR8'
    snr_level = ''  # '0dBsnr'
    path_to_out_root = '/media/ssd/TIMIT/DENOISER/'

    ids = []
    for sub_dir_id1 in os.listdir(os.path.join(path_to_root, sub_dir)):
        if sub_dir_id1.endswith('0'):
            ids.append(sub_dir_id1)

    for sub_dir_id2 in ids:
        print('\nGenerate denoiser data...')
        if snr_level == '':
            dir_root = f'{path_to_root}{sub_dir}/{sub_dir_id2}'
            dir_output = f'{path_to_out_root}target'
        else:
            dir_root = f'{path_to_root}{sub_dir}/{sub_dir_id2}_{snr_level[:-3]}'
            dir_output = f'{path_to_out_root}SNR_{snr_level}'
        print(f'Current dir: {dir_root}')
        print(f'Output dir: {dir_output}')
        get_dataset(
            dir_root=dir_root,
            dir_output=dir_output,
            prefix=snr_level
        )
    print("\nDONE!\n")
