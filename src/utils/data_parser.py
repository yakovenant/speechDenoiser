import os
import shutil


def read_tsv(filepath):
    with open(filepath) as file:
        data_dict = {
            'files': [],
            'labels': []
        }
        for line in file:
            data = line.split('\t')
            if data[0] == "filename":
                continue
            else:
                data_dict['files'].append(data[0])
                data_dict['labels'].append(data[-1].split('\n')[0])
    return data_dict


def main(dataroot: str, datasave: str, filepath: str):

    if not os.path.isdir(datasave):
        os.makedirs(datasave)
    data_dict = read_tsv(filepath)

    for i, file in enumerate(data_dict['files']):
        label = os.path.join(datasave, data_dict['labels'][i])
        if not os.path.isdir(label):
            os.makedirs(label)
        audio_path_from = os.path.join(dataroot, file)
        audio_path_to = os.path.join(label, file)
        if os.path.isfile(audio_path_from):
            if os.path.isfile(audio_path_to):
                continue
            else:
                shutil.copy(audio_path_from, audio_path_to)
                print("Saved audio: " + audio_path_to)
    print("done.")


if __name__ == '__main__':
    print("\nRun data parser...")
    main(
        dataroot=f"/home/iakovenant/datasets/audio/DESED_public_eval/audio/eval/public/",
        datasave=f"/home/iakovenant/datasets/audio/DESED_public_eval/audio/sorted/",
        filepath=f"/home/iakovenant/datasets/audio/DESED_public_eval/metadata/eval/public.tsv"
    )
