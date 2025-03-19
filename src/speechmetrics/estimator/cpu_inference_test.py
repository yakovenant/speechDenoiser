import os, argparse, time
import torch


os.system(f"taskset -pc 0 {os.getpid()}")
device = torch.device('cpu')
map_loc_dict = {f'cuda:{n}': 'cpu' for n in range(16)}


def main(args):

    sr, dur_sec = 16000, 10
    num_iters = 10

    state_dict = torch.load(model_path, map_location=map_loc_dict, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    print('Processing...')
    test_input = torch.rand(1, sr*dur_sec).to(device)
    times = []
    with torch.no_grad():
        for i in range(num_iters):
            time_start = time.time()
            test_res = model(test_input)
            time_fin = time.time()
            cur_infer_time = time_fin - time_start
            times.append(cur_infer_time)
    times_avg = sum(times) / num_iters
    print(f'Average inference time using single core CPU and {num_iters} iterations is {times_avg:.4f} seconds.')

if __name__ == "__main__":

    main(args)
    print("\nDONE!\n")
