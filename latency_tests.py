from time import time
import argparse

import yaml
import torch
import awesomeyaml
from ptflops import get_model_complexity_info

from src.models.utils import instantiate_supernet

parser = argparse.ArgumentParser(description="FedorAS")
parser.add_argument('yamls', nargs='+', type=str)
parser.add_argument("--decision", type=str, default="", help="path to extract from supernet. This is a comma-separated string of integers (use this to evaluate a single decision)")
parser.add_argument("--decision_list", type=str, default="decisions_list.yaml", help="Path to a .yaml containing list of decisions to evaluate (default: decision_list.yaml)")
parser.add_argument("--batch", type=int, default=1, help="batch size (default 1)")
parser.add_argument("--iter", type=int, default=100, help="Number of iterations/batches to use (default 100)")
parser.add_argument("--warmup", type=int, default=5, help="Number of iterations/batches for warmup (default 5)")
parser.add_argument("--verbose", action='store_true', help="Make it verbose")



def get_dummy_dataloader(num_images: int, batch: int, input_shape: tuple):
    """Returns a very rudimentary (but sufficient
    for our profiling purposes) dataset"""
    dataset = [(torch.randn(input_shape), i) for i, _ in enumerate(range(num_images))]
    # Then prepare a dataloader 
    return torch.utils.data.DataLoader(dataset, batch_size=batch)

def benchmark_single_decision(decision, verbose: bool):

    if verbose:
        print("%" * 50)
        print(f"Decision: {decision}")
    # parse config
    args = parser.parse_args()
    cfg = awesomeyaml.Config.build_from_cmdline(*args.yamls)

    # parsing decision
    decision_as_list = [int(d) for d in decision.split(',')]

    # measure flops/params in model by passing a single input
    supernet = instantiate_supernet(cfg.model)
    mmodel = supernet.realise(decision_as_list)
    input_shape = cfg.model.type.keywords['input_size'][1:]
    if verbose:
        print(f"Detected input shape for {cfg.dataset.name} dataset: {input_shape}")
    flops, params = get_model_complexity_info(mmodel, tuple(input_shape), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
    if verbose:
        print(f"FLOPS: {flops/1e6:.2f} MFLOPs // params: {params/1e6:.3f} Million")

    # prepare input tensor
    dataloader = get_dummy_dataloader(args.iter+args.warmup, args.batch, input_shape)

    # figure out device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Measure latency
    if verbose:
        print("Measuring latency using:")
        print(f"\t Bach: {args.batch}")
        print(f"\t Iterations: {args.iter}")
        print(f"\t Warmup: {args.warmup}")
        print(f"\t Device: {device}")

    time_per_batch = []
    mmodel.eval()
    mmodel.to(device)
    
    with torch.no_grad():
        for idx, (data, lbl) in enumerate(dataloader):
            data = data.to(device)
            t_start = time()
            _ = mmodel(data)
            t_end = time()

            if idx >= args.warmup:
                time_per_batch.append(t_end - t_start)


    t_tensor = torch.tensor(time_per_batch)
    if verbose:
        print(f"Result: {1000*t_tensor.mean():.1f}±{1000*t_tensor.std():.1f} ms")

    return flops, params, t_tensor


if __name__ == "__main__":

    args = parser.parse_args()

    if args.decision:
        benchmark_single_decision(args.decision, verbose=True)
    else:
        with open(args.decision_list, 'r') as file:
            decisions_data = yaml.safe_load(file)
        
        # For each tier
        for tier, decisions in decisions_data['tiers'].items():
            tier_params = []
            tier_flops = []
            tier_latency = []
            for d in decisions: # for each decision in the tier
                flops, params, t_tensor= benchmark_single_decision(d, verbose=args.verbose)
                tier_params.append(params)
                tier_flops.append(flops)
                tier_latency.append(t_tensor)
            
            # report data per tier
            print(f"Tier: {tier}")
            print(f"\t Latency (milliseconds): {1000*torch.mean(torch.cat(tier_latency)):.1f}±{1000*torch.std(torch.cat(tier_latency)):.1f}")

