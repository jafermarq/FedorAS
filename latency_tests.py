from time import time
import argparse

import yaml
import torch
from torchvision.models import resnet18, mobilenet_v2
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
parser.add_argument("--cpu_threads", type=int, default=0, help="Sets number of cpu threads for PyTorch (default 0, i.e. all will be used)")
parser.add_argument("--verbose", action='store_true', help="Make it verbose")
parser.add_argument("--compile", action='store_true', help="Compiles model (use if Pytorch >= 2.0)")
parser.add_argument("--common", action='store_true', help="Benchmark a few popular architectures architectures")


mb2_cfg = [(1,  16, 1, 1),
        (6,  24, 2, 1),
        (6,  32, 3, 2),
        (6,  64, 4, 2),
        (6,  96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1)]

def get_dummy_dataloader(num_images: int, batch: int, input_shape: tuple):
    """Returns a very rudimentary (but sufficient
    for our profiling purposes) dataset"""
    dataset = [(torch.randn(input_shape), i) for i, _ in enumerate(range(num_images))]
    # Then prepare a dataloader 
    return torch.utils.data.DataLoader(dataset, batch_size=batch, num_workers=2)


def benchmark_common_architectures(verbose: bool):

    args = parser.parse_args()
    cfg = awesomeyaml.Config.build_from_cmdline(*args.yamls)

    # prepare models
    models = []
    model = resnet18(num_classes=10)
    model.conv1 = torch.nn.Conv2d(3,64, kernel_size=3, stride=1, padding=1)
    model.maxpool = torch.nn.Identity()
    models.append(model)

    model = mobilenet_v2(inverted_residual_setting=mb2_cfg, num_classes=10)
    models.append(model)

    for m in models:
        input_shape = cfg.model.type.keywords['input_size'][1:]

        if verbose:
            print(f"Detected input shape for {cfg.dataset.name} dataset: {input_shape}")
        flops, params = get_model_complexity_info(m, tuple(input_shape), as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        if verbose:
            print(f"FLOPS: {flops/1e6:.2f} MFLOPs // params: {params/1e6:.3f} Million")

        dataloader = get_dummy_dataloader(args.iter+args.warmup, args.batch, input_shape)

        t_tensor = _bench(m, dataloader, verbose)

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

    t_tensor = _bench(mmodel, dataloader, verbose)

    return flops, params, t_tensor


def _bench(model, dataloader, verbose: bool):

    # figure out device
    cuda = torch.cuda.is_available()
    if cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # Measure latency
    if verbose:
        print("Measuring latency using:")
        print(f"\t Bach: {args.batch}")
        print(f"\t Iterations: {args.iter}")
        print(f"\t Warmup: {args.warmup}")
        print(f"\t Device: {device}")

    time_per_batch = []
    model.eval()
    model.to(device)

    if args.compile:
        model = torch.compile(model)
    
    with torch.no_grad():
        for idx, (data, lbl) in enumerate(dataloader):
            data = data.to(device)
            if cuda:
                starter.record()
            else:
                t_start = time()
            _ = model(data)
            if cuda:
                ender.record()
                torch.cuda.synchronize()
                elapsed = starter.elapsed_time(ender) # in milliseconds
            else:
                t_end = time()
                elapsed = 1000 * (t_end - t_start) # to milliseconds

            if idx >= args.warmup:
                time_per_batch.append(elapsed)


    t_tensor = torch.tensor(time_per_batch)
    if verbose:
        print(f"Result: {t_tensor.mean():.1f}±{t_tensor.std():.1f} ms")

    return t_tensor


if __name__ == "__main__":

    args = parser.parse_args()

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    if args.decision:
        benchmark_single_decision(args.decision, verbose=True)
    else:
        if args.common:
            benchmark_common_architectures(args.verbose)
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
                print(f"\t Latency (milliseconds): {torch.mean(torch.cat(tier_latency)):.1f}±{torch.std(torch.cat(tier_latency)):.1f}")

