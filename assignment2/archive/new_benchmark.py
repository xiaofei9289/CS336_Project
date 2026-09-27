import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
import timeit
import argparse

# model configuration
MODEL_CONFIGS = {
    "small":  dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large":  dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl":     dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10B":    dict(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", 
    choices = ["forward", "forward_backward", "train"],
    default = "forward",
    help = (
        "forward: forward pass only;" 
        "forward_backward: forward pass + loss + backward pass;"
        "train: forward pass + loss + backward pass + parameter update"
        ),
    )
# model configuration
parser.add_argument(
    "--model-size",
    choices=list(MODEL_CONFIGS),
    default="small",
)
# measurement steps
parser.add_argument(
    "--measurement-steps",
    type=int,
    default=10,
    help="number of timed steps",
)

parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--context-length", type=int, default=512)
parser.add_argument("--rope-theta", type=float, default=10000.0)

# if None is specified, the configuration will be inherited from the model size
parser.add_argument("--d-model", type=int, default=None)
parser.add_argument("--num-layers", type=int, default=None)
parser.add_argument("--num-heads", type=int, default=None)
parser.add_argument("--d-ff", type=int, default=None)

# input configuration
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--sequence-length", type=int, default=512)

parser.add_argument(
    "--warmup-steps",
    type=int,
    default=5,
    help="number of warmup steps; may be 0",
)

args = parser.parse_args()

if args.warmup_steps < 0:
    parser.error("--warmup-steps must be greater than or equal to 0")
if args.measurement_steps <= 0:
    parser.error("--measurement-steps must be greater than 0")

# check the configuration
config = MODEL_CONFIGS[args.model_size].copy()
overrides = {}

for name in config:
    value = getattr(args, name)

    if value is not None:
        if value != config[name]:
            overrides[name] = value
        config[name] = value

    # write the final value back to args
    setattr(args, name, config[name])

if overrides:
    config_name = f"custom (based on {args.model_size})"
else:
    config_name = args.model_size


mode = args.mode
vocab_size = args.vocab_size
context_length = args.context_length
batch_size = args.batch_size
sequence_length = args.sequence_length

if mode == "forward":
    measurement_name = "forward time"
elif mode == "forward_backward":
    measurement_name = "forward pass + loss + backward pass time"
elif mode == "train":
    measurement_name = "forward pass + loss + backward pass + parameter update time (excluding zero_grad)"
else:
    raise ValueError(f"Invalid mode: {mode}")

# 4. check the final configuration
sizes = [
    vocab_size,
    context_length,
    batch_size,
    sequence_length,
    args.d_model,
    args.d_ff,
    args.num_layers,
    args.num_heads,
]

if any(size <= 0 for size in sizes):
    parser.error("model and input sizes must be greater than 0")

if args.d_model % args.num_heads != 0:
    parser.error("--d-model must be divisible by --num-heads")

if (args.d_model // args.num_heads) % 2 != 0:
    parser.error("when using the current RoPE implementation, the dimension of each attention head must be even")

if sequence_length > context_length:
    parser.error("--sequence-length must be less than or equal to --context-length")

if not (0 < args.rope_theta < float("inf")):
    parser.error("--rope-theta must be a finite positive number")



# 1. check if cuda is available
print("if cuda is available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

device = "cuda:0"
print("GPU device: ", torch.cuda.get_device_name(device))
print("measurement mode: ", mode)
print("model size: ", config_name)
print("actual model structure: ", config)

if overrides:
    print("overrides: ", overrides)

print("vocab size: ", vocab_size)
print("maximum context length: ", context_length)
print("RoPE theta: ", args.rope_theta)
print("batch size: ", batch_size)
print("actual sequence length: ", sequence_length)

# 2. create model
model = BasicsTransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
)
model = model.to(device)
model.train()

# optimizer
optimizer = None

if mode == "train":
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.0,
    )
    print("optimizer:", type(optimizer).__name__)
    print("learning rate:", optimizer.param_groups[0]["lr"])
    print("weight decay:", optimizer.param_groups[0]["weight_decay"])

# keep gradient tracking enabled for all three modes
assert torch.is_grad_enabled()

# 3. create random input and target

x = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

# create a random target tensor
targets = torch.randint(
    low=0, 
    high=vocab_size, 
    size=(batch_size, sequence_length),
    dtype=torch.long,
    device=device,
)

print("target shape:", targets.shape)
print("target device:", targets.device)
print("target dtype:", targets.dtype)
print("target maximum index:", targets.max().item())
print("target minimum index:", targets.min().item())

assert targets.shape == (batch_size, sequence_length)
assert targets.device == x.device
assert targets.dtype == torch.long
assert ((targets >= 0) & (targets < vocab_size)).all().item()

print("model parameters device:", next(model.parameters()).device)
print("model parameters dtype:", next(model.parameters()).dtype)
print("input device:", x.device)
print("input shape:", x.shape)
print("model is training:", model.training)
print("gradient tracking enabled:", torch.is_grad_enabled())

# ==============================

# 5. set the number of warmup runs
def run_steps():
    logits = model(x)
    loss = None
    if mode in ("forward_backward", "train"):
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        loss.backward()

        if mode == "train":
            optimizer.step()
    return logits, loss


warmup_runs = args.warmup_steps
measurement_runs = args.measurement_steps
print("measurement steps:", measurement_runs)
print("warmup steps:", warmup_runs)

if warmup_runs == 0:
    print("note: no correctness check for warmup steps")

    if mode == "train":
        print("note: the first measurement round will include the initialization of the optimizer state")

# 6. run the warmup runs
print("\n warmup starts...")
for i in range(warmup_runs):
    # clear the gradient of the model
    model.zero_grad(set_to_none=True)
    # only save the old parameters in the first warmup run for train mode
    if mode == "train" and i == 0:
        param_before = model.lm_head.weight.detach().clone()
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)


    # start timing
    start_time = timeit.default_timer()
    # operate the forward pass
    logits, loss = run_steps()
    #=======================


    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    # stop timing
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    print(
    f"warmup run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )

    # check the first time warmup output, and put it out of the time calculation
    if i==0:
        print("output device:", logits.device)
        print("output shape:", logits.shape)
        assert logits.shape == (batch_size, sequence_length, vocab_size)
        assert logits.device == x.device 
        assert logits.requires_grad


        if mode in ("forward_backward", "train"):
            assert loss is not None
            assert loss.shape == torch.Size([])
            assert loss.requires_grad
            assert torch.isfinite(loss).item()
            assert loss.device == targets.device

            param = model.lm_head.weight
            assert param.grad is not None
            assert param.grad.shape == param.shape
            assert torch.isfinite(param.grad).all().item()
            del param 

        # check if the parameters have changed in train mode
        if mode == "train":
            param_after = model.lm_head.weight.detach()

            assert torch.isfinite(param_after).all().item()

            changed = (param_after != param_before).any().item()
            max_change = (param_after - param_before).abs().max().item()

            assert changed, "the selected parameters have not changed"

            print("parameter changed:", changed)
            print(f"maximum absolute change: {max_change:.8e}")

            del param_before, param_after
        print("the present mode check passed")
    # release the output of the warmup run
    del logits, loss

# 7. start the formal measurements
print("\n formal measurement starts...")
time_ms = []
for i in range(measurement_runs):

    # clear the gradient of last round
    model.zero_grad(set_to_none=True)
    # waiting for GPU to be ready
    torch.cuda.synchronize(device)

    #=======================
    # start timing
    start_time = timeit.default_timer()

    logits, loss = run_steps()

    # waiting for GPU to finish
    torch.cuda.synchronize(device)
    end_time = timeit.default_timer()
    # calculate the time taken
    time_taken = (end_time - start_time) * 1000 # in milliseconds
    time_ms.append(time_taken)
    print(
    f"formal measurement run {i + 1}: "
    f"{measurement_name}: {time_taken:.6f} milliseconds"
    )
    # release the output of the measurement run
    del logits, loss

# clear the gradient of the last round
model.zero_grad(set_to_none=True)

# 8. statistics the formal measurement results
print("\n statistics starts...")
# calculate the average time taken
average_time = sum(time_ms) / len(time_ms)
variance = sum((t - average_time) ** 2 for t in time_ms) / len(time_ms)
std_ms = variance ** 0.5
print(f"measurement name: {measurement_name}")
print(f"average time of {measurement_name}: {average_time:.6f} milliseconds")
print(f"standard deviation of {measurement_name}: {std_ms:.6f} milliseconds")
print(f"measurement runs: {len(time_ms)}")
print(f"min time of {measurement_name}: {min(time_ms):.6f} milliseconds")
print(f"max time of {measurement_name}: {max(time_ms):.6f} milliseconds")



