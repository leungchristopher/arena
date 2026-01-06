# %%
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "ARENA_3.0"
branch = "main"

# Install dependencies
try:
    import transformer_lens
except:
    %pip install transformer_lens==2.11.0 einops jaxtyping git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/ARENA_3.0-{branch}


if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
# %%
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import datasets
import einops
import numpy as np
import torch as t
import torch.nn as nn
import wandb
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

device = t.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part1_transformer_from_scratch"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part1_transformer_from_scratch.solutions as solutions
import part1_transformer_from_scratch.tests as tests
from plotly_utils import imshow

MAIN = __name__ == "__main__"
# %%
reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,  # you'll learn about these arguments later!
)

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

print(sorted_vocab[:20])
print()
print(sorted_vocab[250:270])
print()
print(sorted_vocab[990:1010])
print()
# %%
# Tokenisation is weird with numbers
print(reference_gpt2.to_str_tokens("56873+3184623=123456789-1000000000"))
# %%
# Tokenise text
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)
print(tokens)
print(tokens.shape)
print(reference_gpt2.to_str_tokens(tokens))
# %%
# Map tokens to logits
logits, cache = reference_gpt2.run_with_cache(tokens)
print(logits.shape)
# %%
# Softmax distribution
probs = logits.softmax(dim=-1)
print(probs.shape)
# %%
most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])

print(list(zip(reference_gpt2.to_str_tokens(tokens), most_likely_next_tokens)))
# %%
# Map distribution to token
next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)
print(repr(next_char))
# %%
print(f"Sequence so far: {reference_gpt2.to_string(tokens)[0]!r}")

for i in range(10):
    print(f"{tokens.shape[-1] + 1}th char = {next_char!r}")
    # Define new input sequence, by appending the previously generated token
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    # Pass our new sequence through the model, to get new output
    logits = reference_gpt2(tokens)
    # Get the predicted token at the end of our sequence
    next_token = logits[0, -1].argmax(dim=-1)
    # Decode and print the result
    next_char = reference_gpt2.to_string(next_token)
# %%
# Clean transformer implementation
for activation_name, activation in cache.items():
    # Only print for first layer
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")

# Print all parameters
for name, param in reference_gpt2.named_parameters():
    # Only print for first layer
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")
# %%
# As a reference - note there's a lot of stuff we don't care about in here, to do with library internals or other architectures
print(reference_gpt2.cfg)
# %%
@dataclass
class Config:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12


cfg = Config()
print(cfg)
# %%
def rand_float_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randn(shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def rand_int_test(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    random_input = t.randint(100, 1000, shape).to(device)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape, "\n")


def load_gpt2_test(cls, gpt2_layer, input):
    cfg = Config(debug=True)
    layer = cls(cfg).to(device)
    layer.load_state_dict(gpt2_layer.state_dict(), strict=False)
    print("Input shape:", input.shape)
    orig_input = input.clone()
    output = layer(orig_input)
    assert t.allclose(input, orig_input), "Input has been modified, make sure operations are not done in place"
    if isinstance(output, tuple):
        output = output[0]
    print("Output shape:", output.shape)
    try:
        reference_output = gpt2_layer(input)
    except:
        reference_output = gpt2_layer(input, input, input)
    print("Reference output shape:", reference_output.shape, "\n")
    comparison = t.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    print(f"{comparison.sum() / comparison.numel():.2%} of the values are correct\n")
    assert 1 - (comparison.sum() / comparison.numel()) < 1e-5, (
        "More than 0.01% of the values are incorrect"
    )
# %%
class LayerNorm(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.w = nn.Parameter(t.ones(cfg.d_model))
        self.b = nn.Parameter(t.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_var = residual.var(dim=-1, keepdim=True, unbiased=False) + self.cfg.layer_norm_eps
        residual = (residual - residual_mean) / residual_var.sqrt()
        return residual * self.w + self.b


rand_float_test(LayerNorm, [2, 4, 768])
load_gpt2_test(LayerNorm, reference_gpt2.ln_final, cache["resid_post", 11])
tests.test_layer_norm_epsilon(LayerNorm, cache["resid_post", 11])
# %%
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]
        


rand_int_test(Embed, [2, 4])
load_gpt2_test(Embed, reference_gpt2.embed, tokens)

# %%
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(self.W_pos[:seq_len], "position d_model -> batch position d_model", batch=batch)


rand_int_test(PosEmbed, [2, 4])
load_gpt2_test(PosEmbed, reference_gpt2.pos_embed, tokens)
# %%
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def apply_causal_mask(
        self,
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"],
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        causal_mask = t.triu(attn_scores, diagonal=1).bool()
        masked_attn_scores = t.masked_fill(attn_scores, causal_mask, self.IGNORE)
        return masked_attn_scores


tests.test_causal_mask(Attention.apply_causal_mask)
# %%
import circuitsvis as cv
from IPython.display import display

display(
    cv.attention.attention_patterns(
        tokens=reference_gpt2.to_str_tokens(reference_text), attention=cache["pattern", 0][0]
    )
)
# %%
display(
    cv.attention.attention_heads(
        tokens=reference_gpt2.to_str_tokens(reference_text), attention=cache["pattern", 0][0]
    )
)
# %%
# Attention
class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=device))

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        q = einops.einsum(normalized_resid_pre, self.W_Q, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_Q
        k = einops.einsum(normalized_resid_pre, self.W_K, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_K
        v = einops.einsum(normalized_resid_pre, self.W_V, "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") + self.b_V

        attn_scores = einops.einsum(q, k, "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k")
        attn_scores_masked = self.apply_causal_mask(attn_scores / (self.cfg.d_head ** 0.5))
        attn_pattern = attn_scores_masked.softmax(-1)

        z = einops.einsum(v, attn_pattern, "batch posn_k n_heads d_head, batch n_heads posn_q posn_k -> batch posn_q n_heads d_head")

        attn_out = einops.einsum(z, self.W_O, "batch posn_q n_heads d_head, n_heads d_head d_model -> batch posn_q d_model") + self.b_O
        
        return attn_out
        


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        # You should copy your solution from earlier
        causal_mask = t.triu(attn_scores, diagonal=1).bool()
        masked_attn_scores = t.masked_fill(attn_scores, causal_mask, self.IGNORE)
        return masked_attn_scores


tests.test_causal_mask(Attention.apply_causal_mask)
rand_float_test(Attention, [2, 4, 768])
load_gpt2_test(Attention, reference_gpt2.blocks[0].attn, cache["normalized", 0, "ln1"])
# %%
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        layer1 = einops.einsum(normalized_resid_mid, self.W_in, "batch posn d_model, d_model d_mlp -> batch posn d_mlp") + self.b_in
        layer2 = gelu_new(layer1)
        layer3 = einops.einsum(layer2, self.W_out, "batch posn d_mlp, d_mlp d_model -> batch posn d_model") + self.b_out

        return layer3



rand_float_test(MLP, [2, 4, 768])
load_gpt2_test(MLP, reference_gpt2.blocks[0].mlp, cache["normalized", 0, "ln2"])

# %%
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        step1 = self.attn(self.ln1(resid_pre)) + resid_pre
        step2 = self.mlp(self.ln2(step1)) + step1
        return step2

rand_float_test(TransformerBlock, [2, 4, 768])
load_gpt2_test(TransformerBlock, reference_gpt2.blocks[0], cache["resid_pre", 0])

# %%
class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        unembed = einops.einsum(normalized_resid_final, self.W_U, "batch position d_model, d_model d_vocab -> batch position d_vocab") + self.b_U

        return unembed



rand_float_test(Unembed, [2, 4, 768])
load_gpt2_test(Unembed, reference_gpt2.unembed, cache["ln_final.hook_normalized"])

# %%
class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        demo1 = self.embed(tokens) + self.pos_embed(tokens)
        demo2 = demo1
        for block in self.blocks:
            demo2 = block(demo2)
        demo3 = self.ln_final(demo2)
        demo4 = self.unembed(demo3)
        return demo4



rand_int_test(DemoTransformer, [2, 4])
load_gpt2_test(DemoTransformer, reference_gpt2, tokens)
# %%
demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)

# %%
def get_log_probs(
    logits: Float[Tensor, "batch posn d_vocab"], tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:
    log_probs = logits.log_softmax(dim=-1)
    # Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
    log_probs_for_tokens = (
        log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    )

    return log_probs_for_tokens


pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# %%
test_string = """Mitigating the risk of extinction from AI should be a global priority alongside other societal-scale risks such as"""
for i in tqdm(range(100)):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

print(test_string)

# %% 
model_cfg = Config(
    debug=False,
    d_model=32,
    n_heads=16,
    d_head=2,
    d_mlp=32 * 4,
    n_layers=4,
    n_ctx=128,
    d_vocab=reference_gpt2.cfg.d_vocab,
)
model = DemoTransformer(model_cfg)
# %%
@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    wandb_project: str | None = "day1-demotransformer"
    wandb_name: str | None = None


args = TransformerTrainingArgs()

# %%
dataset = datasets.load_dataset("roneneldan/TinyStories", split="train")
print(dataset)
print(dataset[0]["text"])

# %% 
tokenized_dataset = tokenize_and_concatenate(
    dataset,
    reference_gpt2.tokenizer,
    streaming=False,
    max_length=model.cfg.n_ctx,
    column_name="text",
    add_bos_token=True,
    num_proc=4,
)

dataset_dict = tokenized_dataset.train_test_split(test_size=1000)
train_loader = DataLoader(
    dataset_dict["train"], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_loader = DataLoader(
    dataset_dict["test"], batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# %%
first_batch = train_loader.dataset[: args.batch_size]

print(first_batch.keys())
print(first_batch["tokens"].shape)

# %%
class TransformerTrainer:
    def __init__(self, args: TransformerTrainingArgs, model: DemoTransformer):
        super().__init__()
        self.model = model
        self.args = args
        self.sampler = solutions.TransformerSampler(self.model, reference_gpt2.tokenizer)
        self.optimizer = t.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0

        self.train_loader = DataLoader(
            dataset_dict["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            dataset_dict["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def training_step(self, batch: dict[str, Int[Tensor, "batch seq"]]) -> Float[Tensor, ""]:
        """
        Calculates the loss on the tokens in the batch, performs a gradient update step, and logs the loss.

        Remember that `batch` is a dictionary with the single key 'tokens'.
        """
        tokens = batch["tokens"].to(device)
        logits = self.model(tokens)
        loss = -get_log_probs(logits, tokens).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.step += 1
        wandb.log({"train loss": loss}, step=self.step)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluate the model on the test set and return the accuracy.
        """
        self.model.eval()
        correct, total = 0, 0
        for batch in self.test_loader:
            tokens = batch["tokens"].to(device)
            logits = self.model(tokens)[:, :-1]
            prediction = logits.argmax(dim=-1)
            correct += (prediction == tokens[:, 1:]).sum().item()
            total = tokens.size(0) * (tokens.size(1) - 1)
        
        accuracy = correct / total
        wandb.log({"accuracy": accuracy}, step=self.step)
        self.model.train()
        return accuracy

    def train(self):
        """
        Trains the model, for `self.args.epochs` epochs. Also handles wandb initialisation, and early stopping
        for each epoch at `self.args.max_steps_per_epoch` steps.
        """
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args)
        accuracy = np.nan

        progress_bar = tqdm(total=self.args.max_steps_per_epoch * self.args.epochs)

        for epoch in range(self.args.epochs):
            for i, batch in enumerate(self.train_loader):
                loss = self.training_step(batch)
                progress_bar.update()
                progress_bar.set_description(
                    f"Epoch {epoch + 1}, loss: {loss:.3f}, accuracy: {accuracy:.3f}"
                )
                if i >= self.args.max_steps_per_epoch:
                    break

            accuracy = self.evaluate()
            sample_text = self.sampler.sample("Once upon a time", max_tokens_generated=50)
            print(sample_text)

        wandb.finish()

# %%
## Sampling



# See the full run here: https://api.wandb.ai/links/dquarel/nrxuwnv7
model = DemoTransformer(model_cfg).to(device)
args = TransformerTrainingArgs()
trainer = TransformerTrainer(args, model)
trainer.train()
# %%
def sampling_fn(model: DemoTransformer, prompt: str) -> str:
    sampler = solutions.TransformerSampler(model, reference_gpt2.tokenizer)
    output = sampler.sample(prompt, temperature=0.7, top_p=0.95, max_tokens_generated=16)
    return output


model = DemoTransformer(model_cfg).to(device)
# %%
# Should be entirely random, because it uses a newly initialized model
print(sampling_fn(model, prompt="John and Mary went to the"))
class TransformerSampler:
    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs) -> str:
        """
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an end-of-sequence token. kwargs are
        passed to sample_next_token, to give detailed instructions on how new tokens are chosen.
        """
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]

        for i in range(max_tokens_generated):
            logits = self.model(input_ids[None, -self.cfg.n_ctx:])
            logits = logits[0,-1]
            sampled_token = t.tensor([TransformerSampler.sample_next_token(input_ids, logits, **kwargs)], device=device)
            input_ids = t.cat([input_ids, sampled_token], dim=-1)
            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
            if sampled_token == getattr(self.tokenizer, "eos_token_id", None):
                break
        
        return self.tokenizer.decode(input_ids)

    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None,
    ) -> int:
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(
                input_ids, logits, frequency_penalty
            )
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)

    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Returns the most likely token (as an int).
        """
        return logits.argmax().item()

    @staticmethod
    def apply_temperature(
        logits: Float[Tensor, "d_vocab"], temperature: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies temperature scaling to the logits.
        """
        return logits / temperature

    @staticmethod
    def apply_frequency_penalty(
        input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float
    ) -> Float[Tensor, "d_vocab"]:
        """
        Applies a frequency penalty to the logits.
        """
        d_vocab = logits.size(0)
        occurrences = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * occurrences


    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        """
        Samples from the distribution defined by the logits.
        """
        sample = t.distributions.categorical.Categorical(logits=logits).sample()
        return sample.item()

    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        """
        Samples from the top k most likely tokens.
        """
        topk_logits, topk_logits_ids = logits.topk(k)
        sampled_token_idx = t.distributions.categorical.Categorical(logits=topk_logits).sample()
        return topk_logits_ids[sampled_token_idx].item()

    @staticmethod
    def sample_top_p(
        logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1
    ) -> int:
        """
        Samples from the most likely tokens which make up at least p cumulative probability.
        """
        # obtain probabilities
        logits_sorted, ids = logits.sort(descending=True, stable=True)
        probs = logits_sorted.softmax(-1)
        cumulative_probs = probs.cumsum(-1)
        kept = t.searchsorted(cumulative_probs, top_p, side="left").item() + 1
        kept = max(kept, min_tokens_to_keep)
        kept_idx = ids[:kept]
        kept_logits = logits[kept_idx]

        sample = t.distributions.categorical.Categorical(logits=kept_logits).sample()
        return kept_idx[sample].item()
    
    

    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int | None = None,
    ) -> list[tuple[float, str]]:
        """
        Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
        prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
        we've generated `num_returns_sequences` terminating sequences.
        """
        # raise NotImplementedError()


t.set_grad_enabled(False)  # gradients are not necessary for sampling


model = DemoTransformer(Config()).to(device)
model.load_state_dict(reference_gpt2.state_dict(), strict=False)
tokenizer = reference_gpt2.tokenizer
sampler = TransformerSampler(model, tokenizer)

prompt = "Jingle bells, jingle bells, jingle all the way"
print(f"Testing greedy decoding\nPrompt:   {prompt!r}")

expected = "Jingle bells, jingle bells, jingle all the way up to the top of the mountain."
output = sampler.sample(prompt, max_tokens_generated=8, temperature=0.0)

print(f"Expected: {expected!r}\nActual:   {output!r}\n")
assert output == expected

print("Tests passed!")

# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097,
}
frequency_of_top_5 = defaultdict(int)

N = 10_000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits)
    frequency_of_top_5[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word]
    observed_freq = frequency_of_top_5[word] / N
    print(
        f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}"
    )
    assert abs(observed_freq - expected_freq) < 0.01, (
        "Try increasing N if this fails by a small amount."
    )

print("Tests passed!")
# %%
logits = t.tensor([1, 2]).log()

cold_logits = TransformerSampler.apply_temperature(logits, temperature=0.001)
print('A low temperature "sharpens" or "peaks" the distribution: ', cold_logits)
t.testing.assert_close(cold_logits, 1000.0 * logits)

hot_logits = TransformerSampler.apply_temperature(logits, temperature=1000.0)
print("A high temperature flattens the distribution: ", hot_logits)
t.testing.assert_close(hot_logits, 0.001 * logits)

print("Tests passed!")
# %%
sampler = TransformerSampler(model, tokenizer)

N_RUNS = 1
your_prompt = "Jingle bells, jingle bells, jingle all the way"
cases = [
    ("High freq penalty", dict(frequency_penalty=100.0)),
    ("Negative freq penalty", dict(frequency_penalty=-3.0)),
    ("Too hot!", dict(temperature=2.0)),
    ("Pleasantly cool", dict(temperature=0.7)),
    ("Pleasantly warm", dict(temperature=0.9)),
    ("Too cold!", dict(temperature=0.01)),
]

table = Table("Name", "Kwargs", "Output", title="Sampling - Manual Testing")

for name, kwargs in cases:
    for i in range(N_RUNS):
        output = sampler.sample(your_prompt, max_tokens_generated=24, **kwargs)
        table.add_row(name, str(kwargs), repr(output) + "\n")

rprint(table)
# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_5 = {
    " church": 0.0648,
    " house": 0.0367,
    " temple": 0.0145,
    " same": 0.0104,
    " Church": 0.0097,
}
topk_5_sum = sum(expected_top_5.values())

observed_freqs = defaultdict(int)

N = 10000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_k=5)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_5:
    expected_freq = expected_top_5[word] / topk_5_sum
    observed_freq = observed_freqs[word] / N
    print(
        f"Word: {word!r:<9}. Expected freq = {expected_freq:.4f}, observed freq = {observed_freq:.4f}"
    )
    assert abs(observed_freq - expected_freq) < 0.01
# %%
sampler = TransformerSampler(model, tokenizer)

your_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

output = sampler.sample(your_prompt, temperature=0.7, top_k=40, max_tokens_generated=64)

rprint(f"Your model said:\n\n[bold dark_orange]{output}")
# %%
prompt = "John and Mary went to the"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
logits = model(input_ids)[0, -1]

expected_top_10pct = {
    " church": 0.0648,
    " house": 0.0367,  # These are the two most likely tokens, and add up to >10%
}
top_10pct_sum = sum(expected_top_10pct.values())

observed_freqs = defaultdict(int)

N = 10_000
for _ in tqdm(range(N)):
    token = TransformerSampler.sample_next_token(input_ids.squeeze(), logits, top_p=0.1)
    observed_freqs[tokenizer.decode(token)] += 1

for word in expected_top_10pct:
    expected_freq = expected_top_10pct[word] / top_10pct_sum
    observed_freq = observed_freqs[word] / N
    print(
        f"Word: {word!r:<9}. Expected freq {expected_freq:.4f}, observed freq {observed_freq:.4f}"
    )
    assert abs(observed_freq - expected_freq) < 0.01, (
        "Try increasing N if this fails by a small amount."
    )
# %%
sampler = TransformerSampler(model, tokenizer)

your_prompt = "Barack Obama is an American best known for"
output = sampler.sample(your_prompt, temperature=0.7, top_p=0.95, max_tokens_generated=1000)
rprint(f"Your model said:\n\n[bold dark_orange]{output}")
# %%
## Implementation of BeamSearch

@dataclass
class Beams:
    """Class to store beams during beam search."""

    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def __getitem__(self, batch_idx) -> "Beams":
        """Allows you to create new beams from old beams by slicing along batch dim (useful for `filter`)."""
        return Beams(
            self.model, self.tokenizer, self.logprob_sums[batch_idx], self.tokens[batch_idx]
        )

    @property
    def logprobs_and_completions(self) -> list[tuple[float, str]]:
        """Returns self as a list of logprob sums and completions (useful for getting final output)."""
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]

    def generate(self, k: int, no_repeat_ngram_size: int | None = None) -> "Beams":
        """
        Starting from the current set of beams (i.e. self.tokens) and returns a new set of `len(self.tokens) * k` beams,
        containing the best `k` continuations for each of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with a repeating n-gram
        of this length.
        """
        # We want the logprobs outputted for the next token, for each sequence in the current set of beams.
        logprobs = self.model(self.tokens)[:, -1, :].log_softmax(-1)
        topk_logprobs, topk_ids = self.get_topk_non_repeating(logprobs, no_repeat_ngram_size, k=k)
        # Now we need to add in the new logprobs and concatenate!
        new_logprob_sums = einops.repeat(self.logprob_sums, "b -> b k", k=k) + topk_logprobs
        new_tokens = t.concat([einops.repeat(self.tokens, "b s -> b k s", k=k), topk_ids.unsqueeze(-1)], dim=-1)

        return Beams(self.model, self.tokenizer, new_logprob_sums.flatten(), new_tokens.flatten(0, 1))

    def filter(self, k: int) -> tuple["Beams", "Beams"]:
        """
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `k` which are also not terminated.
            early_terminations: Beams
                filtered version of self, containing all best `k` which are also terminated.
        """
        # Get the indices of top `k` beams
        top_beam_indices = self.logprob_sums.topk(k=k, dim=0).indices.tolist()
        # Get the indices of terminated sequences
        new_tokens = self.tokens[:, -1]
        terminated_indices = t.nonzero(new_tokens == self.tokenizer.eos_token_id)

        # Get the indices of the `k` best sequences (some terminated, some not terminated)
        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]

        # Return the beam objects from these indices
        return self[best_continuing], self[best_terminated]

    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: int | None,
        k: int,
    ) -> tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        """
        logprobs:
            tensor of the log-probs for the next token
        no_repeat_ngram_size:
            size of ngram to avoid repeating
        k:
            number of top logits to return, for each beam in our collection

        Returns:
            equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure that no returned tokens would produce an
            ngram of size `no_repeat_ngram_size` which has already appeared in `self.tokens`.
        """
        batch, seq_len = self.tokens.shape

        # If completion isn't long enough for a repetition, or we have no restrictions, just return topk
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size - 1):
            # Otherwise, we need to check for ngram repetitions
            # First, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size - 1) :]
            # Next, find all the tokens we're not allowed to generate, by checking all past ngrams for a match
            for i in range(seq_len - (no_repeat_ngram_size - 1)):
                ngrams = self.tokens[:, i : i + no_repeat_ngram_size]  # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(-1)  # (batch,)
                ngram_end_tokens = ngrams[:, [-1]]  # (batch, 1)
                # Fill logprobs with neginf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = t.where(
                    ngrams_are_repeated, -1.0e4, logprobs[range(batch), ngram_end_tokens]
                )

        # Finally, get our actual tokens
        return logprobs.topk(k=k, dim=-1)

    def print(self, title="Best completions", max_print_chars=80) -> None:
        """
        Prints out a set of sequences with their corresponding logprob sums.
        """
        if len(self.tokens) == 0:
            return
        table = Table("logprob sum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = (
                    text[: int(0.3 * max_print_chars)]
                    + " ... "
                    + text[-int(0.7 * max_print_chars) :]
                )
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int | None = None,
) -> list[tuple[float, str]]:
    """
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting from the initial
    prompt) until either of the two stopping criteria are met: (1) we've generated `max_new_tokens` tokens, or (2)
    we've generated `num_returns_sequences` terminating sequences.
    """
    assert num_return_sequences <= num_beams
    self.model.eval()

    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

    final_logprobs_and_completions = []  # we add to this list as we get terminated beams
    best_beams = Beams(
        self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens
    )  # start with just 1 beam

    for _ in tqdm(range(max_new_tokens)):
        t.cuda.empty_cache()

        # Generate & filter beams
        best_beams = best_beams.generate(k=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)
        best_beams, best_beams_terminated = best_beams.filter(k=num_beams)

        # Add terminated beams to our list, and return early if we have enough
        final_logprobs_and_completions.extend(best_beams_terminated.logprobs_and_completions)
        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    # Return terminated beams plus the best ongoing beams of length `orig_len + max_new_tokens`
    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    return final_logprobs_and_completions[:num_return_sequences]


TransformerSampler.beam_search = beam_search
# %%
# Start with prompt "When I was", get top 3 tokens (and their logprobs), and use that to create & display the top 3 beams
prompt = "When I was"
tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
logprobs = model(tokens)[0, -1].log_softmax(-1)
top_logprobs, top_tokens = logprobs.topk(k=3, dim=-1)

new_tokens = t.concat([tokens.repeat(3, 1), top_tokens.unsqueeze(-1)], dim=-1)

beams = Beams(model, tokenizer, logprob_sums=top_logprobs, tokens=new_tokens)
beams.print()
# %%
print("Testing generate...")
new_beams = beams.generate(k=3, no_repeat_ngram_size=1)
new_beams.print()

expected_values = [
    (-3.1, "When I was a kid"),
    (-4.8, "When I was a child"),
    (-4.9, "When I was a little"),
]

for i, (logprob_sum, completion) in enumerate(new_beams.logprobs_and_completions[:3]):
    assert abs(logprob_sum - expected_values[i][0]) < 0.1, f"{i}"
    assert completion == expected_values[i][1], f"{i}"

print("All tests for `generate` passed!")
# %%
print("Testing `filter`...")

best_beams, terminated_beams = new_beams.filter(3)
best_beams.print()

expected_values = [
    (-3.1, "When I was a kid"),
    (-3.2, "When I was growing up"),
    (-4.6, "When I was in the"),
]

for i, (logprob_sum, completion) in enumerate(best_beams.logprobs_and_completions):
    assert abs(logprob_sum - expected_values[i][0]) < 0.1, f"{i}"
    assert completion == expected_values[i][1], f"{i}"

assert len(terminated_beams.logprobs_and_completions) == 0

print("All tests for `filter` passed!")
# %%
print("Testing `no_repeat_ngram_size`...")

new_beams = beams
for _ in range(5):
    new_beams = new_beams.generate(k=1)
new_beams.print(title="Completions with no ngram restriction")
assert all(
    "I was" in completion.removeprefix(prompt)
    for _, completion in new_beams.logprobs_and_completions
), "Without restriction, all beams should be completed as '...I was...'"

new_beams = beams
for _ in range(5):
    new_beams = new_beams.generate(k=1, no_repeat_ngram_size=2)
new_beams.print(title="Completions with no repeated bigrams")
assert all(
    "I was" not in completion.removeprefix(prompt)
    for _, completion in new_beams.logprobs_and_completions
), "With no repeated bigrams, no beams should contain a second '...I was...'"
# %%
sampler = TransformerSampler(model, tokenizer)

prompt = "The ships hung in the sky in much the same way that"
orig_len = len(tokenizer.encode(prompt))

final_logitsums_and_completions = sampler.beam_search(
    prompt=prompt,
    num_return_sequences=3,
    num_beams=40,
    max_new_tokens=60,
    no_repeat_ngram_size=2,
)

# Print all the best output
for logprob_sum, text in final_logitsums_and_completions:
    avg_logprob_as_prob = t.tensor(logprob_sum / (len(tokenizer.encode(text)) - orig_len)).exp()
    rprint(f"Avg token prob = {avg_logprob_as_prob:.3f}\nBest output:\n[bold dark_orange]{text}")
# %%
