"""Train an NCM Sudoku generator from scratch on top of the released, frozen tokenizer.

Hazy Research released NCM's models and evaluation but not its training code. This follows the paper: the tokenizer
is frozen; the generator sees the puzzle as a prefix and, for each level, the cumulative reconstruction of the coarser
levels, and predicts every level's codes in one forward pass. A fraction of the conditioning codes is corrupted so
the model learns to recover from its own mistakes at generation time.

  python train.py --ncm C:/path/ncm --ckpt <ncm checkpoints>/sudoku/hard --data train-hard-200k.pickle \
                  --test <ncm data>/sudoku/hard --layers 4 --embd 256 --heads 4 --steps 20000 --out runs/small
  python train.py ... --check-released     # teacher-forced accuracy of the released generator: validates the inputs
"""
import argparse, json, math, os, pickle, sys, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--ncm", required=True, help="HazyResearch/ncm checkout (with ncm-cpu-port.patch applied)")
ap.add_argument("--ckpt", required=True, help="released sudoku/<level> dir: tokenizer.pt (frozen), generator.pt for --check-released")
ap.add_argument("--data", default="")
ap.add_argument("--test", required=True)
ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--embd", type=int, default=256)
ap.add_argument("--heads", type=int, default=4)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=6e-4)
ap.add_argument("--corrupt", type=float, default=0.1, help="fraction of conditioning codes replaced at random")
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--eval-n", type=int, default=200)
ap.add_argument("--out", default="runs/ncm-small")
ap.add_argument("--check-released", action="store_true")
ap.add_argument("--resume", action="store_true", help="continue from <out>/state.pt (model, optimizer, step)")
a = ap.parse_args()

sys.path.insert(0, a.ncm)
os.chdir(a.ncm)                                              # the registry resolves classes relative to the repo
from omegaconf import OmegaConf
from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

dev = "cuda" if torch.cuda.is_available() else "cpu"
CFG = OmegaConf.load(os.path.join(a.ncm, "config", "eval", "sudoku-hard.yaml"))
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(a.ckpt, "tokenizer.pt")
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vq = tk.vqvae.to(dev).eval()
for p in vq.parameters():
    p.requires_grad_(False)

mcfg = OmegaConf.create(OmegaConf.to_container(CFG.model))
if not a.check_released:
    mcfg.n_layer, mcfg.n_embd, mcfg.n_head = a.layers, a.embd, a.heads
net = NextConceptTransformer(vqvae_local=vq, config=mcfg).to(dev)
if a.check_released:
    net.load_state_dict(torch.load(os.path.join(a.ckpt, "generator.pt"), map_location="cpu", weights_only=True)["model"], strict=True)
P, LEVELS = int(mcfg.prefix_len), list(vq.vq.levels)
V = int(mcfg.vocab_size[0])


@torch.no_grad()
def encode_all(sol, bs=2048):
    """The frozen tokenizer's codes for every solution, computed once (encoding is deterministic)."""
    return torch.cat([vq.encode(sol[s:s + bs]) for s in range(0, sol.shape[0], bs)])


def batch_inputs(puz, sol, corrupt, idx=None):
    """Teacher-forced inputs and targets for a batch of (puzzle, solution) token tensors."""
    with torch.no_grad():
        if idx is None:
            idx = vq.encode(sol)                             # (B, 322) codes, coarse to fine
        cond = idx
        if corrupt > 0:
            flip = torch.rand(idx.shape, device=dev) < corrupt
            cond = torch.where(flip, torch.randint_like(idx, V), idx)
        nxt = vq.vq.idx_to_ncm_input(cond)                   # (B, 321, C): levels 1.. see the coarser reconstruction
        z_prefix = vq.encoder.wte(puz)[:, :P, :]             # raw_token_prefix: the puzzle digits
    q0 = net.build_l0_query_input(puz.shape[0], nxt.dtype)   # learned query for level 0 (trainable)
    return torch.cat([z_prefix, q0, nxt], dim=1), idx


@torch.no_grad()
def solve(puz):
    g = net.generate(None, puz, temperature=0.0, top_k=0, top_p=1.0, cfg_scale=0.0, cfg_null_eos=11)
    _, zl = vq.vq.idx_to_zhats(g, return_zlevels=True)
    return vq.decoder(zl[-1]).argmax(-1)                     # (B, 89) tokens


def evaluate(puzzles, solutions, n, bs=100):
    net.eval()
    solved = cells = 0
    for s in range(0, n, bs):
        pz = torch.tensor(puzzles[s:s + bs], device=dev)
        ref = torch.tensor(solutions[s:s + bs], device=dev)
        out = solve(pz)
        solved += (out == ref).all(-1).sum().item()
        cells += (out == ref).float().mean(-1).sum().item()
    net.train()
    return solved / n, cells / n


test_p = pickle.load(open(os.path.join(a.test, "puzzles.pickle"), "rb"))
test_s = pickle.load(open(os.path.join(a.test, "solutions.pickle"), "rb"))
n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

if a.check_released:
    net.eval()
    with torch.no_grad():
        x, idx = batch_inputs(torch.tensor(test_p[:256], device=dev), torch.tensor(test_s[:256], device=dev), 0.0)
        logits = net(x)
        acc = (logits.argmax(-1) == idx).float().mean().item()
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), idx.reshape(-1)).item()
    solved, cell = evaluate(test_p, test_s, 200)
    print(f"released generator ({n_params / 1e6:.1f}M): teacher-forced code accuracy {acc:.4f}, loss {loss:.4f}; "
          f"generation solved {solved:.1%} of 200, cells {cell:.2%}")
    sys.exit()

d = pickle.load(open(a.data, "rb"))
train_p = torch.tensor(d["puzzles"], device=dev)
train_s = torch.tensor(d["solutions"], device=dev)
t_enc = time.time()
train_idx = encode_all(train_s)
N = train_p.shape[0]
print(f"encoded {N} solutions in {time.time() - t_enc:.0f} s", flush=True)
opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.05)
sched = lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps)))   # warm-up, then cosine
os.makedirs(a.out, exist_ok=True)
log = open(os.path.join(a.out, "log.jsonl"), "a")
print(f"training {n_params / 1e6:.2f}M parameters on {N} boards, {dev}", flush=True)

start = 1
state_path = os.path.join(a.out, "state.pt")
if a.resume and os.path.exists(state_path):
    st = torch.load(state_path, map_location=dev, weights_only=True)
    net.load_state_dict(st["model"])
    opt.load_state_dict(st["opt"])
    start = st["step"] + 1
    print(f"resumed at step {start}", flush=True)

net.train()
t0, seen = time.time(), 0
for step in range(start, a.steps + 1):
    for g in opt.param_groups:
        g["lr"] = a.lr * sched(step)
    i = torch.randint(0, N, (a.bs,), device=dev)
    x, idx = batch_inputs(train_p[i], None, a.corrupt, train_idx[i])
    with torch.autocast(device_type="cuda" if dev == "cuda" else "cpu", dtype=torch.bfloat16):
        logits = net(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), idx.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    seen += a.bs
    if step % 100 == 0:
        rec = {"step": step, "loss": round(loss.item(), 4), "lr": round(a.lr * sched(step), 6),
               "boards_per_s": round(seen / (time.time() - t0), 1), "minutes": round((time.time() - t0) / 60, 1)}
        if step % a.eval_every == 0 or step == a.steps:
            rec["solved"], rec["cells"] = evaluate(test_p, test_s, a.eval_n)
            torch.save({"model": net.state_dict(), "config": OmegaConf.to_container(mcfg), "step": step},
                       os.path.join(a.out, "generator.pt"))
        if step % 500 == 0 or step == a.steps:                # resumable state: an interruption costs minutes, not hours
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(), "step": step}, state_path + ".tmp")
            os.replace(state_path + ".tmp", state_path)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)
