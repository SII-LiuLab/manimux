#!/usr/bin/env python3
"""Pre-flight for the RDT2-FM smoke train: run one real forward, nothing else.

Answers, in order, the things that can silently go wrong before a training run
is worth starting:

  1. Does ckpt/RDT2-FM actually load into RDTRunner? models/hub_mixin.py:36
     loads with strict=False, so a shape/name mismatch would be SILENT and we
     would "fine-tune" a randomly initialised model. We diff the keys ourselves.
  2. Does our webdataset collate into what the model expects?
  3. Does the Qwen2.5-VL backbone forward and hand back 14 layers of KV cache?
  4. What is the initial loss, and how much VRAM does one step actually take?

Run from anywhere; it cds into the RDT2 repo itself.
"""
import os
import sys
from functools import partial

RDT2_DIR = os.environ.get("RDT2_DIR", "/home/jw/Desktop/project/RDT2")
WDS_CONFIG = os.environ.get(
    "WDS_CONFIG", "/home/jw/Desktop/dataset/exchange_ball_v0_rdt2/dataset.yaml"
)
BATCH = int(os.environ.get("BATCH", "2"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.chdir(RDT2_DIR)
sys.path.insert(0, RDT2_DIR)

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

from models.rdt_runner import RDTRunner  # noqa: E402
from models.normalizer import LinearNormalizer  # noqa: E402
from rdt.dataset import collate_fn, get_instructions_and_blended_train_dataset  # noqa: E402


def hr(title):
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def vram(tag):
    if torch.cuda.is_available():
        print(f"  [vram] {tag}: alloc {torch.cuda.memory_allocated()/2**30:.2f} GiB  "
              f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")


def main():
    dev = torch.device("cuda")
    dtype = torch.bfloat16
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"arch {torch.cuda.get_arch_list()}")
    print(f"gpu: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    with open("configs/rdt/post_train.yaml") as f:
        config = yaml.safe_load(f)

    # ---- 1. checkpoint load, verified, not trusted ----------------------
    hr("1. RDT2-FM checkpoint")
    fm_dir = os.path.join(RDT2_DIR, "ckpt", "RDT2-FM")
    rdt = RDTRunner.from_pretrained(fm_dir)
    n_params = sum(p.numel() for p in rdt.parameters())
    print(f"  RDTRunner built: {n_params/1e6:.1f}M params")

    # hub_mixin loads with strict=False -> diff the keys by hand.
    raw = torch.load(os.path.join(fm_dir, "pytorch_model.bin"), map_location="cpu", weights_only=False)
    raw = raw.get("module", raw)
    model_keys, ckpt_keys = set(rdt.state_dict()), set(raw)
    missing, unexpected = model_keys - ckpt_keys, ckpt_keys - model_keys
    print(f"  ckpt tensors {len(ckpt_keys)} / model tensors {len(model_keys)}")
    print(f"  missing (left at random init): {len(missing)}")
    print(f"  unexpected (ignored in ckpt) : {len(unexpected)}")
    for name in list(missing)[:8]:
        print(f"    ! missing  {name}")
    for name in list(unexpected)[:8]:
        print(f"    ! unexpect {name}")

    # Shape mismatches also pass silently under strict=False.
    bad = [k for k in model_keys & ckpt_keys
           if tuple(rdt.state_dict()[k].shape) != tuple(raw[k].shape)]
    print(f"  shape mismatches: {len(bad)}")
    for k in bad[:8]:
        print(f"    ! {k}: model {tuple(rdt.state_dict()[k].shape)} "
              f"vs ckpt {tuple(raw[k].shape)}")

    # Prove the weights actually landed: compare a tensor value.
    probe = sorted(model_keys & ckpt_keys)[0]
    same = torch.equal(rdt.state_dict()[probe].cpu().float(), raw[probe].cpu().float())
    print(f"  probe tensor '{probe}' equals ckpt: {same}")
    if missing or bad or not same:
        print("  >> CHECKPOINT DID NOT FULLY LOAD. Fine-tuning would start from "
              "partly-random weights.")
    del raw

    rdt = rdt.to(dev, dtype=dtype)
    rdt.train()
    vram("rdt on gpu")

    # ---- 2. data ---------------------------------------------------------
    hr("2. dataset + collate")
    with open(WDS_CONFIG) as f:
        wds_config = yaml.safe_load(f.read().format(hostname="preflight"))
    instructions, train_dataset = get_instructions_and_blended_train_dataset(wds_config)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", padding_side="left", use_fast=True
    )
    cfn = partial(collate_fn, processor=processor, instructions=instructions,
                  image_corruption=False, state_dim=config["common"]["state_dim"])
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH, collate_fn=cfn, num_workers=0
    )
    batch = next(iter(loader))
    for k, v in batch["vision_language_model_inputs"].items():
        print(f"  vlm_inputs[{k}]: {tuple(v.shape)} {v.dtype}")
    print(f"  actions: {tuple(batch['actions'].shape)} {batch['actions'].dtype}")
    print(f"  states : {tuple(batch['states'].shape)} (all zeros: "
          f"{bool((batch['states'] == 0).all())})")

    normalizer = LinearNormalizer.load(wds_config["kwargs"]["normalizer_path"])
    nsamples = normalizer["action"].normalize(batch["actions"])
    print(f"  normalized actions: range [{nsamples.min():.3f}, {nsamples.max():.3f}], "
          f"mean {nsamples.mean():.3f}, std {nsamples.std():.3f}")
    if nsamples.abs().max() > 10:
        print("  >> normalized actions are far outside [-1,1]: our data does not "
              "match the UMI normalizer's assumed scale.")

    # ---- 3. VLM forward --------------------------------------------------
    hr("3. Qwen2.5-VL backbone forward")
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        os.path.join(RDT2_DIR, "ckpt", "RDT2-VQ"),
        torch_dtype=dtype, attn_implementation="flash_attention_2", device_map=dev,
    )
    vlm.eval()
    vram("vlm loaded")
    inputs = {k: v.to(dev) for k, v in batch["vision_language_model_inputs"].items()}
    with torch.no_grad():
        out = vlm(**inputs, use_cache=True)
    sel = config["model"]["selected_layers"]
    kv = [out.past_key_values[i] for i in sel]
    print(f"  total layers cached: {len(out.past_key_values)}, selected: {len(kv)}")
    k0, v0 = kv[0]
    print(f"  layer0 K {tuple(k0.shape)} V {tuple(v0.shape)} {k0.dtype}")
    vram("after vlm forward")

    # ---- 4. one loss + backward -----------------------------------------
    hr("4. loss + backward")
    loss = rdt(
        lang_kv_cache=kv,
        lang_attn_mask=inputs["attention_mask"].to(torch.bool),
        img_tokens=None,
        state_tokens=batch["states"].to(dev, dtype),
        action_gt=nsamples.to(dev, dtype),
    )
    print(f"  initial loss: {loss.item():.5f}")
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(rdt.parameters(), 1e9).item()
    # Structurally dead in KV-cache mode, by design -- not a bug:
    #   cond_norm / cross_attn.wkv / cross_attn.norm_k  project language tokens
    #   into K/V, but attention.py:178 takes the `ck`/`cv` branch when a
    #   lang_kv_cache is supplied, so these never see the forward graph.
    #   state_adaptor.0.weight gets zero grad because collate_fn emits
    #   states = torch.zeros(...); RDT2 does not condition on proprioception.
    DEAD = (".cond_norm.weight", ".cross_attn.wkv.weight",
            ".cross_attn.norm_k.weight", "state_adaptor.0.weight")
    dead, live_dry = [], []
    for n, p in rdt.named_parameters():
        has = p.grad is not None and p.grad.abs().sum() > 0
        if has:
            continue
        (dead if n.endswith(DEAD) else live_dry).append(n)
    nz = sum(1 for _, p in rdt.named_parameters()
             if p.grad is not None and p.grad.abs().sum() > 0)
    tot = sum(1 for _ in rdt.parameters())
    print(f"  grad norm: {gnorm:.5f}   params with nonzero grad: {nz}/{tot}")
    print(f"  no grad, expected (kv-cache mode / zero state): {len(dead)}")
    if live_dry:
        print(f"  no grad, UNEXPECTED: {len(live_dry)} -> {live_dry[:8]}")
    vram("after backward")

    hr("verdict")
    ok = (not missing and not bad and same
          and torch.isfinite(loss) and gnorm > 0 and not live_dry)
    peak = torch.cuda.max_memory_allocated() / 2**30
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"  peak VRAM at batch={BATCH}: {peak:.2f} / {total:.1f} GiB "
          f"({100*peak/total:.0f}%)")
    print("  headroom suggests batch <= "
          f"~{max(1, int(BATCH * (total * 0.9) / peak))} (rough, activations "
          "do not scale perfectly linearly)")
    print(f"\n  {'PASS - safe to start the smoke train' if ok else 'FAIL - see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
