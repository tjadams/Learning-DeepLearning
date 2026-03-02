# TylerVLA
From-scratch/vibe-coded implementation of the simplest possible VLA that me and ChatGPT could think of.

## Project Goals
(a) code VLA end-to-end from scratch / vibe-coded 
(b) train on literally a couple minutes of real-world data from my SO-ARM-101
(c) maybe simulate in MuJoCo
(d) deploy on an SO-ARM-101

## Overview
Tyler VLA = frozen encoders + tiny action head (BC)

Frozen V+L encoders → tiny policy head → actions
(You only train the tiny head; everything else is fixed.)

CLIP-BC Policy
- Vision: frozen CLIP image encoder (or any pretrained CNN/Vit you can load)
- Language: frozen CLIP text encoder (same model)
- Fusion (combine vision + language): concatenate embeddings (or FiLM if you want one extra step)
- Policy head: small MLP (2–4 layers)
- Action output: SO-ARM-101 commands (e.g., Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper)

Why this fits “couple minutes”: you’re not trying to learn perception or language from scratch—just a small mapping from already-meaningful embeddings to actions.

Inputs
- Image: 128×128 or 224×224 RGB
- Text: command string (e.g., “pick up the red block”)

Forward pass
v = VisionEncoder(img) → 512 or 768 dim (frozen)
t = TextEncoder(text) → 512 or 768 dim (frozen)
z = concat([v, t]) (optionally add previous action / proprio)
â = MLP(z) → action dims

Loss
- Behavior cloning (supervised): L = MSE(â, a) (and BCE for gripper if binary)

Inference loop
- Read camera frame
- Encode text once per episode
- Run policy at e.g. 10–30 Hz
- Send action to SO-ARM-101 controller

Couple minutes of data can work, but only under these conditions:
- You’re training one task (or a couple very similar tasks)
- You can run at 10–30 Hz to get enough samples (2 min @ 20 Hz ≈ 2400 steps)
- You keep the policy output simple (delta pose + gripper)
- You do normalization + action smoothing
- You accept it may overfit and behave well only in the same setup/lighting

## Source
My private ChatGPT convo: https://chatgpt.com/c/69a4fe0f-95b4-8333-9c24-dd978ce531d1 