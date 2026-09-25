"""Action codecs: MineStudio sandbox dict ↔ pretrained_wm (keyboard[4], mouse[2]).

Also parses cold-start Qwen VLA text:
  <actions> keyPress(w) ; keyPress(w) ; ... </actions>
into a list of MineStudio action dicts (copied semantics from
ares/.../qwen3vl_minestudio_agent_align_cold_start.py — local copy so we do
not edit ares agent sources for this bridge).
"""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch

# pretrained_wm latent training uses 4 movement keys + camera (pitch, yaw) normalized.
KB_DIM = 4
IDX_TO_KEY = {0: "forward", 1: "back", 2: "left", 3: "right"}
KEY_TO_IDX = {v: k for k, v in IDX_TO_KEY.items()}

# From train/data/vpt_to_worldmodel.py
CAM_VALUE = 20.0
CAMERA_CLIP_DEG = 10.0
# Cold-start / OpenHA camera scaler (matches free_energy agent)
CAMERA_SCALER = 360.0 / 2400.0

_DEFAULT_ACTION: Dict[str, Any] = {
    "ESC": 0, "attack": 0, "back": 0, "camera": [0.0, 0.0], "drop": 0,
    "forward": 0, "inventory": 0, "jump": 0, "left": 0, "right": 0,
    "sneak": 0, "sprint": 0, "use": 0, "pickItem": 0, "swapHands": 0,
}
for _i in range(1, 10):
    _DEFAULT_ACTION[f"hotbar.{_i}"] = 0

_KEYPRESS_TO_MINESTUDIO = {
    "w": "forward", "s": "back", "a": "left", "d": "right",
    "space": "jump", "left.ctrl": "sprint", "left.shift": "sneak",
    "q": "drop", "e": "inventory", "esc": "ESC", "escape": "ESC",
}

ACTIONS_PATTERN = re.compile(r"<actions>(.*?)</actions>", re.DOTALL)


def noop_action() -> Dict[str, Any]:
    a = _DEFAULT_ACTION.copy()
    a["camera"] = [0.0, 0.0]
    return a


def sandbox_to_mg2(action: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """One MineStudio tick → pretrained_wm (kb[4], mouse[2])."""
    kb = torch.zeros(KB_DIM)
    for idx, key in IDX_TO_KEY.items():
        if float(action.get(key, 0)) > 0.5:
            kb[idx] = 1.0
    pitch = float(action.get("camera", [0.0, 0.0])[0])
    yaw = float(action.get("camera", [0.0, 0.0])[1])
    mouse = torch.tensor([
        max(-CAM_VALUE, min(CAM_VALUE, pitch / CAMERA_CLIP_DEG * CAM_VALUE)),
        max(-CAM_VALUE, min(CAM_VALUE, yaw / CAMERA_CLIP_DEG * CAM_VALUE)),
    ])
    return kb, mouse


def mg2_to_sandbox(kb_row, mouse_row) -> Dict[str, Any]:
    a = noop_action()
    for idx, key in IDX_TO_KEY.items():
        if float(kb_row[idx]) > 0.5:
            a[key] = 1
    pitch = float(mouse_row[0]) / CAM_VALUE * CAMERA_CLIP_DEG
    yaw = float(mouse_row[1]) / CAM_VALUE * CAMERA_CLIP_DEG
    a["camera"] = [pitch, yaw]
    return a


def chunk_to_mg2(actions: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """List of MineStudio ticks → kb[T,4], mouse[T,2]."""
    kbs, mss = [], []
    for a in actions:
        kb, ms = sandbox_to_mg2(a)
        kbs.append(kb)
        mss.append(ms)
    if not kbs:
        return torch.zeros(1, KB_DIM), torch.zeros(1, 2)
    return torch.stack(kbs, 0), torch.stack(mss, 0)


def _clamp_camera(v: float, limit: float = 180.0) -> float:
    return max(-limit, min(limit, v))


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        return s[1:-1]
    return s


def _apply_atomic(token: str, action: Dict[str, Any], camera_scaler: float) -> bool:
    token = token.strip()
    if not token:
        return False
    low = token.lower()
    if low in ("no_op", "no_op()"):
        return False
    if low in ("done", "done()"):
        return True

    m = re.match(r"keyPress\((.+)\)", token)
    if m:
        for raw in re.split(r",|\s+and\s+", m.group(1)):
            key = _strip_quotes(raw).lower()
            if not key:
                continue
            if key in _KEYPRESS_TO_MINESTUDIO:
                action[_KEYPRESS_TO_MINESTUDIO[key]] = 1
            elif key.isdigit() and 1 <= int(key) <= 9:
                action[f"hotbar.{int(key)}"] = 1
        return False

    m = re.match(r"mouseClick\(\s*['\"]?(\w+)['\"]?\s*\)", token)
    if m:
        btn = m.group(1).lower()
        if btn == "left":
            action["attack"] = 1
        elif btn == "right":
            action["use"] = 1
        elif btn == "middle":
            action["pickItem"] = 1
        return False

    m = re.match(r"mouse[Mm]ove\(([^,]+),\s*([^)]+)\)", token)
    if m:
        try:
            dx = float(re.sub(r"[a-zA-Z_=]", "", m.group(1)).strip())
            dy = float(re.sub(r"[a-zA-Z_=]", "", m.group(2)).strip())
            cam = action["camera"]
            action["camera"] = [
                _clamp_camera(cam[0] + dy * camera_scaler),
                _clamp_camera(cam[1] + dx * camera_scaler),
            ]
        except ValueError:
            pass
        return False
    return False


def parse_actions_text(
    text: str,
    action_chunks_len: int = 4,
    camera_scaler: float = CAMERA_SCALER,
) -> Tuple[List[Dict[str, Any]], bool]:
    """Parse Qwen cold-start output → (list of MineStudio actions, done)."""
    blocks = ACTIONS_PATTERN.findall(text)
    if not blocks:
        return [], False
    sub_steps = [s.strip() for s in blocks[-1].split(";")]
    actions: List[Dict[str, Any]] = []
    done = False
    for sub in sub_steps:
        action = noop_action()
        low = sub.strip().lower()
        if low in ("done", "done()"):
            done = True
            break
        tokens = re.findall(r"[a-zA-Z_]+\([^)]*\)|no_op|done", sub)
        for tok in tokens:
            if _apply_atomic(tok, action, camera_scaler):
                done = True
                break
        if done:
            break
        actions.append(action)
    if not done:
        pad = [noop_action() for _ in range(action_chunks_len)]
        actions = (actions + pad)[:action_chunks_len]
    return actions, done


# Legacy short, single-turn prompt used by the original k1_t0 experiment.
PROMPT_MINECRAFT_LEGACY = """You are an AI agent performing tasks in Minecraft based on given instructions, action history, and visual observations(screenshots). Your goal is to choose the next 200ms of actions, composed of 4 steps, each spaced 50ms apart. each action begins at its execution time and lasts for 50ms.

## Action Space
* mouseMove(dx, dy)
* mouseClick(left or right or middle)
* keyPress(keys)  # w/s/a/d/space/q/e/1-9/left.ctrl/left.shift/ESC
* no_op
If multiple actions are activated in one action step, use and connect eg.  mouseMove(10, 10) mouseClick(left) keyPress(w, a) .

# Instruction
1. The output consists of exactly 4 action steps. each step is separated by ;.
2. Output ONLY the plain text wrapped in <actions> and </actions>. Do not add line breaks, quotes, or any additional text.

## Output Format
<actions> action1 action2 ; action3 ; action4 ; action5 </actions>

## User Instruction
{instruction}
"""


# Full OpenHA / collaborator_b minecraft_v1 system prompt (byte-aligned with
# collaborator_a/.../openha_eval.py MINECRAFT_V1_SYSTEM_PROMPT and gamedojo
# prompts/minecraft_v1.md). Use this for eval parity.
PROMPT_MINECRAFT_V1 = """You are an AI agent performing tasks in Minecraft based on given instructions, action history, and visual observations(screenshots). Your goal is to choose the next 200ms of actions, composed of 4 steps, each spaced 50ms apart. each action begins at its execution time and lasts for 50ms.

## Action Space
* mouseMove(dx, dy) # Move the mouse position; dx and dy represent horizontal and vertical movement, respectively.
* mouseClick(left or right or middle) # left click, right click, or middle click the mouse
- left # Attack; In GUI, pick up the stack of items or place the stack of items in a GUI cell; when used as a double click
(attack - no attack - attack sequence), collect all items of the same kind present in inventory as a single stack.
- right # Place the item currently held or use the block the player is looking at. In GUI, pick up the stack of items or
place a single item from a stack held by mouse.
* keyPress(keys) # press the keyboard buttons
- w # Move forward.
- s # Move backward.
- a # Strafe left.
- d # Strafe right.
- e # Open or close inventory and the 2x2 crafting grid.
- space # Jump.
- q # Drop a single item from the stack of items the player is currently holding. If the player presses ctrl-Q then it
drops the entire stack. In the GUI, the same thing happens except to the item the mouse is hovering over.
- 1-9 # Switch active item to the one in a given hotbar cell.
- left.ctrl # Move fast in the current direction of motion.
- left.shift # Move carefully in current direction of motion. In the GUI it acts as a modifier key: when used with attack
it moves item from/to the inventory to/from the hotbar, and when used with craft it crafts the maximum number of
items possible instead of just 1.
- ESC # Open or close inventory and the 2x2 crafting grid.
* no_op # wait and do not interact with the world
If multiple actions are activated in one action step, use and connect eg.  mouseMove(10, 10) mouseClick(left) keyPress(w, a) .
Your history thoughts will accumulate continuously in history conversations.


# Instruction
1. The output consists of exactly 4 action steps. each step is separated by ;. for example if you want to keep move forward in current direction for 200ms, you could output: keyPress(w) ; keyPress(w) ; keyPress(w) ; keyPress(w)
2. Output ONLY the plain text warpped in <actions> and </actions> in the exact format above. Do not add line breaks, quotes, or any additional text.

## Output Format
<actions> action1 action2 action3 ; action4 action5 ; action6 ; action7 action8 </actions>

## User Instruction
{instruction}
"""

# Backward-compatible alias (legacy short prompt callers).
PROMPT_MINECRAFT_NO_THOUGHT = PROMPT_MINECRAFT_V1


def cold_start_step_instruction(instruction: str, step_num: int) -> str:
    """Mirror OpenHA MinecraftV1Game._build_step_instruction."""
    if int(step_num) == 0:
        return f"Task: {instruction}\nChoose your actions."
    return "Choose your next actions."
