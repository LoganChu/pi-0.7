"""The paper's actual memory-label pipeline: an off-the-shelf LLM does the writing.

MEM sec. III-B generates its text-memory training labels by handing a pretrained
language model the subtask instructions for an episode plus a success/failure
flag per subtask, and instructing it to compress -- their example being that
several individually-described bowls should come back as "three bowls in the top
right cabinet". `text_memory.TemplateWriter` reproduces that behaviour with
rules; this reproduces the *mechanism*.

Why keep both. The template writer is deterministic, instant, and always
available, which is what the experiments need. The LLM writer is faithful, and
it is the one that keeps working when the event vocabulary grows past what rules
can enumerate. They are interchangeable: both turn an episode into per-subtask
memory strings.

    uv sync --extra llm
    python -m memlab.llm_memory --task tidy_up --seed 0

STATUS: the prompt construction and response parsing below are covered by tests
and are exercised on every run. The generation call itself needs a downloaded
model and has NOT been run in this repo's test suite -- treat the first call as
unverified.
"""

from __future__ import annotations

import argparse
import re

from .text_memory import Subtask, episode_subtasks

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

SYSTEM_PROMPT = """\
You maintain a robot's memory of a task it is performing in a kitchen.
Rewrite the memory after each subtask so it stays SHORT.

Rules:
- Keep facts the robot still needs: which cabinets it has already searched,
  where objects are now, what it is holding.
- Compress rather than list. "a light green bowl, a dark blue bowl and a
  bright yellow bowl" becomes "three bowls in the top right cabinet".
- Delete facts that are no longer true. If the robot took the bread out of a
  cabinet, the bread is no longer in that cabinet.
- Note a subtask that FAILED, briefly, so the robot does not retry it blindly.
- Reply with the memory only. One line. No preamble, no bullet points.\
"""


def build_prompt(previous_memory: str, subtask: Subtask) -> str:
    """One turn of the labelling conversation."""
    observed = "; ".join(subtask.observations) if subtask.observations else "nothing"
    outcome = "succeeded" if subtask.succeeded else "FAILED"
    return (
        f"Current memory: {previous_memory or 'nothing yet.'}\n"
        f"Subtask just attempted: {subtask.instruction} ({outcome})\n"
        f"Observed: {observed}\n"
        f"Updated memory:"
    )


def parse_response(raw: str) -> str:
    """Take the first real line, and strip the framing small models add.

    Small instruct models reliably ignore "one line, no preamble" some of the
    time, so this is not defensive coding for its own sake.
    """
    text = re.sub(r"<think>.*?</think>", " ", raw, flags=re.DOTALL)
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        line = re.sub(r"^(updated memory|memory)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = line.lstrip("-*• ").strip()
        if line:
            return line
    return ""


class LLMMemoryWriter:
    """Wraps a small instruct model as MEM's memory labeller."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 max_new_tokens: int = 64) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "LLMMemoryWriter needs the 'llm' extra: uv sync --extra llm"
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id).to(self.device).eval()

    def write(self, previous_memory: str, subtask: Subtask) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(previous_memory, subtask)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        completion = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return parse_response(completion)

    def label_episode(self, episode) -> tuple[str, ...]:
        """Memory text after each subtask, in order."""
        memory, labels = "", []
        for subtask in episode_subtasks(episode):
            memory = self.write(memory, subtask) or memory
            labels.append(memory)
        return tuple(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label an episode's memory with an LLM.")
    parser.add_argument("--task", default="tidy_up", choices=["find_object", "tidy_up"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompts without loading a model")
    args = parser.parse_args()

    from .oracle import collect

    episode = collect(args.task, 1, seed=args.seed, keep_frames=False)[0]
    subtasks = episode_subtasks(episode)

    if args.dry_run:
        # Thread the template writer's memory through, so the prompts show the
        # shape the LLM actually sees rather than a first turn repeated N times.
        from .text_memory import memory_trace

        # updates[i] is the memory in force *entering* subtask i, which is
        # exactly the "previous memory" the prompt wants.
        prior = [u.text for u in memory_trace(episode).updates]
        for i, subtask in enumerate(subtasks):
            print(build_prompt(prior[i], subtask))
            print("-" * 60)
        return

    writer = LLMMemoryWriter(args.model)
    print(f"  {args.task} seed={args.seed}, {len(subtasks)} subtasks, "
          f"model={args.model} on {writer.device}")
    memory = ""
    for subtask in subtasks:
        memory = writer.write(memory, subtask) or memory
        print(f"    [{subtask.instruction[:30]:30s}] {memory}")


if __name__ == "__main__":
    main()
