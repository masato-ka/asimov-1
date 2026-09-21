"""Entry points that register the Asimov-1 tasks before delegating to mjlab.

mjlab's own ``train`` / ``play`` commands only auto-import the tasks bundled with mjlab,
so out-of-tree tasks must be imported (which registers them) first.

  uv run asimov-train Asimov-Velocity-Flat --env.scene.num-envs 2048
  uv run asimov-play  Asimov-Velocity-Flat --checkpoint-file <path>
"""


def main() -> None:
  import mjlab_asimov  # noqa: F401  (registers the tasks)
  from mjlab.scripts.train import main as train_main

  train_main()


def play() -> None:
  import mjlab_asimov  # noqa: F401  (registers the tasks)
  from mjlab.scripts.play import main as play_main

  play_main()
