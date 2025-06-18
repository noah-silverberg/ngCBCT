import logging
import importlib


def run_app(app: str, args: list):
    """Run a training or other application via dynamic import and its main()."""
    # Prepend default args
    argv = []
    argv.extend(args)
    logging.info(f"RUNNING: {app} {argv}\n")
    module_name, class_name = app.rsplit(".", 1)
    module = importlib.import_module("pipeline." + module_name)
    cls = getattr(module, class_name)
    instance = cls(argv)
    instance.main()
    logging.info(f"\nFINISHED: {app}")
