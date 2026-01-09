- Use `uv run SCRIPT.py` or `uv run python ARGS` to run python instead of plain `python`.
- After making edits, run `uvx ruff format --preview .` to format the file, then run `uvx ruff check --fix .` to lint, then run `uvx ty check FILEPATH` to type check (`ty` is prerelease software, and typechecking often will have false positives). Only do this if you think you're finished, or if you can't figure out a bug. Maybe linting will make it obvious. Don't fix linting or typing errors in files you haven't modified.
- Don't hard-wrap comments. Only use linebreaks for new paragraphs. Let the editor soft wrap content.
- Prefer negative if statements in combination with early returns/continues. Rather than nesting multiple positive if statements, just check if a condition is False, then return/continue in a loop. This reduces indentation.
- This project uses Python 3.12. You can use `dict`, `list`, `tuple` instead of the imports from `typing`. You can use `| None` instead of `Optional`.
- File descriptors from `open()` are called `fd`.
- Use types where possible, including `jaxtyping` hints.
- Decorate functions with `beartype.beartype` unless they use a `jaxtyping` hint, in which case use `jaxtyped(typechecker=beartype.beartype)`.
- Variables referring to a absolute filepath should be suffixed with `_fpath`. Filenames are `_fname`. Directories are `_dpath`.
- submitit and jaxtyping don't work in the same file. See [this issue](https://github.com/patrick-kidger/jaxtyping/issues/332). To solve this, all jaxtyped functions/classes need to be in a different file to the submitit launcher script.

# Inference

- `inference.py` has resume logic - it skips masks that already exist in `pred_masks/`. Don't delete pred_masks when restarting; it will continue from where it left off.

# Test Data

- CSV file for testing: `/local/scratch/stevens.994/datasets/cambridge-segmented/master.csv`
