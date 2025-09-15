# SST-GUI

This repo is a GUI (via [Streamlit](https://streamlit.io/)) for the idea presented in [SST](https://arxiv.org/abs/2501.06749).

1. Run `uv run streamlit run gui.py`. This starts a Streamlit app (which you can open in a browser) to specify where your images are located, speciy how to group images, and pick out reference frames and the corresponding masks. All of this data will be saved in a `spec.toml` file.
2. Run `uv run inference.py --spec spec.toml`. The `spec.toml` file will include all of the decisions you made with the GUI, and will then generate masks for the rest of the images using SAM 2 and the reference masks you generated with the GUI.

## Design

