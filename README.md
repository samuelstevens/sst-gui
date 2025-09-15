# SST-GUI

This repo is a GUI (via [Streamlit](https://streamlit.io/)) for the idea presented in [SST](https://arxiv.org/abs/2501.06749).

1. Run `uv run streamlit run gui.py`. This starts a Streamlit app (which you can open in a browser) to specify where your images are located, speciy how to group images, and pick out reference frames and the corresponding masks. All of this data will be saved in a `spec.toml` file.
2. Run `uv run inference.py --spec spec.toml`. The `spec.toml` file will include all of the decisions you made with the GUI, and will then generate masks for the rest of the images using SAM 2 and the reference masks you generated with the GUI.

# GUI

First, upload metadata and tell the system how to get your images.

![Upload a CSV file.](docs/assets/metadata.png)

![Configure how to read and group images](docs/assets/configure.png)

Then wait for SAM 2 to get all masks (with no prompt) from 5 images.

![All masks in an image](docs/assets/all-masks.png)

Then filter the masks to the objects you care about and label the objects with integer ids.

![Filtered masks in an image](docs/assets/filtered-masks.png)
