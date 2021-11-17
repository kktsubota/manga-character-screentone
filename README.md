# Manga Character Screentone Synthesis
Official PyTorch implementation of "Synthesis of Screentone Patterns of Manga Characters" presented in IEEE ISM 2019.
I only provide a demo script now.

## Environment

```bash
pip install requirements.txt -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

## Usage
### Dataset Preparation
1. Crop manga character images along with the bounding boxes of "body" annotation in Manga109.

2. Extract line drawings from manga character images using [theano implementation](https://github.com/ljsabc/MangaLineExtraction).
You can also use [official PyTorch implementation](https://github.com/ljsabc/MangaLineExtraction_PyTorch).
(I used the theano implementation for our experiments in our paper. I checked that the PyTorch implementation also works well.)

### Inference

```bash
# download a pre-trained model
wget https://github.com/kktsubota/manga-character-screentone/releases/download/pre/model.pth

# apply a screentone generator
python apply_gen.py <path to a line-drawing image> --model_path model.pth

# render a manga image
python render.py <path to a line-drawing image> label.png
```

## Contact
Please contact me via e-mail if you have any troubles when running this code. My e-mail address is shown in our paper.

## Acknowledgements
* I thank to Chengze Li for providing a code that renders screentones.
* Some parts of our code are borrowed from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.
