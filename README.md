# Food-101 · ResNet-18 transfer learning

Fine-tuning ResNet-18 on [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) —
101 food classes, 101,000 photos — to **75.98% top-1 / 93.49% top-5** on the
official held-out test set, in 16 minutes on a laptop GPU.

Comes with a web page you can drop a photo into, so you can see what it actually
does rather than take the numbers on faith.

> Looking for the bigger model? [food101-resnet50-transfer-learning](https://github.com/Adnan5000/food101-resnet50-transfer-learning)
> reaches 83.46% top-1 with the same recipe. This one is ~4x faster to train and
> less than half the size on disk.

## Results

Measured on Food-101's **official** 25,250-image test split — the standard one,
so these are comparable to published numbers. The model never trained on any of it.

| Metric | Value |
|---|---|
| Top-1 accuracy | **75.98%** |
| Top-5 accuracy | **93.49%** |
| Trainable parameters | 11,228,325 |
| Training time | 15.6 min (5 epochs, RTX 3060 Laptop, 6GB) |
| Throughput | ~570 img/s at batch 64 |
| Checkpoint size | 42.9 MB |

Per epoch:

| Epoch | train loss | train acc | **val top-1** | val top-5 |
|---:|---:|---:|---:|---:|
| 1 | 3.249 | 34.5% | 62.97% | 86.89% |
| 2 | 2.195 | 60.0% | 69.70% | 90.63% |
| 3 | 1.998 | 65.9% | 73.14% | 92.18% |
| 4 | 1.882 | 69.6% | 75.14% | 93.11% |
| 5 | 1.803 | 72.2% | **75.98%** | **93.49%** |

Still improving at epoch 5 — more epochs would add a little, not a lot.
Raw numbers in [`training_history.json`](training_history.json).

**Top-1 vs top-5:** top-1 is "the single best guess was right." Top-5 is "the
right answer was in the first five." Top-5 is always higher and it is not the
headline number, though it is often quoted as one.

## Try it on your own photos

You need Python 3.12+ and about 1.5 GB of disk for PyTorch. No GPU required —
inference runs fine on CPU at ~40 ms per image.

```bash
git clone https://github.com/Adnan5000/food101-resnet18-transfer-learning
cd food101-resnet18-transfer-learning

python -m venv .venv
# Windows:  .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

Download `resnet18_food101.pt` from
[**Releases**](https://github.com/Adnan5000/food101-resnet18-transfer-learning/releases)
into a `weights/` folder (it is 42.9 MB, kept out of the repo so clones stay fast):

```bash
mkdir -p weights
# put resnet18_food101.pt in weights/
uvicorn serve:app --port 8001
```

Open <http://localhost:8001>, drag in a photo of food, and you get the top five
guesses with confidences and the inference time.

There is also a plain HTTP endpoint:

```bash
curl -X POST "http://localhost:8001/v1/classify-upload?top_k=5" -F "file=@dinner.jpg"
```

```json
{
  "predictions": [
    { "class": "pancakes",            "confidence": 0.6013 },
    { "class": "spaghetti_bolognese", "confidence": 0.0455 },
    { "class": "creme_brulee",        "confidence": 0.0282 }
  ],
  "inference_ms": 36
}
```

## Training it yourself

```bash
pip install -r requirements-train.txt
python train.py --cache-dir ./data --epochs 5
```

First run downloads Food-101 (~5 GB) and builds a ~10 GB cache. On a 6 GB GPU
the defaults fit comfortably; drop `--batch-size` if you have less.

## What actually moved the number

Starting point was a frozen backbone — ImageNet weights held fixed, training only
the 101-way classifier on top. That plateaus at **~44%** no matter how long you
run it. Two changes account for almost all the gain:

**1. Unfreezing the backbone.** A frozen network is a linear probe over features
built for ImageNet's 1000 classes. Food-101 is fine-grained — the model has to
separate red velvet cake from carrot cake — and that needs the convolutional
filters themselves to adapt. Training the full network means 11,228,325
parameters instead of 51,813: **217× more**.

**2. Using the whole dataset.** The frozen run trained on 100 images per class.
Food-101 ships 750 per class, so 89% was being thrown away. 8,080 → 75,750 images.

Smaller contributions: SGD with Nesterov momentum and a OneCycle schedule instead
of constant-LR Adam; the backbone at a 10× lower learning rate than the
randomly-initialised head, since pretrained features need gentler updates;
label smoothing 0.1; weight decay 1e-4; augmentation with a 0.5 crop-scale floor
and colour jitter; mixed precision for speed.

Note the final epoch shows train accuracy (72.2%) *below* validation (75.98%).
That is not a bug — training images get deliberately distorted while validation
images are shown clean, and the training figure is a running average over an
epoch that started much worse.

## Two things that would have been silent bugs

**Class ordering.** The HuggingFace mirror of Food-101 orders its labels
differently from the canonical `classes.txt` — `cheesecake` before
`cheese_plate`. Trusting the index would have transposed those two classes with
no error anywhere. `train.py` asserts the sets match and remaps the indices, and
the checkpoint stores its own label list so inference cannot pair weights with
the wrong order.

**The download host.** `torchvision.datasets.Food101` points at
data.vision.ee.ethz.ch, which measured **0.17 MB/s** during development — about
seven hours for 4.65 GB. The HuggingFace mirror served the same data at 3.4 MB/s.
Same images, same official split.

## Honest limits

- **Food-101 only.** 101 classes. Anything else gets confidently mapped to the
  nearest of those 101 — there is no "not food" output.
- **Confident errors are the real failure mode.** Red velvet cake comes back as
  carrot cake at 0.70. A near-tie is recoverable by showing the user alternatives;
  a confident mistake gives the interface no signal at all. Accuracy does not
  capture this, and it matters more than the headline number for anything that
  acts on the output.
- **Single centre crop at inference.** Test-time augmentation would add a point
  or so at several times the cost.
- Trained on one 6 GB laptop GPU for 16 minutes. Not a leaderboard attempt.

## Licence

Code is MIT. The Food-101 dataset and weights derived from it are ETH Zurich's
and published for research use — see [LICENSE](LICENSE).
