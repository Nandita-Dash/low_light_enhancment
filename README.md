# Low-light Image Enhancement using Autoencoder and Histogram Matching

Web app for project A042. Upload a dark image and get an enhanced one, with a before/after slider,
histogram comparison, and PSNR/SSIM when a ground truth image is provided.

## Examples
Real low-light captures from the LOL test set (left) and the enhanced result (right), demo mode at the default
settings, scored against the normal-exposure shot of the same scene:

| | |
|---|---|
| ![Kitchen](static/examples/kitchen_compare.jpg) | **Kitchen**<br>PSNR 5.1 → 17.9 dB<br>SSIM 0.20 → 0.70 |
| ![Window](static/examples/window_compare.jpg) | **Window**<br>PSNR 5.6 → 17.6 dB<br>SSIM 0.19 → 0.67 |
| ![Bowling alley](static/examples/bowling_compare.jpg) | **Bowling alley**<br>PSNR 6.8 → 18.9 dB<br>SSIM 0.19 → 0.67 |

These are three of the better results. Over all 15 test pairs the average is PSNR 7.8 → 15.9 dB and SSIM
0.19 → 0.60. On the darkest captures demo mode loses most of the colour and smooths heavy noise into flat
patches; that is the job of the trained autoencoder, which demo mode only stands in for.

In the app, click one under the image box: it loads the low-light capture with its normal-exposure shot as ground
truth, so PSNR and SSIM appear. Rebuild them with `python make_examples.py --data path/to/LOL`.
Images: LOL dataset, Wei et al., "Deep Retinex Decomposition for Low-Light Enhancement", BMVC 2018, used for
research and education.

## Pipeline
1. Resize (max side 768 px) and run the convolutional autoencoder (`model.py`) for brightening and denoising.
2. Histogram matching (`enhance.py`) per RGB channel against a reference distribution:
   a reference image you upload, or the average of the LOL high-light images (`weights/reference_cdf.npy`).
3. Metrics and histograms returned to the browser.

## Setup
```
pip install -r requirements.txt
```

## Train (needs the LOL dataset)
```
python train.py --data path/to/LOL --epochs 100
```
Training uses random 256 px crops at native resolution (the same scale the app runs at) and validates on full images.
This writes `weights/autoencoder.pt` and `weights/reference_cdf.npy`. A GPU is recommended.

## Run
```
python app.py
```
Open http://127.0.0.1:5000. Set `HOST`, `PORT` and `FLASK_DEBUG=1` (local development only) through environment variables. Without trained weights the app runs in demo mode so you can still try the interface
and histogram matching. The demo first stage works in LAB: noise-adaptive denoising (strength set from the measured
noise level), a lightness lift to the chosen brightness with automatic black and white points, a colour boost that
keeps up with the lift, and colour-cast correction.

Controls: histogram matching on/off, matching strength, and target brightness (0.3-0.8, default 0.6).
With the default reference, matching is applied to lightness only, with a capped contrast gain, and never darkens.
The mean brightness it aims for is restored with a gain anchored at the darkest level, so blacks stay black.
The default reference (learned or built in) is shifted to the target brightness, so the brightness control also works with
trained weights. With trained weights it has no effect when matching is off or a reference image is uploaded, and the
slider is greyed out in those cases.

## Tests
```
pip install pytest
pytest
```

## Files
- `app.py` Flask server and API
- `enhance.py` pipeline, histogram matching, metrics
- `model.py` autoencoder
- `train.py` training on LOL
- `templates/index.html` frontend
- `make_examples.py` builds the example images in `static/examples/` from the LOL test set
