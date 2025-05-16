# Local dependencies

## open_clip

We used `open_clip` to train our CLIP models. During our work, we made some simple adjustments to the `open_clip` code but ultimately did not really make use of the adjusted functionality (except for some added model configs). So in theory, our code should also work with the official `open_clip==2.24.0`. However, to allow full reproducibility, we include our altered version in this repository.

## sparse_autoencoder

We adapted our SAE training code from https://github.com/neuroexplicit-saar/discover-then-name. Thus, we also included the `sparse_autoencoder` package which Rao et al. used in their work.
