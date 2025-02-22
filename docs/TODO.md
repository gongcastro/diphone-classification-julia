## Audio and data processing

- [ ] Better spectrograms (actual Mel-spectrograms). So far spectrograms are being generated using the DSP.jl package, which does not seem to allow much flexibility. Look for custom scripts.
- [ ] Achieve better control over the number of channels ("frequency bins"). Figure out what the `n` argument in `DSP.spectrogram()` stands for.
- [ ] Allow for spectrograms with variable number of time steps. This is proving somewhat tricky because Flux.jl seems to expect inputs of identical dimensions. For now, spectrograms are being padded to fit maximum size.

## Model

- [ ] Better accuracy measurements, F1, recall, etc.
- [ ] Analyse test accuracy more systematically (CV vs. VS, timecourse, etc.)
- [ ] More efficient logging of model evaluations (loss, accuracy)
- [ ] Implement an encoder-decoder structure
- [ ] Analyse model embeddings
