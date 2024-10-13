from typing import Union

import librosa
import numpy as np
from decomposed_waveform import DecomposedWaveform


def get_energy_librosa(waveform, length, *, hop_size, win_size, domain="db"):
    """
    Definition of energy: RMS of the waveform, in dB representation
    :param waveform: [T]
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param win_size: Window size, in number of samples
    :param domain: db or amplitude
    :return: energy
    """
    energy = librosa.feature.rms(
        y=waveform, frame_length=win_size, hop_length=hop_size
    )[0]
    if len(energy) < length:
        energy = np.pad(energy, (0, length - len(energy)))
    energy = energy[:length]
    if domain == "db":
        energy = librosa.amplitude_to_db(energy)
    elif domain == "amplitude":
        pass
    else:
        raise ValueError(f"Invalid domain: {domain}")
    return energy


def get_tension_base_harmonic(
    waveform: Union[np.ndarray, DecomposedWaveform],
    samplerate,
    f0,
    length,
    *,
    hop_size=None,
    fft_size=None,
    win_size=None,
    domain="logit",
):
    """
    Definition of tension: radio of the real harmonic part (harmonic part except the base harmonic)
    to the full harmonic part.
    :param waveform: All other analysis parameters will not take effect if a DeconstructedWaveform is given
    :param samplerate: sampling rate
    :param f0: reference f0
    :param length: Expected number of frames
    :param hop_size: Frame width, in number of samples
    :param fft_size: Number of fft bins
    :param win_size: Window size, in number of samples
    :param domain: The domain of the final ratio representation.
     Can be 'ratio' (the raw ratio), 'db' (log decibel) or 'logit' (the reverse function of sigmoid)
    :return: tension
    """
    if not isinstance(waveform, DecomposedWaveform):
        waveform = DecomposedWaveform(
            waveform=waveform,
            samplerate=samplerate,
            f0=f0,
            hop_size=hop_size,
            fft_size=fft_size,
            win_size=win_size,
        )
    waveform_h = waveform.harmonic()
    waveform_base_h = waveform.harmonic(0)
    energy_base_h = get_energy_librosa(
        waveform_base_h,
        length,
        hop_size=waveform.hop_size,
        win_size=waveform.win_size,
        domain="amplitude",
    )
    energy_h = get_energy_librosa(
        waveform_h,
        length,
        hop_size=waveform.hop_size,
        win_size=waveform.win_size,
        domain="amplitude",
    )
    tension = np.sqrt(
        np.clip(energy_h**2 - energy_base_h**2, a_min=0, a_max=None)
    ) / (energy_h + 1e-5)
    if domain == "ratio":
        tension = np.clip(tension, a_min=0, a_max=1)
    elif domain == "db":
        tension = np.clip(tension, a_min=1e-5, a_max=1)
        tension = librosa.amplitude_to_db(tension)
    elif domain == "logit":
        tension = np.clip(tension, a_min=1e-4, a_max=1 - 1e-4)
        tension = np.log(tension / (1 - tension))
    return tension
