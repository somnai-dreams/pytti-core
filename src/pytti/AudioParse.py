import subprocess

import numpy as np
from loguru import logger
from scipy.signal import butter, sosfilt

SAMPLERATE = 44100


class SpectralAudioParser:
    """
    reads a given input file, scans along it and parses the amplitude in selected bands using butterworth bandpass filters.
    the amplitude is normalized into the 0..1 range for easier use in transformation functions.
    """

    def __init__(self, input_audio, offset, frames_per_second, filters):
        if len(filters) < 1:
            raise ValueError(
                "When using input_audio, at least 1 audio filter must be specified"
            )
        for filt in filters:
            _validate_filter(filt)

        pipe = subprocess.Popen(
            # fmt: off
            [
                "ffmpeg", "-i", input_audio,
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", str(SAMPLERATE),
                "-ac", "1",
                "-",
            ],
            # fmt: on
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )

        # read the audio from the pipe in 0.5s blocks (2 bytes per sample)
        chunks = []
        while True:
            buf = pipe.stdout.read(SAMPLERATE)
            chunks.append(np.frombuffer(buf, dtype=np.int16))
            if len(buf) < SAMPLERATE:
                break
        _, stderr = pipe.communicate()
        if pipe.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed to decode {input_audio!r}: "
                f"{stderr.decode(errors='replace')[-500:]}"
            )
        self.audio_samples = np.concatenate(chunks) if chunks else np.array([], np.int16)
        if len(self.audio_samples) == 0:
            raise RuntimeError(f"No audio samples decoded from {input_audio!r}")

        self.duration = len(self.audio_samples) / SAMPLERATE
        logger.debug(
            f"initialized audio file {input_audio}, samples read: "
            f"{len(self.audio_samples)}, total duration: {self.duration}s"
        )
        self.offset = offset
        if offset > self.duration:
            raise ValueError(
                f"Audio offset set at {offset}s but input audio is only "
                f"{self.duration}s long"
            )
        # analyze all samples for the current frame
        self.window_size = int(1 / frames_per_second * SAMPLERATE)
        self.filters = filters

        # parse band maxima first for normalizing the filtered signal to 0..1 at arbitrary points in the file later
        # (band-passing the entire track instead of windows creates maxima that are way off)
        steps = int((self.duration - self.offset) * frames_per_second)
        interval = 1 / frames_per_second
        maxima = {}
        for step in range(steps):
            sample_offset = int(step * interval * SAMPLERATE)
            cur_maxima = bp_filtered(
                self.audio_samples[sample_offset : sample_offset + self.window_size],
                filters,
            )
            for key, value in cur_maxima.items():
                maxima[key] = max(maxima.get(key, value), value)
        self.band_maxima = maxima
        logger.debug(
            f"initialized band maxima for {len(filters)} filters: {self.band_maxima}"
        )

    def get_params(self, t) -> dict[str, float]:
        """
        Return the amplitude parameters at the given point in time t within the audio track, or {} if the track has ended.
        Amplitude/energy parameters are normalized into the [0,1] range.
        """
        sample_offset = int((t + self.offset) * SAMPLERATE)
        logger.debug(f"Analyzing audio at {self.offset + t}s")
        if sample_offset < len(self.audio_samples):
            window_samples = self.audio_samples[
                sample_offset : sample_offset + self.window_size
            ]
            if len(window_samples) < self.window_size:
                logger.debug(
                    f"Audio input ended mid-window at {t + self.offset}s; returning null result"
                )
                return {}
            return bp_filtered_norm(window_samples, self.filters, self.band_maxima)
        else:
            logger.debug("Audio input has ended. Returning null result")
            return {}

    def get_duration(self):
        return self.duration


def _validate_filter(filt):
    nyquist = SAMPLERATE / 2
    if not filt.variable_name:
        raise ValueError(
            "Every audio filter needs a variable_name to expose to expressions"
        )
    if filt.f_center is None or filt.f_width is None:
        raise ValueError(
            f"Audio filter {filt.variable_name!r} needs both f_center and "
            "f_width set (in Hz)"
        )
    lower = filt.f_center - filt.f_width / 2
    upper = filt.f_center + filt.f_width / 2
    if not 0 < lower < upper < nyquist:
        raise ValueError(
            f"Audio filter {filt.variable_name!r} has an invalid band: "
            f"f_center={filt.f_center}, f_width={filt.f_width} gives "
            f"({lower}, {upper}) Hz; needs 0 < low < high < {nyquist}"
        )


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype="bandpass", output="sos")
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


def bp_filtered(window_samples, filters) -> dict[str, float]:
    results = {}
    for filt in filters:
        offset = filt.f_width / 2
        lower = filt.f_center - offset
        upper = filt.f_center + offset
        filtered = butter_bandpass_filter(
            window_samples, lower, upper, SAMPLERATE, order=filt.order
        )
        results[filt.variable_name] = np.max(np.abs(filtered))
    return results


def bp_filtered_norm(window_samples, filters, norm_factors) -> dict[str, float]:
    results = bp_filtered(window_samples, filters)
    for key in results:
        # normalize
        results[key] = results[key] / norm_factors[key]
    return results
