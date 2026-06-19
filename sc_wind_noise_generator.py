"""
Single-Channel Wind Noise Generator

Authors   : Daniele Mirabilii and Emanuël Habets

Reference : D. Mirabilii, A. Lodermeyer, F. Czwielong, S. Becker and E.A P. Habets, 
            Simulating wind noise with airflow speed-dependent characteristics, 
            Proc. of International Workshop on Acoustic Signal Enhancement (IWAENC), 2022.

Copyright (C) 2023 Friedrich-Alexander-Universität Erlangen-Nürnberg, Germany

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
import scipy as sp
from scipy import signal
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
import spectrum
import soundfile as sf
import sounddevice as sd

class WindNoiseGenerator:
    """Wind Noise Generator Class"""

    def __init__(
        self,
        fs=48000,
        duration=5,
        generate=True,
        wind_profile=None,
        gustiness=3,
        short_term_var=True,
        start_seed=None,
        calm_stay_prob=0.92,
        b_par=2,
        a_par=2,
        profile_smooth_s=0.5,
        speed_jitter_std=10.0,
        speed_jitter_smooth_s=0.1,
        calm_gain_fade_s=0.1,
    ):
        """Initizalize object"""

        self.fs = fs
        self.duration = duration
        self.samples = fs * duration
        self.generate = generate
        self.gustiness = gustiness
        self.wind_profile = wind_profile
        self.short_term_var = short_term_var
        self.calm_stay_prob = calm_stay_prob
        self.b_par = b_par
        self.a_par = a_par
        self.profile_smooth_s = profile_smooth_s
        self.speed_jitter_std = speed_jitter_std
        self.speed_jitter_smooth_s = speed_jitter_smooth_s
        self.calm_gain_fade_s = calm_gain_fade_s
        self.calm_mask = None
        if start_seed is not None:
            np.random.seed(start_seed)

    def generate_wind_noise(self):
        """Generate single-channel wind noise by filtering excitation signal"""

        if self.generate:
            wind_profile = self._generate_wind_speed_profile()
        else:
            wind_profile = self._import_wind_speed_profile()

        exc = self.generate_excitation_signal(wind_profile)
        exc_filtered = self._filter(exc, wind_profile, 2048)

        if self.calm_mask is not None and np.any(self.calm_mask):
            exc_filtered *= self._build_calm_gain(self.calm_mask)

        # TODO: scale to match the actual wind noise level on the Zylia
        exc_filtered = 97335 * exc_filtered

        return exc_filtered, wind_profile

    @classmethod
    def generate_multi_mic(
        cls,
        n_mics,
        fs=48000,
        duration=5,
        wind_profile=None,
        gustiness=3,
        short_term_var=True,
        scene_seed=None,
        mic_seeds=None,
        calm_stay_prob=0.92,
        b_par=2,
        a_par=2,
        profile_smooth_s=0.5,
        speed_jitter_std=10.0,
        speed_jitter_smooth_s=0.1,
        calm_gain_fade_s=0.1,
    ):
        """
        Generate uncorrelated wind noise at multiple mics sharing one wind profile.

        A single wind-speed profile is generated (or taken from ``wind_profile``),
        then each mic is synthesised independently with its own random excitation
        while seeing the same environmental wind conditions.

        Parameters
        ----------
        n_mics : int
            Number of microphone channels to generate.
        wind_profile : array-like, optional
            Pre-computed wind speed profile. When omitted, one is generated from
            ``gustiness`` and the other scene parameters.
        scene_seed : int, optional
            Seed for shared profile generation. Ignored when ``wind_profile`` is
            given.
        mic_seeds : sequence of int, optional
            Per-mic seeds for independent synthesis. Length must equal
            ``n_mics``. When omitted and ``scene_seed`` is set, uses
            ``scene_seed + 1``, ``scene_seed + 2``, ... Otherwise each mic
            uses the global RNG without re-seeding.

        Returns
        -------
        audio : ndarray, shape (n_mics, n_samples)
        wind_profile : ndarray, shape (n_samples,)
        """
        if n_mics < 1:
            raise ValueError('n_mics must be at least 1')

        common_kwargs = dict(
            fs=fs,
            duration=duration,
            short_term_var=short_term_var,
            calm_stay_prob=calm_stay_prob,
            b_par=b_par,
            a_par=a_par,
            profile_smooth_s=profile_smooth_s,
            speed_jitter_smooth_s=speed_jitter_smooth_s,
            calm_gain_fade_s=calm_gain_fade_s,
        )

        if wind_profile is None:
            scene = cls(
                generate=True,
                gustiness=gustiness,
                speed_jitter_std=speed_jitter_std,
                start_seed=scene_seed,
                **common_kwargs,
            )
            wind_profile = scene._generate_wind_speed_profile()
        else:
            wind_profile = np.asarray(wind_profile, dtype=float)

        if mic_seeds is None:
            mic_seeds = (
                [scene_seed + i + 1 for i in range(n_mics)]
                if scene_seed is not None
                else [None] * n_mics
            )
        elif len(mic_seeds) != n_mics:
            raise ValueError(f'mic_seeds must have length {n_mics}')

        n_samples = int(fs * duration)
        audio = np.empty((n_mics, n_samples), dtype=float)
        for mic_idx, mic_seed in enumerate(mic_seeds):
            mic = cls(
                generate=False,
                wind_profile=wind_profile,
                speed_jitter_std=0.0,
                start_seed=mic_seed,
                **common_kwargs,
            )
            audio[mic_idx], _ = mic.generate_wind_noise()

        return audio, wind_profile

    def generate_excitation_signal(self, wind_profile):
        """Generate excitation signal"""

        window_size = 128
        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        wgn = np.concatenate(
            (np.zeros(window_size), np.random.randn(self.samples), np.zeros(window_size)))
        wgn_length = len(wgn)

        lt_var = self._generate_long_term_variance(wind_profile)
        lt_var = np.concatenate((np.zeros(window_size), lt_var, np.zeros(window_size)))

        st_var = self._generate_short_term_variance_garch(wind_profile)
        cond_var = np.abs(st_var)

        num_windows = (wgn_length - window_size) // hops + 1
        exc = np.zeros(wgn_length)

        for time_frame in range(num_windows-1):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            gain_ltst = lt_var[idx]
            if self.short_term_var:
                gain_ltst *= np.sqrt(cond_var[time_frame])
            noise_seg_ltst = gain_ltst * wgn[idx] * hann_window
            exc[idx] += noise_seg_ltst

        exc = exc[window_size:-window_size]

        return exc

    def _generate_short_term_variance_garch(self, wind_profile):
        """Generate short-term variance of GARCH process"""

        window_size = 128
        hops = window_size // 2 # overlap

        profile = np.concatenate(
            (2 * np.ones(window_size), wind_profile, 2 * np.ones(window_size)))
        profile_length = len(profile)

        num_windows = (profile_length - window_size) // hops + 1
        st_var = np.zeros(num_windows)
        cond_var = np.zeros(num_windows)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 0, 18)
            alpha, beta, omega = self._speed2par(speed)

            if alpha + beta > 1:
                beta = 0

            cond_var[time_frame] = omega + alpha * \
                st_var[time_frame-1]**2 + beta*(cond_var[time_frame-1])
            st_var[time_frame] = np.sqrt(np.abs(cond_var[time_frame])) * \
                np.random.randn()

        return st_var/max(np.abs(st_var))

    def _generate_long_term_variance(self, wind_profile):
        """Generate long-term variance"""

        # Regression parameter noise variance/wind speed
        regression_coeff = np.array([8.00071114414022, -220.332082908370])

        # Long-term noise variance based on wind speed profile in dB scale
        variance_profile_db = np.polyval(regression_coeff, wind_profile)

        # Long-term noise variance in linear scale
        variance_profile = 10 ** (variance_profile_db / 10)
        var_lt = np.sqrt(np.abs(variance_profile))  # long-term gain

        return var_lt

    @staticmethod
    def _sample_calm_mask(n, calm_stay_prob=0.92, start_calm=False):
        """Return True for long-term segments where wind speed should be zero."""

        mask = np.zeros(n, dtype=bool)
        calm = start_calm
        for i in range(n):
            mask[i] = calm
            if calm:
                calm = np.random.rand() < calm_stay_prob
            else:
                calm = np.random.rand() < (1 - calm_stay_prob)
        return mask

    def _speed_points(self):
        """Number of long-term speed segments scaled with duration."""

        return max(1, int(self.gustiness / 5 * self.duration))

    def _interp_lt_to_samples(self, lt_values):
        """Hold/interpolate long-term segment values across the audio duration."""

        lt_values = np.asarray(lt_values, dtype=float)
        return np.interp(
            np.linspace(0, len(lt_values) - 1, self.samples),
            np.arange(len(lt_values)),
            lt_values,
        )

    def _moving_average(self, signal_in, window_s):
        """Smooth a profile with a normalized moving-average window."""

        window_size = max(1, int(window_s * self.fs))
        return uniform_filter1d(signal_in, size=window_size, mode='nearest')

    def _generate_speed_fluctuations(self):
        """Generate smoothed additive speed fluctuations."""

        fluctuations = self.speed_jitter_std * np.random.randn(self.samples)
        smooth_s = self.speed_jitter_smooth_s
        if smooth_s <= 0:
            return fluctuations

        hann_window = np.hanning(max(1, int(self.fs * smooth_s)))
        hann_window /= np.sum(hann_window)
        return signal.fftconvolve(fluctuations, hann_window, mode='same')

    def _build_calm_gain(self, calm_mask):
        """Build a per-sample gain that silences calm regions after LPC filtering."""

        calm_gain = 1.0 - calm_mask.astype(float)
        if self.calm_gain_fade_s <= 0:
            return calm_gain

        fade_samples = max(1, int(self.calm_gain_fade_s * self.fs))
        hann_window = np.hanning(fade_samples)
        hann_window /= np.sum(hann_window)
        return signal.fftconvolve(calm_gain, hann_window, mode='same')

    def _finalize_wind_speed_profile(self, wind_speed_profile_lt, calm_mask_lt=None):
        """Upsample, smooth, and add jitter to a long-term wind speed profile."""

        if calm_mask_lt is None:
            calm_mask_lt = wind_speed_profile_lt <= 0

        calm_mask_lt = np.asarray(calm_mask_lt, dtype=bool)
        wind_speed_profile = self._interp_lt_to_samples(wind_speed_profile_lt)
        calm_mask = self._interp_lt_to_samples(calm_mask_lt.astype(float)) >= 0.5

        if self.profile_smooth_s > 0:
            wind_speed_profile = self._moving_average(
                wind_speed_profile, self.profile_smooth_s)

        if self.speed_jitter_std > 0:
            fluctuations = self._generate_speed_fluctuations()
            windy_mask = ~calm_mask
            wind_speed_profile[windy_mask] += fluctuations[windy_mask]

        wind_speed_profile = np.clip(wind_speed_profile, 0, None)
        self.calm_mask = calm_mask
        return wind_speed_profile

    def _generate_wind_speed_profile(self, b_par=None, a_par=None):
        """Generate the wind speed profile with optional calm periods."""

        b_par = self.b_par if b_par is None else b_par
        a_par = self.a_par if a_par is None else a_par
        speed_points = self._speed_points()

        if self.calm_stay_prob is not None:
            calm_mask_lt = self._sample_calm_mask(speed_points, self.calm_stay_prob)
        else:
            calm_mask_lt = np.zeros(speed_points, dtype=bool)

        windy_speeds = b_par * np.random.weibull(a_par, speed_points)
        wind_speed_profile_lt = np.where(calm_mask_lt, 0.0, windy_speeds)

        return self._finalize_wind_speed_profile(wind_speed_profile_lt, calm_mask_lt)

    def _import_wind_speed_profile(self):
        """Read the wind speed profile from input"""

        wind_speed_profile_lt = np.asarray(self.wind_profile, dtype=float)
        calm_mask_lt = wind_speed_profile_lt <= 0
        return self._finalize_wind_speed_profile(wind_speed_profile_lt, calm_mask_lt)

    def _filter(self, exc, wind_profile, window_size):
        """Filter the excitation signals with the AR filter coefficients"""

        hops = window_size // 2  # overlap
        hann_window = np.hanning(window_size)  # hanning window

        profile = np.concatenate(
            (np.zeros(window_size), wind_profile, np.zeros(window_size)))

        exc = np.concatenate((np.zeros(window_size), exc, np.zeros(window_size)))
        exc_length = len(exc)

        # Overlap-add approach for the time-varying filtering of the excitation signal
        num_windows = (exc_length - window_size) // hops + 1
        exc_filtered = np.zeros(exc_length)

        for time_frame in range(num_windows):
            start_idx = time_frame * hops
            end_idx = start_idx + window_size
            idx = np.arange(start_idx, end_idx)

            speed = np.clip(np.mean(profile[idx]), 2, 18)
            lpc = self._lsf2lpc(speed)

            exc_seg = exc[idx] * hann_window
            exc_seg_filtered = sp.signal.lfilter(
                np.array([1.0]), lpc, exc_seg)

            exc_filtered[idx] += exc_seg_filtered

        exc_filtered = exc_filtered[window_size:-window_size]

        return exc_filtered

    def _speed2par(self, speed):
        """Convert speed to GARCH parameters"""

        gp_alpha = np.array([-2.73244444508231e-05, 0.00141129711949206, -
                            0.0274652794467908, 0.257613241095714, -0.139824587447063])
        gp_beta = np.array(
            [-9.75160902595897e-05, 0.00464300106846736, -0.0871968755558256, 0.651013973757802])
        gp_omega = np.array(
            [9.69585296574741e-05, -0.00231853830578967, 0.0124681159197788])

        alpha = np.polyval(gp_alpha, speed)
        beta = np.polyval(gp_beta, speed)
        omega = np.polyval(gp_omega, speed)

        return alpha, beta, omega

    def _lsf2lpc(self, speed):
        """Generate LPC coefficients from the LSF-speed models given a speed value"""

        # Regression coefficients of the LFS-speed model
        # The n-th LFS coefficient corresponds to the n-th column
        regression_coeff = np.array([[-2.63412497797108e-06, 5.93162248595821e-05,
                                      0.000215613938043173, -0.000149723789407121,
                                      -0.000213703084399375],
                                     [9.50240139044154e-05,	-0.00271741166649528,
                                      -0.0103783584000284, 0.00483963669507075,
                                      0.00931864887930701],
                                     [-0.000699199223507821, 0.0428714179385289,
                                      0.177250839818556, -0.0329542145779793,
                                      -0.129910107562929],
                                     [0.0106849674771013, -0.234688122194936,
                                      -1.21337646113093, -0.168053225019258,
                                      0.568371362156217],
                                     [-0.000966851130291645, 0.541693139684727,
                                      3.24796925730457, 2.54984352038733,
                                      1.86097523205089]])
        order = 5

        # Estimate LFS based on the speed value
        lfs_estimated = np.zeros(order)

        for order_idx in range(order):
            lfs_estimated[order_idx] = np.polyval(regression_coeff[:, order_idx], speed)

        # Convert LFS into LPC coefficients
        lpc_a = spectrum.lsf2poly(lfs_estimated)

        return lpc_a

    def plot_signals(self, wns, wind_profile):
        """
        Plot the generated wind noise signals and the associated wind profile.

        Parameters
        ----------
        wns : ndarray
            Single-mic waveform ``(n_samples,)`` or multi-mic ``(n_mics, n_samples)``.
        wind_profile : ndarray
            Wind speed profile in m/s.

        Example:
         wn = WindNoiseGenerator(fs=16000, duration=10)
         wn_sample, wind_profile = wn.generate_wind_noise()
         wn.plot_signals(wn_sample, wind_profile)

         audio, wind_profile = WindNoiseGenerator.generate_multi_mic(2, fs=16000, duration=10)
         wn.plot_signals(audio, wind_profile)
        """
        wns = np.asarray(wns, dtype=float)
        if wns.ndim == 1:
            wns = wns[np.newaxis, :]
        elif wns.ndim != 2:
            raise ValueError('wns must be 1-D (n_samples,) or 2-D (n_mics, n_samples)')

        n_mics = wns.shape[0]
        n_samples = min(wns.shape[1], int(self.duration * self.fs))
        time_ind = np.arange(n_samples) / self.fs
        wns = wns[:, :n_samples]
        wind_profile = np.asarray(wind_profile, dtype=float)[:n_samples]

        peak = np.max(np.abs(wns))
        amp_ylim = (-peak * 1.05, peak * 1.05) if peak > 0 else (-1, 1)
        spec_vmin = 20 * np.log10(peak) - 120 if peak > 0 else -120
        freq_formatter = plt.FuncFormatter(lambda x, pos: f"{x/1e3:g}")

        if n_mics == 1:
            fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)
            axs[0].plot(time_ind, wns[0])
            axs[0].set_ylabel('Amplitude')
            axs[0].grid(True)
            axs[0].autoscale(enable=True, axis='x', tight=True)
            axs[0].set_ylim(amp_ylim)

            axs[1].specgram(
                wns[0], Fs=self.fs, NFFT=512, noverlap=128,
                mode='magnitude', scale='dB', cmap='inferno',
                vmin=spec_vmin,
            )
            axs[1].autoscale(enable=True, axis='x', tight=True)
            axs[1].set_ylabel('Frequency [Hz]')
            axs[1].set_ylim(0, 8000)
            axs[1].yaxis.set_major_formatter(freq_formatter)

            axs[2].plot(time_ind, np.abs(wind_profile), label='Wind Profile')
            axs[2].set_ylim(0, 6)
            axs[2].set_ylabel('Wind speed [m/s]')
            axs[2].grid(True)
            axs[2].axis('tight')
        else:
            fig = plt.figure(constrained_layout=True)
            gs = fig.add_gridspec(3, n_mics, height_ratios=[2, 2, 1])

            ax_wave = fig.add_subplot(gs[0, :])
            for mic_idx in range(n_mics):
                ax_wave.plot(time_ind, wns[mic_idx], label=f'Mic {mic_idx + 1}')
            ax_wave.set_ylabel('Amplitude')
            ax_wave.set_ylim(amp_ylim)
            ax_wave.grid(True)
            ax_wave.legend(loc='upper right')

            for mic_idx in range(n_mics):
                ax_spec = fig.add_subplot(gs[1, mic_idx])
                ax_spec.specgram(
                    wns[mic_idx], Fs=self.fs, NFFT=512, noverlap=128,
                    mode='magnitude', scale='dB', cmap='inferno',
                    vmin=spec_vmin,
                )
                ax_spec.set_title(f'Mic {mic_idx + 1}')
                ax_spec.set_ylabel('Frequency [Hz]')
                ax_spec.set_ylim(0, 8000)
                ax_spec.yaxis.set_major_formatter(freq_formatter)

            ax_profile = fig.add_subplot(gs[2, :])
            ax_profile.plot(time_ind, np.abs(wind_profile), label='Wind Profile')
            ax_profile.set_xlabel('Time [s]')
            ax_profile.set_ylim(0, 6)
            ax_profile.set_ylabel('Wind speed [m/s]')
            ax_profile.grid(True)

        if n_mics == 1:
            fig.tight_layout()
        plt.show()
