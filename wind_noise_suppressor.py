from scipy.signal import butter, filtfilt
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d


class WindNoiseSuppressor:
    def __init__(self, fs=16000, frame_s=0.128, overlap=0.75, alpha=0.5, d_m=2e-2, theta_s=0.0, c_M=343.0, G_min=0.05, smooth_size=5):
        self.fs = fs
        self.frame_s = frame_s
        self.overlap = overlap
        self.alpha = alpha
        self.d_m = d_m
        self.theta_s = theta_s
        self.c_M = c_M
        self.G_min = G_min
        self.smooth_size = smooth_size

        self.nperseg = int(frame_s * fs)
        self.noverlap = int(overlap * self.nperseg)

    def short_term_psd(self, stft_a, stft_b, alpha=0.5, mode='sum'):
        """
        Recursively smoothed short-term PSD (Nelke eq. 4.34-4.35).

        Phi(l, k) = alpha * Phi_hat(l-1, k) + (1 - alpha) * |X(l,k) +/- Y(l,k)|^2
        """
        if mode == 'sum':
            combo = stft_a + stft_b
        elif mode == 'diff':
            combo = stft_a - stft_b
        else:
            raise ValueError("mode must be 'sum' or 'diff'")

        instant_power = np.abs(combo) ** 2
        phi = np.empty_like(instant_power, dtype=float)
        phi[:, 0] = instant_power[:, 0]

        for frame in range(1, instant_power.shape[1]):
            phi[:, frame] = alpha * phi[:, frame - 1] + (1.0 - alpha) * instant_power[:, frame]

        return phi
    
    def multichannel_stft(self, audio):
        """
        Compute STFT for multichannel audio.
        """
        audio = np.asarray(audio)
        if audio.ndim != 2:
            raise ValueError("audio must have shape (n_samples, n_mics)")
            
        stft_channels = []
        for ch in range(audio.shape[1]):
            _, _, stft_ch = signal.stft(
                audio[:, ch], fs=self.fs, 
                nperseg=self.nperseg, noverlap=self.noverlap
            )
            stft_channels.append(stft_ch)
        stft = np.stack(stft_channels, axis=-1)
        return stft

    def inverse_multichannel_stft(self, stft):
        """
        Inverse STFT for multichannel audio.
        """
        audio_channels = []
        for ch in range(stft.shape[-1]):
            _, ch_out = signal.istft(stft[..., ch], fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
            audio_channels.append(ch_out)
        audio = np.stack(audio_channels, axis=-1)
        return audio

    def compute_suppression_gain(self, audio, mic_pair=(0, 1)):
        """
        Compute Nelke eq. (4.42) sum-diff suppression gain from multichannel audio.
        """
        audio = np.asarray(audio)
        if audio.ndim != 2:
            raise ValueError("audio must have shape (n_samples, n_mics)")

        ch_a, ch_b = mic_pair
        if audio.shape[1] <= max(ch_a, ch_b):
            raise ValueError(f"audio must have at least {max(ch_a, ch_b) + 1} channels")

        stft = self.multichannel_stft(audio)

        X, Y = stft[..., ch_a], stft[..., ch_b]
        phi_sum = self.short_term_psd(X, Y, alpha=self.alpha, mode='sum')
        phi_diff = self.short_term_psd(X, Y, alpha=self.alpha, mode='diff')
        power_ratio = phi_diff / np.maximum(phi_sum, 1e-12)

        d_tilde_m = np.cos(self.theta_s) * self.d_m
        mu = np.arange(power_ratio.shape[0])
        PR_s = np.sin(np.pi * d_tilde_m * mu * self.fs / (self.c_M * self.nperseg)) ** 2

        G_diff = PR_s[:, None] / np.maximum(power_ratio, PR_s[:, None] * 0.1)
        G_diff = np.clip(G_diff, self.G_min, 1.0)
        G_diff = uniform_filter1d(G_diff, size=self.smooth_size, axis=1, mode='nearest')

        return G_diff, stft

    def apply_suppression_gain(self, stft, G_diff, hp_cutoff=100, n_samples=None):
        """
        Apply suppression gain to audio.
        """
        stft_hat = G_diff[..., np.newaxis] * stft
        audio_hat = self.inverse_multichannel_stft(stft_hat)
        if n_samples is not None:
            audio_hat = audio_hat[:n_samples]
        b, a = butter(N=4, Wn=hp_cutoff / (self.fs / 2), btype='high', analog=False)
        audio_hat = filtfilt(b, a, audio_hat, axis=0)
        return audio_hat

    def suppress_wind_noise(self, audio, mic_pair=(0, 1), hp_cutoff=100, n_samples=None):
        """
        Suppress wind noise from audio.
        """
        G_diff, stft = self.compute_suppression_gain(audio, mic_pair=mic_pair)
        audio_hat = self.apply_suppression_gain(stft, G_diff, hp_cutoff=hp_cutoff, n_samples=n_samples)
        return audio_hat
