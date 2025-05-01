import librosa
import numpy as np
import soundfile as sf


class AudioProcessor:
    """Audio processing utilities."""

    __slots__ = ("_sample_rate", "_frame_length", "_hop_length", "_n_fft", "_n_mels", "_fmin", "_fmax")

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_length: int = 1024,
        hop_length: int = 256,
        n_fft: int = 1024,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: float = 8000.0,
    ) -> None:
        """Initialize audio processor.

        Args:
            sample_rate: Sample rate in Hz
            frame_length: Frame length in samples
            hop_length: Hop length in samples
            n_fft: FFT size
            n_mels: Number of mel bands
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
        """
        self._sample_rate = sample_rate
        self._frame_length = frame_length
        self._hop_length = hop_length
        self._n_fft = n_fft
        self._n_mels = n_mels
        self._fmin = fmin
        self._fmax = fmax

    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        return self._sample_rate

    @property
    def frame_length(self) -> int:
        """Get frame length."""
        return self._frame_length

    @property
    def hop_length(self) -> int:
        """Get hop length."""
        return self._hop_length

    @property
    def n_fft(self) -> int:
        """Get FFT size."""
        return self._n_fft

    @property
    def n_mels(self) -> int:
        """Get number of mel bands."""
        return self._n_mels

    @property
    def fmin(self) -> float:
        """Get minimum frequency."""
        return self._fmin

    @property
    def fmax(self) -> float:
        """Get maximum frequency."""
        return self._fmax

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Audio signal as numpy array
        """
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio

    def save_audio(self, audio: np.ndarray, file_path: str) -> None:
        """Save audio file.

        Args:
            audio: Audio signal as numpy array
            file_path: Path to save audio file
        """
        sf.write(file_path, audio, self.sample_rate)

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram.

        Args:
            audio: Audio signal as numpy array

        Returns:
            Mel spectrogram as numpy array
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    def compute_inverse_mel_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """Compute inverse mel spectrogram.

        Args:
            mel_spec: Mel spectrogram as numpy array

        Returns:
            Audio signal as numpy array
        """
        mel_spec = librosa.db_to_power(mel_spec)
        audio = librosa.feature.inverse.mel_to_audio(
            mel_spec,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.frame_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        return audio
