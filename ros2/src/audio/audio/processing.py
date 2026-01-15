import numpy as np
from .utils.grid_and_tdoas import (
    fibonacci_half_sphere,
    fibonacci_sphere,
    calculate_tdoa,
)
from .utils.localisation_algos import SRP_PHAT_offline, SVD_PHAT_offline


class AudioProcessor:
    def __init__(
        self,
        mic_pos_path,
        fs,
        nb_of_channels,
        nb_points,
        loc_type,
        grid_type,
        window_size,
        nfft,
        FRAME_SIZE,
    ):
        # -- Setup config --
        self.mic_pos_path = mic_pos_path
        self.fs = fs
        self.nb_of_channels = nb_of_channels
        self.nb_points = nb_points
        self.loc_type = loc_type
        self.grid_type = grid_type
        self.window_size = window_size
        self.nfft = nfft
        self.FRAME_SIZE = FRAME_SIZE

        # -- Load microphone array --
        self.mic_pos = np.load(mic_pos_path)

        # -- Compute grid --
        self._precompute_grid()

        # -- Compute TDOA candidates list
        self._precompute_tdoa_candidates()

        # -- Compute frequency vector
        self._precompute_f()

        # -- Compute window vector
        self._precompute_window()

        # Loc algo - dependant precomputation
        if self.loc_type == "SRP-PHAT":
            print("Precomputing SRP-PHAT")
            self._precompute_srp_phat()  # W matrix
        elif self.loc_type == "SVD-PHAT":
            print("Precomputing SVD-PHAT")
            self._precompute_svd_phat()  # D, Vh_k

    def process_frame(self, audio_frame, frame_timestamp):
        """
        audio_frame: [FRAME_SIZE, NB_OF_CHANNELS] numpy array
        Output: [x, y, z] DOA vector
        """

        # fft and Cross spectrum
        XXs = self.get_cross_spec_from_sig(audio_frame)

        # Loc algo
        if self.loc_type == "SRP-PHAT":
            SRP = self.SRP_PHAT(XXs)
        elif self.loc_type == "SVD-PHAT":
            SRP = self.SVD_PHAT(XXs)
        else:
            SRP = np.zeros(audio_frame)

        # Compute DOA from SRP
        DOA_coordinates = self.get_DOA_from_SRP(SRP)

        return DOA_coordinates.tolist()  # Result

    def get_cross_spec_from_sig(self, xs):
        # (Normalization if necessary)
        # STFT [nb_of_bins, nb_of_channels]
        Xs = np.fft.rfft(xs, self.nfft, axis=0)

        # Cross-spectrum [nb_of_bins, nb_of_channels, nb_of_channels]
        XXs = np.einsum("fc,fd->fcd", Xs, np.conj(Xs))

        return XXs

    def get_DOA_from_SRP(self, SRP):
        # DOA id  = argmax in SRP
        DOA_id = np.argmax(SRP)

        # Look into grid for DOA coordinates
        DOA_coordinates = self.scan_grid[DOA_id, :]

        return DOA_coordinates

    def SRP_PHAT(self, XXs):
        # PHAT [nb_of_bins, nb_of_channels, nb_of_channels]
        XXs_PHAT = XXs / np.abs(XXs + 1e-10)

        # Vectorize XXs_PHAT [nb_of_pairs*nb_of_bins,]
        XXs_PHAT_vec = XXs_PHAT[
            :, self.triu_indices[0], self.triu_indices[1]
        ].T.flatten()

        # SRP [nb_of_doas]
        SRP = np.real(self.W @ XXs_PHAT_vec)

        return SRP

    def SVD_PHAT(self, XXs):
        # PHAT [nb_of_bins, nb_of_channels, nb_of_channels]
        XXs_PHAT = XXs / np.abs(XXs + 1e-10)

        # Vectorize XXs_PHAT [nb_of_pairs*nb_of_bins,]
        XXs_PHAT_vec = XXs_PHAT[
            :, self.triu_indices[0], self.triu_indices[1]
        ].T.flatten()

        z = self.Vh_k @ XXs_PHAT_vec

        SRP = np.real(self.D @ z)

        return SRP

    def _precompute_grid(self):
        if self.grid_type == "fibonacci_sphere":
            self.scan_grid = fibonacci_sphere(self.nb_points)
        elif self.grid_type == "fibonacci_half_sphere":
            self.scan_grid = fibonacci_half_sphere(self.nb_points)
        else:
            raise ValueError(f"Unknown grid_type: {self.grid_type}")

    def _precompute_tdoa_candidates(self):
        self.TDOAs_candidates = calculate_tdoa(self.mic_pos, self.scan_grid)

    def _precompute_f(self):
        f = np.fft.rfftfreq(self.nfft, d=1 / self.fs)
        self.f = f.astype(np.float32)

    def _precompute_window(self):
        ws = np.tile(np.hanning(self.FRAME_SIZE), (self.nb_of_channels, 1))
        self.ws = ws.astype(np.float32)

    def _precompute_srp_phat(self):
        self.W = SRP_PHAT_offline(self.TDOAs_candidates, self.nb_of_channels, self.f)

        self.triu_indices = np.triu_indices(self.nb_of_channels, k=1)

    def _precompute_svd_phat(self):
        self.D, self.Vh_k = SVD_PHAT_offline(self.TDOAs_candidates, self.f, delta=0.01)

        self.triu_indices = np.triu_indices(self.nb_of_channels, k=1)
