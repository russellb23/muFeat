# Copyright 2019 Russell. All Rights Reserved
# Use of the complete code or any part of this code is governed by the
# License terms found in the LICENSE file
# Feature extraction module
# Compute Perceptual, Spectral and Mel features
# Source separation: REPET and similairty by coseine metric
# Russell B for psiket

import librosa as lbr
import scipy.stats as scistat
import numpy as np
import math

class BeatSpectrogram:
    """Methods for computing beat spectrogram, beat spectrum and beat period"""

    def __init__(self, spec):
        self.spec = spec  # Either Spectrogram / Mel spectrogram
        self.psec = np.abs(lbr.magphase(self.spec)[0]) ** 2  # Power spectrum
        self.bspec = np.empty((self.spec.shape[1]))
        self.bspec_ = np.empty((self.spec.shape[0], self.spec.shape[1]))
        self.L = None  # Largest 1/4th of lag discarded
        self.D = 2  # Fixed deviation for the peak
        self.J = None  # minimum three periods, hence taking l//3
        self.I = None
        self.D_ = None  # 3/4th of integer multiple as variable area
        self.H = None
        self.H_ = None  # Variable delta
        self.P = None  # Repating period

    def compute_BSpectrum(self, ret):
        """ Compute beat spectrogram of the audio signal
        returns: 1D beat spectrum of the audio signal & 2D beat spectrogram
        """
        assert type(ret) == str, \
        "Please specify a string \'spectrum\'|\'spectrogram\'|\'both\'"

        self.pspec = np.abs(lbr.magphase(self.spec)[0]) ** 2  # Power spectrogram

        # Beat spectrogram
        for i in range(self.pspec.shape[0]):
            self.bspec_[i] = lbr.autocorrelate(self.pspec[i], max_size=self.pspec[i].size)

        # Beat spectrum
        for j in range(self.pspec.shape[0]):
            self.bspec += self.bspec_[j,:]

        # Normalize the beat spectrum
        self.bspec/=self.bspec[0]
        # Setting the first element to zero, so as to remove DC and preserve Length
        self.bspec[self.bspec==1]=0
        # Return what has been asked for
        if ret == 'specrtum':
            return self.bspec
        elif ret == 'spectrogram':
            return self.bspec_
        elif ret == 'both':
            return self.bspec, self.bspec_
        else:
            print("Please specify \'spectrum\'|\'spectrogram\'|\'both\'")

    def compute_BPeriod(self, bspectrum, minP):
        """ Compute beat period of the beat spectrum
        minP; int >= 2: minimum number of repeating periods expected
        returns: beat period (int)
        """
        self.L = int(bspectrum.size - (bspectrum.size/4))
        self.J = np.empty((int(self.L/minP)))
        peak_list = np.argsort(bspectrum)[::-1] # Descending peak amplitudes

        # Estimate the period of first prominent peak
        for m in range(1, int(self.L/minP)):
            for n in np.arange(peak_list[0], int(self.L/minP), m):
                self.D_ = int(0.75 * n)
                self.H = np.argmax(bspectrum[int(n-self.D):int(n+self.D)])
                self.H_ = np.argmax(bspectrum[int(n-self.D_):int(n+self.D_)])

                if self.H == self.H_:
                    self.I = self.I + bspectrum[self.H] - np.mean(bspectrum[n])
            self.J[m] = self.I/(self.L//m)
        self.P = np.argmax(self.J)

    def compute_BHisto(self, bspectrum):
        """ Compute beat histogram and information there in """
        unimplemented()


class PerceptualFeatures:
    """ Computes the perceptual features like stereo, beat etc.,"""
    def __init__(self, audio, srate, nfft, hop, pow):
        self.audio = audio
        self.srate = srate
        self.nfft = nfft
        self.hop = hop
        self.pow = pow

        self.onset_env = None
        self.tempo = None
        self.beats = None
        self.onset = None
        self.spec = lbr.feature.melspectrogram(self.audio, \
                self.srate, n_fft=self.nfft, hop_length=self.hop, \
                power=self.pow) # fft length and hop are for onset detection optimized
        self.pspec = None
        self.bspec_ = np.zeros((self.spec.shape[0], self.spec.shape[1])) #np.ndarray
        self.bspec = np.zeros(self.spec.shape[1])

    def compute_intensity(self):
        """ Compute the intnsity of the signal """
        return math.sqrt(np.mean((self.audio)**2))

    def compute_Onset(self, lag, filtr_size, aggregate, feature):
        """ Compute onset strength envelope and onset strength from the same """
        if aggregate == 'mean' and feature == 'mel':
            self.onset_env = lbr.onset.onset_strength(y=self.audio, sr=self.srate, \
            S=self.spec, lag=lag, max_size=filtr_size, aggregate=np.mean, \
            feature=lbr.feature.melspectrogram)
            self.onset = lbr.onset.onset_detect(onset_envelope=self.onset_env, hop_length=self.hop)

        elif aggregate == 'median' and feature == 'mel':
            self.onset_env = lbr.onset.onset_strength(y=self.audio, sr=self.srate, \
            S=self.spec, lag=lag, max_size=filtr_size, aggregate=np.median, \
            feature=lbr.feature.melspectrogram)
            self.onset = lbr.onset.onset_detect(onset_envelope=self.onset_env, \
            hop_length=self.hop)

        elif aggregate == 'mean' and feature == 'cqt':
            self.onset_env = lbr.onset.onset_strength(y=self.audio, sr=self.srate, \
            S=self.spec, lag=lag, max_size=filtr_size, aggregate=np.mean, \
            feature=lbr.feature.cqt)
            self.onset = lbr.onset.onset_detect(onset_envelope=self.onset_env, \
            hop_length=self.hop)

        elif aggregate == 'median' and feature == 'cqt':
            self.onset_env = lbr.onset.onset_strength(y=self.audio, sr=self.srate, \
            S=self.spec, lag=lag, max_size=filtr_size, aggregate=np.median, \
            feature=lbr.feature.cqt)
            self.onset = lbr.onset.onset_detect(onset_envelope=self.onset_env, \
            hop_length=self.hop)

        return [self.onset_env, self.onset]

    def compute_Tempo(self, tightness=100, trim=False, units='frames'):
        """ Compute the tempo and beat event locations
            timghtness: float(scalar) tightness of beat distribution around tempo
            trim: bool trim lead/trail beats with weak onset envelope
            units: units to encoe detected beat events (str)
            return: tuple of tempo(bpm) and beat events
        """
        self.tempo, self.beats =
        lbr.beat.beat_track(onset_envelope=self.onset_env, \
                tightness=tightness, trim=trim, units=units)
        return self.tempo, self.beats

    def compute_Tempogram(self):
        """ COmpute tempogram: local autocorrelaitons of onset strength env
            return np.ndarray of localized autocorrelation of onset env strength
        """
        return lbr.feature.tempogram(y=self.audio, sr=self.srate, \
                onset_envelope=self.onset_env, hop_length=self.hop, \
                win_length=self.nfft, window='hann')


class SpectralEnergyFeatures:
    """ COmputes spectral features such as mean. median,std, var and cum of E"""

    def __init__(self, audio, srate, spec, fft_len, hop):
        self.audio = audio # TS audio
        self.srate = srate # float > 0
        self.spec = np.abs(lbr.magphase(spec)[0])
        self.p_spec = self.spec ** 2
        self.fft_len = fft_len # int > 0
        self.hop = hop # default nfft/4
        self.fbands = None

    def compute_EFeatures(self, fbands):
        """ Compute energy statistics: mean, median, var, skew, kurt, etc., 
                 frequency channel wise
        fbands: np.array of frequencies generated by 'fft_frequencies'
        return: np.ndarray of (8 X fbands.size)
        """
        e_std = np.std(self.p_spec, axis=1, dtype='float64')
        return np.vstack((fbands, np.sum(self.p_spec, axis=1, dtype='float64'), \
        np.mean(self.p_spec, axis=1, dtype='float64'), e_std, e_std ** 2, \
        np.median(self.p_spec, axis=1), scistat.skew(self.p_spec, axis=1), \
        scistat.kurtosis(self.p_spec, axis=1)))

    def compute_SFeatures(self, fbands):
        """ Compute spectral features: centroid, rolloff, flatness and bandwidth
        fbands: np.array of frequencies generated by fft/mel frequencies
        return: tuple of np.ndarray
        """
        return [lbr.feature.spectral_centroid(S=self.spec, freq=self.fbands), \
        lbr.feature.spectral_bandwidth(S=self.spec, sr=self.srate, \
        n_fft=self.fft_len, hop_length=self.hop, freq=self.fbands), \
        lbr.feature.spectral_rolloff(S=np.abs(self.spec), sr=self.srate, freq=self.fbands), \
        lbr.feature.spectral_flatness(S=np.abs(self.spec), amin=1e-6, power=2.)]

    def compute_SContrast(self, fmin, nbands, quantile, linear=False):
        """ Compute spectral contrast(irregularity)
            fmin: minimum frequency fmin > 21
            nbands: no of frequency bands. defaults to 6
            quantile: Quantile to use
            linear: boolean
            return: np.ndarrray of spectral contrast
        """
        return lbr.feature.spectral_contrast(S=self.spec, sr=self.srate, \
        n_fft=self.fft_len, hop_length=self.hop, fmin=fmin, n_bands=nbands,\
         quantile=quantile, linear=False)

    def compute_TFeatures(self):
        """ COmpute temporal features such as zero crossing and ZCR
        returns: tuple of ZCR od ZC index ZCR: np.ndarray ZC idx: tuple
        """
        return (lbr.feature.zero_crossing_rate(y=self.audio,
            frame_length=self.fft_len, \
                    hop_length=self.hop, center=False), \
        np.nonzero(lbr.zero_crossings(y=self.audio)))


class MelFeatures:
    """ Compute mel-frequency cepstral coefficients for a given audio
        audio: time series audio signal (numpy ndarray)
        srate: sampling rate (int > 0)
        nfft: length of fft window (int > 0)
        hop: hop length (int > 0; hop < nfft)
        nmfcc: number of mel coefficients to compute (int >0 && <= 40)
        power: power to be used (float > 0) default 2.
    """

    def __init__(self, audio, srate, nfft, hop, nmfcc, power=2.):
        self.audio = audio
        self.srate = srate
        self.nfft = nfft
        self.hop = hop
        # self.nmfcc = None
        self.power = power
        self.melspec = lbr.feature.melspectrogram(y=self.audio, sr=self.srate,\
                n_fft=self.nfft, hop_length=self.hop, power=self.power)
        self.mfc = None

    def compute_MelCoeff(self, n_coeff):
        """ Compute mel frequency cepstral co-coefficients
        n_coeff: int > 0 && n_coeff <=40
        return np.ndarray of mfcc statistics: min, max, mean, median, std, var, kurtosis, e_skew """
        self.mfc = np.abs(lbr.feature.mfcc(S=lbr.power_to_db(self.melspec), n_mfcc=n_coeff))

        mfc_std = np.std(self.mfc, axis=1)
        return np.concatenate((np.min(self.mfc, axis=1), np.max(self.mfc, axis=1), \
        np.mean(self.mfc, axis=1), np.median(self.mfc, axis=1), mfc_std ** 2, \
        scistat.kurtosis(self.mfc, axis=1), scistat.skew(self.mfc, axis=1)), axis=0)


class SourceXtraction:
    """ Compute time-frequency masking
    """
    def __init__(self, spec):
        self.spec = spec # Full spectrogram
        # self.background = None # Extracted repeating pattern
        self.mask = None # Mask to be multiplied with the spectrogram
        self.mask_coefficient= None
        self.pow = None
        self.mask_type = None

    def compute_Mask(self, background, mask_coefficient, mask_type, pow=1.):
        """ Compute the mask
        background: np.ndarray of computed repeating structure
        mask_coefficient: float > 0
        pow: default to 1.
        mask_type: (str) 'fg' or 'bg'

        returns: np.ndarray of background.shape
        """
        assert self.spec.shape == background.shape, 'Dimension mismatch'

        if mask_type == 'fg':
            return lbr.util.softmask((self.spec - background), \
                    mask_coefficient * self.spec, power=pow)
        elif mask_type == 'bg':
            return lbr.util.softmask(self.spec, \
                    mask_coefficient * (self.spec - background), power=pow)
        else:
            raise RuntimeError

    def extract_Source(self, mask, hop, winL, windw='hann'):
        """ Extract/Separate the voice from background or vice versa
            mask: np.ndarray (symmtrized)
            hop: int > 0 && hop <= nfft==winL
            winL: int > 0: window length (use the same length of stft function)
            window: (str) defaults to 'hann' (use the same of stft function)
            return: np.ndarray of time series audio signal
        """
        assert self.spec.shape == mask.shape, 'Dimension mismatch'
        self.src_ = self.spec * mask
        return lbr.core.istft(self.src_, hop_length=hop, win_length=winL, \
                window=windw)


class VoiceXtract:
    """
    Extract vocal tract from the song: nearest neighbour filter & cosine distance
        spectrogram: stft output of the time series audio signal (numpy ndarray)
    """

    def __init__(self, spectrogram):#, width, margin_b, margin_f, power):
        self.spec = lbr.magphase(spectrogram)[0]
        self.bg = None
        self.width = None
        self.aggregate = None

    def compute_Bg(self, width, aggregate):
        """ Compute TF mask of vocal and instruments """

        self.bg = lbr.decompose.nn_filter(self.spec, aggregate=np.median, metric='cosine', width=width)
        self.bg = np.minimum(self.spec, self.bg)

        return self.bg


class Repet:
    """ Separate voice from background using REPET algorithm by Rafii """

    def __init__(self, spectrogram):
        self.spec = np.abs(lbr.magphase(spectrogram)[0])
        self.segments = None # Number of segments
        self.seg_model_ = None
        self.seg_model = None
        self.V = None # Spectrogram sliced to match the repeat spectrogram model
        self.W_ = list()
        self.W = None
        self.P = None
        self.bg = None # Extracted repeating background
        self.J = None

    def compute_RepetBG(self, bspectrum, minP):
        """Compute beat period from the beat spectrum """
        self.L = int(bspectrum.size - (bspectrum.size/4))
        self.J = np.empty(int(self.L//minP))
        for j in range(1,int(self.L/minP)):
            self.I = 0
            self.integer_mults = np.arange(2, int(self.L/minP), j)
            for i in self.integer_mults:
                self.delta = int(0.75 * i)
                self.H = np.argmax(bspectrum[int(i-self.del_p):int(i+self.del_p)])
                self.H_delta = np.argmax(bspectrum[int(i-self.delta):int(i+self.delta)])

                if self.H == self.H_delta:
                    self.I = self.I + bspectrum[self.H] - np.mean(bspectrum[i])
            self.J[j] = self.I/(self.L//j)
        self.P = np.argmax(self.J)
        """ Computing repet segment model"""
        self.segments = self.spec.shape[1]//self.P # No of segments
        self.seg_model_ = np.reshape(self.spec[:,0:self.segments*self.P], \
                (self.segments, self.spec.shape[0], self.P))
         # Predec segment model with shape
        self.seg_model = np.empty((self.spec.shape[0], self.P))
        # Element-wise median
        for n in range(self.spec.shape[0]):
            for m in range(self.P):
                self.seg_model[n][m] = np.median(self.seg_model_[:,n,m])
                # print(self.seg_model.shape)

        # print(self.seg_model.shape)
        """ Extract repeating pattern from the spectrogram to separate voice """
        # Repeting spectrogram model W = minimum(S, V)
        # Slice only the segment times from full spectrogram
        self.V = self.spec[:,0:self.segments*self.seg_model.shape[1]]

        for x in range(self.segments):
            self.W_.append(np.minimum(self.seg_model, \
                    self.spec[:,(x*self.seg_model.shape[1]):(x+1)*self.seg_model.shape[1]]))

        self.W = np.concatenate(tuple(self.W_), axis=1)
        # Symmetrize the background
        self.bg = np.concatenate([self.mask_v[:0:-1, ...], self.mask_v], axis=0)

        return self.bg
