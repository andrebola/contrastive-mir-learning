import sys
import librosa
import numpy as np
#import soundfile as sf
import functools
import torch
#from torch.nn.functional import cosine_similarity
#import essentia.standard as es


def logme(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        print('\n-----------------\n')
        print('   MODEL: {}'.format(f.__name__.upper()))
        print('\n-----------------\n')
        return f(*args, **kwargs)
    return wrapped


class ProgressBar:
    """Progress bar

    """
    def __init__ (self, valmax, maxbar, title):
        if valmax == 0:  valmax = 1
        if maxbar > 200: maxbar = 200
        self.valmax = valmax
        self.maxbar = maxbar
        self.title  = title
        print ('')

    def update(self, val, avg_loss=0):
        # format
        if val > self.valmax: val = self.valmax

        # process
        perc  = round((float(val) / float(self.valmax)) * 100)
        scale = 100.0 / float(self.maxbar)
        bar   = int(perc / scale)

        # render
        if avg_loss:
            # out = '\r %20s [%s%s] %3d / %3d  cost: %.2f  r_loss: %.0f  l_loss: %.4f  clf_loss: %.4f' % (
            out = '\r %20s [%s%s] %3d / %3d  loss: %.5f' % (
                self.title,
                '=' * bar, ' ' * (self.maxbar - bar),
                val,
                self.valmax,
                avg_loss,
                )
        else:
            out = '\r %20s [%s%s] %3d / %3d ' % (self.title, '=' * bar, ' ' * (self.maxbar - bar), val, self.valmax)

        sys.stdout.write(out)
        sys.stdout.flush()


def pad(l, sr):
    # 0-Pad 10 sec at fs hz and add little noise
    z = np.zeros(10*sr, dtype='float32')
    z[:l.size] = l
    z = z + 5*1e-4*np.random.rand(z.size).astype('float32')
    return z


def compute_spectrogram(filename, sr=22000, n_mels=96):
    # zero pad and compute log mel spec
    try:
        audio, sr = librosa.load(filename, sr=sr, res_type='kaiser_fast')
    except:
        audio, o_sr = sf.read(filename)
        audio = librosa.core.resample(audio, o_sr, sr)
    try:
        x = pad(audio, sr)
    except ValueError:
        x = audio
    audio_rep = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=512, n_fft=1024, n_mels=n_mels, power=1.)
    audio_rep = np.log(audio_rep + np.finfo(np.float32).eps)
    return audio_rep


def return_spectrogram_max_nrg_frame(spectrogram):
    frames = librosa.util.frame(np.asfortranarray(spectrogram), frame_length=96, hop_length=12)
    idx_max_nrg = np.argmax(np.sum(np.sum(frames, axis=0), axis=0))
    return frames[:,:,idx_max_nrg]


def return_spectrogram_3_max_nrg_frames(spectrogram):
    frames = librosa.util.frame(np.asfortranarray(spectrogram), frame_length=96, hop_length=12)
    idxes_max_nrg = (-np.sum(np.sum(frames, axis=0), axis=0)).argsort()[:3]
    return frames[:,:,idxes_max_nrg]


def spectrogram_to_audio(filename, y, sr=22000):
    y = np.exp(y)
    x = librosa.feature.inverse.mel_to_audio(y, sr=sr, n_fft=1024, hop_length=512, power=1.)
    librosa.output.write_wav(filename, x, sr)


def extract_spectrogram(filename, sr=16000, n_mels=48):
    audio = cut_audio(filename, sampleRate=sr, segment_duration=29.1)
    frames = melspectrogram(audio, sampleRate=sr, frameSize=512, hopSize=256, numberBands=[48],
                            warpingFormula='slaneyMel', window='hann', normalize='unit_tri')

    return frames['mel_48_db'].T


def melspectrogram(audio, sampleRate=44100, frameSize=2048, hopSize=1024,
                   window='blackmanharris62', zeroPadding=0, center=True,
                   numberBands=[128, 96, 48, 32, 24, 16, 8],
                   lowFrequencyBound=0, highFrequencyBound=None,
                   weighting='linear', warpingFormula='slaneyMel', normalize='unit_tri'):

    if highFrequencyBound is None:
        highFrequencyBound = sampleRate/2

    windowing = es.Windowing(type=window, normalized=False, zeroPadding=zeroPadding)
    spectrum = es.Spectrum()
    melbands = {}
    for nBands in numberBands:
        melbands[nBands] = es.MelBands(numberBands=nBands,
                                       sampleRate=sampleRate,
                                       lowFrequencyBound=lowFrequencyBound,
                                       highFrequencyBound=highFrequencyBound,
                                       inputSize=(frameSize+zeroPadding)//2+1,
                                       weighting=weighting,
                                       normalize=normalize,
                                       warpingFormula=warpingFormula,
                                       type='power')
    norm10k = es.UnaryOperator(type='identity', shift=1, scale=10000)
    log10 = es.UnaryOperator(type='log10')
    amp2db = es.UnaryOperator(type='lin2db', scale=2)

    results = essentia.Pool()

    for frame in es.FrameGenerator(audio, frameSize=frameSize, hopSize=hopSize,
                                   startFromZero=not center):
        spectrumFrame = spectrum(windowing(frame))

        for nBands in numberBands:
            melFrame = melbands[nBands](spectrumFrame)
            results.add('mel_' + str(nBands)+'_db', amp2db(melFrame))
            results.add('mel_' + str(nBands)+'_log1+10kx', log10(norm10k(melFrame)))
            results.add('mel_' + str(nBands), melFrame)

    return results


def cut_audio(filename, sampleRate=44100, segment_duration=None):

    audio = es.MonoLoader(filename=filename, sampleRate=sampleRate)()

    if segment_duration:
        segment_duration = round(segment_duration*sampleRate)
        segment_start = (len(audio) - segment_duration) // 2
        segment_end = segment_start + segment_duration
    else:
        segment_start = 0
        segment_end = len(audio)

    if segment_start < 0 or segment_end > len(audio):
      raise ValueError('Segment duration is larger than the input audio duration')

    return audio[segment_start:segment_end]


def kullback_leibler(y_hat, y):
    """Generalized Kullback Leibler divergence.
    :param y_hat: The predicted distribution.
    :type y_hat: torch.Tensor
    :param y: The true distribution.
    :type y: torch.Tensor
    :return: The generalized Kullback Leibler divergence\
             between predicted and true distributions.
    :rtype: torch.Tensor
    """
    return (y * (y.add(1e-5).log() - y_hat.add(1e-5).log()) + (y_hat - y)).sum(dim=-1).mean()


def embeddings_to_cosine_similarity_matrix(z):
    """Converts a a tensor of n embeddings to an (n, n) tensor of similarities.
    """
    cosine_similarity = torch.matmul(z, z.t())
    embedding_norms = torch.norm(z, p=2, dim=1)
    embedding_norms_mat = embedding_norms.unsqueeze(0)*embedding_norms.unsqueeze(1)
    cosine_similarity = cosine_similarity / (embedding_norms_mat)
    return cosine_similarity


def contrastive_loss(z_audio, z_tag, t=1):
    """Computes contrastive loss following the paper:
        A Simple Framework for Contrastive Learning of Visual Representations
        https://arxiv.org/pdf/2002.05709v1.pdf
        TODO: make it robust to NaN (with low values of t it happens).
        e.g Cast to double float for exp calculation.
    """
    z = torch.cat((z_audio, z_tag), dim=0)
    s = embeddings_to_cosine_similarity_matrix(z)
    N = int(s.shape[0]/2)
    s = torch.exp(s/t)
    try:
        s = s * (1 - torch.eye(len(s), len(s)).cuda())
        # s[range(len(s)), range(len(s))] = torch.zeros((len(s),)).cuda()
    except AssertionError:
        s = s * (1 - torch.eye(len(s), len(s)))
    denom = s.sum(dim=-1)
    num = torch.cat((s[:N,N:].diag(), s[N:,:N].diag()), dim=0)
    return torch.log((num / denom) + 1e-5).neg().mean()
