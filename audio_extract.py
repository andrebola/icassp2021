# -*- coding: utf-8 -*-

from essentia.streaming import *
import essentia.standard as es
import essentia
import librosa
import librosa.display
import numpy as np


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


def analyze_mel(filename, segment_duration=None, maxFrequency=11025, replaygain=True):
    lowlevelFrameSize=2048
    lowlevelHopSize=1024

    # Compute replay gain and duration on the entire file, then load the
    # segment that is centered in time with replaygain applied
    audio = es.MonoLoader(filename=filename)()

    if replaygain:
        replaygain = es.ReplayGain()(audio)
    else:
        replaygain = -6 # Default replaygain value in EasyLoader

    if segment_duration:
        segment_start = (len(audio) / 44100 - segment_duration) / 2
        segment_end = segment_start + segment_duration
    else:
        segment_start = 0
        segment_end = len(audio)/44100

    if segment_start < 0 or segment_end > len(audio)/44100:
      raise ValueError('Segment duration is larger than the input audio duration')

    loader_mel = EasyLoader(filename=filename, replayGain=replaygain,
                            startTime=segment_start, endTime=segment_end)

    # Processing for Mel bands
    framecutter_mel = FrameCutter(frameSize=lowlevelFrameSize,
                                  hopSize=lowlevelHopSize)
    window_mel = Windowing(type='blackmanharris62', zeroPadding=lowlevelFrameSize)

    spectrum_mel = Spectrum()

    melbands128 = MelBands(numberBands=128,
                          lowFrequencyBound=0,
                          highFrequencyBound=maxFrequency,
                          inputSize=lowlevelFrameSize+1)

    melbands96 = MelBands(numberBands=96,
                          lowFrequencyBound=0,
                          highFrequencyBound=maxFrequency,
                          inputSize=lowlevelFrameSize+1)

    melbands48 = MelBands(numberBands=48,
                          lowFrequencyBound=0,
                          highFrequencyBound=maxFrequency,
                          inputSize=lowlevelFrameSize+1)

    melbands32 = MelBands(numberBands=32,
                          lowFrequencyBound=0,
                          highFrequencyBound=maxFrequency,
                          inputSize=lowlevelFrameSize+1)

    melbands24 = MelBands(numberBands=24,
                          lowFrequencyBound=0,
                          highFrequencyBound=maxFrequency,
                          inputSize=lowlevelFrameSize+1)

    melbands16 = MelBands(numberBands=16,
                          lowFrequencyBound=0,
                          highFrequencyBound=maxFrequency,
                          inputSize=lowlevelFrameSize+1)

    melbands8 = MelBands(numberBands=8,
                         lowFrequencyBound=0,
                         highFrequencyBound=maxFrequency,
                         inputSize=lowlevelFrameSize+1)



    # Normalize Mel bands: log10(1+x*10000)
    norm128 = UnaryOperator(type='identity', shift=1, scale=10000)
    log10128 = UnaryOperator(type='log10')

    norm96 = UnaryOperator(type='identity', shift=1, scale=10000)
    log1096 = UnaryOperator(type='log10')

    norm48 = UnaryOperator(type='identity', shift=1, scale=10000)
    log1048 = UnaryOperator(type='log10')

    norm32 = UnaryOperator(type='identity', shift=1, scale=10000)
    log1032 = UnaryOperator(type='log10')

    norm24 = UnaryOperator(type='identity', shift=1, scale=10000)
    log1024 = UnaryOperator(type='log10')

    norm16 = UnaryOperator(type='identity', shift=1, scale=10000)
    log1016 = UnaryOperator(type='log10')

    norm8 = UnaryOperator(type='identity', shift=1, scale=10000)
    log108 = UnaryOperator(type='log10')

    p = essentia.Pool()

    loader_mel.audio >> framecutter_mel.signal
    framecutter_mel.frame >> window_mel.frame >> spectrum_mel.frame

    spectrum_mel.spectrum >> melbands128.spectrum
    spectrum_mel.spectrum >> melbands96.spectrum
    spectrum_mel.spectrum >> melbands48.spectrum
    spectrum_mel.spectrum >> melbands32.spectrum
    spectrum_mel.spectrum >> melbands24.spectrum
    spectrum_mel.spectrum >> melbands16.spectrum
    spectrum_mel.spectrum >> melbands8.spectrum

    melbands128.bands >> norm128.array >> log10128.array >> (p, 'mel128')
    melbands96.bands >> norm96.array >> log1096.array >> (p, 'mel96')
    melbands48.bands >> norm48.array >> log1048.array >> (p, 'mel48')
    melbands32.bands >> norm32.array >> log1032.array >> (p, 'mel32')
    melbands24.bands >> norm24.array >> log1024.array >> (p, 'mel24')
    melbands16.bands >> norm16.array >> log1016.array >> (p, 'mel16')
    melbands8.bands >> norm8.array >> log108.array >> (p, 'mel8')

    essentia.run(loader_mel)

    return p


def analyze(filename, segment_duration=20):

    lowlevelFrameSize=2048
    lowlevelHopSize=1024
    tonalFrameSize=4096
    tonalHopSize=1024

    # Compute replay gain and duration on the entire file, then load the
    # segment that is centered in time with replaygain applied
    audio = es.MonoLoader(filename=filename)()
    replaygain = es.ReplayGain()(audio)

    segment_start = (len(audio) / 44100 - segment_duration) / 2
    segment_end = segment_start + segment_duration

    if segment_start < 0 or segment_end > len(audio)/44100:
      raise ValueError('Segment duration is larger than the input audio duration')

    # TODO
    # There's a bug in streaming mode Python wrapper: running both Mel and HPCP
    # in the same network with the same loader will result in a memory error.
    # This does not happen in C++. As a workaround, compute Mel and HPCP in
    # two separate networks with two separate loaders.

    loader_mel = EasyLoader(filename=filename, replayGain=replaygain,
                            startTime=segment_start, endTime=segment_end)
    loader_hpcp = EasyLoader(filename=filename, replayGain=replaygain,
                            startTime=segment_start, endTime=segment_end)

    # Processing for Mel bands
    framecutter_mel = FrameCutter(frameSize=lowlevelFrameSize,
                                  hopSize=lowlevelHopSize)
    window_mel = Windowing(type='blackmanharris62')
    spectrum_mel = Spectrum()
    melbands = MelBands(numberBands=96,
                        lowFrequencyBound=0,
                        highFrequencyBound=11025)

    # Processing for HPCPs
    framecutter_hpcp = FrameCutter(frameSize=tonalFrameSize,
                                   hopSize=tonalHopSize)
    window_hpcp = Windowing(type='blackmanharris62')
    spectrum_hpcp = Spectrum()
    speaks = SpectralPeaks(maxPeaks=60,
                           magnitudeThreshold=0.00001,
                           minFrequency=20.0,
                           maxFrequency=3500.0,
                           orderBy='magnitude')

    # Normalize Mel bands: log10(1+x*10000)
    norm = UnaryOperator(type='identity', shift=1, scale=10000)
    log10 = UnaryOperator(type='log10')


    hpcp = HPCP(size=12,
                bandPreset=False,
                minFrequency=20.0,
                maxFrequency=3500.0,
                weightType='cosine',
                windowSize=1.)

    p = essentia.Pool()

    loader_mel.audio >> framecutter_mel.signal
    framecutter_mel.frame >> window_mel.frame >> spectrum_mel.frame
    spectrum_mel.spectrum >> melbands.spectrum
    melbands.bands >> norm.array >> log10.array >> (p, 'melbands')
    essentia.run(loader_mel)

    loader_hpcp.audio >> framecutter_hpcp.signal
    framecutter_hpcp.frame >> window_hpcp.frame  >> spectrum_hpcp.frame
    spectrum_hpcp.spectrum >> speaks.spectrum
    speaks.frequencies >> hpcp.frequencies
    speaks.magnitudes >> hpcp.magnitudes
    hpcp.hpcp >> (p, 'hpcp')
    essentia.run(loader_hpcp)

    return p



def analyze_misc(filename, segment_duration=20):

    # Compute replay gain and duration on the entire file, then load the
    # segment that is centered in time with replaygain applied
    audio = es.MonoLoader(filename=filename)()
    replaygain = es.ReplayGain()(audio)

    segment_start = (len(audio) / 44100 - segment_duration) / 2
    segment_end = segment_start + segment_duration

    if segment_start < 0 or segment_end > len(audio)/44100:
      raise ValueError('Segment duration is larger than the input audio duration')

    loader = es.EasyLoader(filename=filename, replayGain=replaygain,
                           startTime=segment_start, endTime=segment_end)

    windowing = es.Windowing(type='blackmanharris62')
    spectrum = es.Spectrum()
    powerspectrum = es.PowerSpectrum()
    centroid = es.Centroid()
    zcr = es.ZeroCrossingRate()
    rms = es.RMS()
    hfc = es.HFC()
    pool = essentia.Pool()

    audio = loader()
    for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=1024):
        frame_spectrum = spectrum(windowing(frame))
        pool.add('rms', rms(frame))
        pool.add('rms_spectrum', rms(frame_spectrum))
        pool.add('hfc', hfc(frame_spectrum))
        pool.add('spectral_centroid', centroid(frame_spectrum))
        pool.add('zcr', zcr(frame))


    audio_st, sr, _, _, _, _ = es.AudioLoader(filename=filename)()
    # Ugly hack because we don't have a StereoResample
    left, right = es.StereoDemuxer()(audio_st)
    resampler = es.Resample(inputSampleRate=sr, outputSampleRate=44100)
    left = resampler(left)
    right = resampler(right)
    audio_st = es.StereoMuxer()(left, right)
    audio_st = es.StereoTrimmer(startTime=segment_start, endTime=segment_end)(audio_st)
    ebu_momentary, _, _, _ = es.LoudnessEBUR128(hopSize=1024/44100, startAtZero=True)(audio_st)
    pool.set('ebu_momentary', ebu_momentary)

    return pool


def analyze_hp(filename, segment_duration=20):

    lowlevelFrameSize=2048
    lowlevelHopSize=1024
    tonalFrameSize=4096
    tonalHopSize=1024

    # Compute replay gain and duration on the entire file, then load the
    # segment that is centered in time with replaygain applied
    audio = es.MonoLoader(filename=filename)()
    replaygain = es.ReplayGain()(audio)

    segment_start = (len(audio) / 44100 - segment_duration) / 2
    segment_end = segment_start + segment_duration

    if segment_start < 0 or segment_end > len(audio)/44100:
      raise ValueError('Segment duration is larger than the input audio duration')

    loader = es.EasyLoader(filename=filename, replayGain=replaygain,
                           startTime=segment_start, endTime=segment_end)
    window = es.Windowing(type='blackmanharris62')
    fft = es.FFT()

    stft = []

    audio = loader()
    for frame in es.FrameGenerator(audio, frameSize=lowlevelFrameSize, hopSize=lowlevelHopSize):
      stft.append(fft(window(frame)))

    # Librosa requires bins x frames format
    stft = np.array(stft).T

    D_harmonic, D_percussive = librosa.decompose.hpss(stft, margin=8)
    D_percussive_magnitude, _ = librosa.magphase(D_percussive)
    D_harmonic_magnitude, _ = librosa.magphase(D_harmonic)

    # Convert back to Essentia format (frames x bins)
    spectrum_harmonic = D_harmonic_magnitude.T
    specturm_percussive = D_percussive_magnitude.T

    # Processing for Mel bands
    melbands = es.MelBands(numberBands=96,
                           lowFrequencyBound=0,
                           highFrequencyBound=11025)

    # Normalize Mel bands: log10(1+x*10000)
    norm = es.UnaryOperator(type='identity', shift=1, scale=10000)
    log10 = es.UnaryOperator(type='log10')

    p = essentia.Pool()

    for spectrum_frame in spectrum_harmonic:
        p.add('melbands_harmonic', log10(norm(melbands(spectrum_frame))))

    for spectrum_frame in specturm_percussive:
        p.add('melbands_percussive', log10(norm(melbands(spectrum_frame))))

    return p

