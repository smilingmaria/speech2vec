function [siga]=Inverse_From_LogMel(Log_MFCSpectrum, yphase, FrameSize,FrameRate, sr, nfft, fbtype, minfreq, maxfreq, sumpower, bwidth)

MelSpec             =power(10, Log_MFCSpectrum);
[spec,wts,iwts]     =MelSpectrum2PowerSpectrum(MelSpec, sr, nfft, fbtype, minfreq, maxfreq, sumpower, bwidth);
log10powerspectrum  =log10(spec);

sig=PowerSpectrum2Wave(log10powerspectrum,yphase, 256, 128); % windowLen,ShiftLen
siga=sig/max(abs(sig)); 