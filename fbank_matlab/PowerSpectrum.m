function [Spectrum,En] = PowerSpectrum(y,FrameLength,FrameRate,FFT_SIZE, flag);
%function [Spectrum,En] = PowerSpectrum(y,FrameLength,FrameRate,FFT_SIZE,flag);
%y: input wave data
%FrameLength: frame window length (256)
%FrameRate: frame shift (128)
%FFT_SIZE: fft size (256)
%If flag==2, power spectrum, else Amplitude

%Xugang Lu
%May 15, 2009, @ATR/NICT

Len      =length(y);
ncols    =fix((Len-FrameLength)/FrameRate);
sp       =zeros(FFT_SIZE,ncols);
Spectrum =zeros(FFT_SIZE/2,ncols);
En       =zeros(1,ncols);
wind     =hamming(FrameLength);
i        =1;
for t = 1:FrameRate:Len-FrameLength;
    sp(:,i)         = fft(wind.*y(t:(t+FrameLength-1)),FFT_SIZE);
    Spectrum(:,i)   = abs(sp(1:FFT_SIZE/2,i));    
    En(i)           =log10(1+sum(abs(Spectrum(:,i)).^2));
    i               = i+1;
end;
if flag==2
    Spectrum  =Spectrum.^2;
else
    Spectrum  =Spectrum;
end

return;

