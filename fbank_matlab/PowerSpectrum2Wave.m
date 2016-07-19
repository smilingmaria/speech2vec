function sig=PowerSpectrum2Wave(log10powerspectrum,yphase,windowLen,ShiftLen)
%sig=InversePowerSpectrum(log10powerspectrum,yphase)
%log10powerspectrum: estimated from deep autoencoder, must be 129*frames,
%must be log10 compressed
%yphase: clean or noisy phase information, must be a matrix as 256*frames
%Xugang Lu @NICT


logpowspectrum             =log(power(10,log10powerspectrum)); %log power spectrum
yphase                     =yphase(1:floor(size(yphase,1)/2)+1,:); %For Odd sample
sig                        =OverlapAdd(sqrt(exp(logpowspectrum)-0.01),yphase,windowLen,ShiftLen);

return;
