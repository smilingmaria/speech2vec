clear all; close all; clc;

wavname = 'N110003.wav';
newname = 'N110003_restruct.wav';

AmFlag = 2;
FrameSize = 400; % Window Len
FrameRate = 120; % FrameShift
FFT_SIZE = 256;
sr = 8000;
minfreq = 120;
maxfreq = sr / 2;
nfilts = 75; %filter num
am_scale = 1000;

dyn_dims = nfilts * 3;

x = audioread(wavname);

% extract function only support single channel signal
x = x(:, 1);
[Log_MFCSpectrum, yphase] = Mel_Spectrum_FromX(x * am_scale, AmFlag, FrameSize, FrameRate, FFT_SIZE, sr, minfreq, nfilts);

%total_feature is the output of extraction process
total_feature = dynamic_feature_ver1(Log_MFCSpectrum, 2);

% cont the revese part
diag_var = var(total_feature,1,2); % var(A, w, dims)
var_Y = diag(diag_var);

after = generalized_MLPG_ver2(total_feature, var_Y, 2, dyn_dims);
siga_delta = Inverse_From_LogMel(after, yphase, FrameSize, FrameRate, sr, FFT_SIZE, 'htkmel', minfreq, maxfreq, 1, 1);

sound(siga_delta,sr)

%audiowrite(newname, siga_delta, sr);
