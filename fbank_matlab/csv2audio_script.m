clear;
path_to_result   = '/home/antonie/Project/speech2vec/raw_data/dsp_hw2/fbank';
%path_to_result   = '/home/antonie/Project/speech2vec/result/dsp_hw2_one_digit_seq2seq/fbank_delta/csv/batch_64_epoch_10000_hidden_50_depth_2,2_dropout_0.2_optimizer_nadam_epoch_8500'
has_delta = false;

path_to_data     = '/home/antonie/Project/speech2vec/raw_data/dsp_hw2';

path_to_yphase   = strcat(path_to_data,'/yphase');


AmFlag = 2;
FrameSize = 400; % Window Len
FrameRate = 120; % FrameShift
FFT_SIZE = 256;
sr = 8000;
minfreq = 120;
maxfreq = sr / 2;
nfilts = 75; %filter nuhto
am_scale = 1000;

dyn_dims = nfilts * 3;


yphase_list = dir(strcat(path_to_yphase,'/*.csv'));
results = dir(strcat(path_to_result,'/*.csv'));

for i = 1:size(results);
    fprintf('Writing flacs %d/%d...\n',i,size(results));
    [ pathstr, name, ext ] = fileparts(results(i).name);
    
    csvfile = strcat(path_to_result,'/',name,ext);
    phasefile = strcat(path_to_yphase,'/',name,'.csv');
    
    % Read feature and phase for reconstruction
    feature = csvread(csvfile);
    yphase = csvread(phasefile);
    
    [ row, col ] = size(yphase);
    
    % Invert for audiowrite
    yphase = yphase.';
    feature = feature.';
    feature = feature(:,1:row);
    if has_delta;
        diag_var = var(feature,1,2); % var(A, w, dims)
        var_Y = diag(diag_var);
        feature = generalized_MLPG_ver2(feature, var_Y, 2, dyn_dims);
    end;
    
    % Inverse from Log MFCC Spectrum
    siga_delta = Inverse_From_LogMel(feature, yphase, FrameSize, FrameRate, sr, FFT_SIZE, 'htkmel', minfreq, maxfreq, 1, 1);
    
    %sound(siga_delta, yphase)
    new_name = strcat(path_to_result,'/reconstructed_',name,'.wav');
    audiowrite(new_name, siga_delta, sr);
end;
