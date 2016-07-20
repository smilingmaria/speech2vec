function csv2audio(path_to_result, path_to_data);

path_to_yphase   = strcat(path_to_data,'/yphase');

has_delta = false;

AmFlag = 2;
FrameSize = 400; % Window Len
FrameRate = 120; % FrameShift
FFT_SIZE = 256;
sr = 16000;
minfreq = 120;
maxfreq = sr / 2;
nfilts = 75; %filter nuhto
am_scale = 1000;

dyn_dims = nfilts * 3;


yphase_list = dir(path_to_yphase);
results = dir(path_to_result);

for i = 1:size(results);
    
    [ pathstr, name, ext ] = fileparts(results(i).name);
    if ~strcmp(ext,'.csv');
        continue;
    end;
    disp(name);
    
    csvfile = strcat(path_to_result,'/',name,ext);
    phasefile = strcat(path_to_yphase,'/',name,'.phase');
    
    % Read feature and phase for reconstruction
    feature = csvread(csvfile);
    yphase = csvread(phasefile);
    % Invert for audiowrite
    feature = feature.';
    
    if has_delta;
        diag_var = var(feature,1,2); % var(A, w, dims)
        var_Y = diag(diag_var);
        feature = generalized_MLPG_ver2(feature, var_Y, 2, dyn_dims);
    end;
    % Inverse from Log MFCC Spectrum
    siga_delta = Inverse_From_LogMel(feature, yphase, FrameSize, FrameRate, sr, FFT_SIZE, 'htkmel', minfreq, maxfreq, 1, 1);
    new_name = strcat(path_to_result,'/reconstructed_',name,'.flac');
    audiowrite(new_name, siga_delta, sr);
end;
end;
