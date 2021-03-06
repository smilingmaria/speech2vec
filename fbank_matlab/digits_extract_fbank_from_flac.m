clear;
%% Select the directory of flac files, the converted features will be saved

path_to_dir = '/home/antonie/Project/speech2vec/raw_data/dsp_hw2/';
path_to_flacs = strcat(path_to_dir,'wavs/');
path_to_phase = strcat(path_to_dir,'yphase/');
path_to_fbank = strcat(path_to_dir,'fbank/');
path_to_fbank_delta = strcat(path_to_dir,'fbank_delta/');

%% Mel Frequency Extraction Parameters
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

counter = 1;

files = dir(strcat(path_to_flacs,'*.wav'));
data_num = length(files);

index2word_map = cell(data_num,2);
yphase_map     = cell(data_num,1);
fbank          = cell(data_num,1);
fbank_delta    = cell(data_num,1);
digit_labels   = []

for file = files';
    [ pathstr, name, ext ] = fileparts(file.name);
    
    fprintf('Processing %d / %d\n',counter,data_num);
    
    flacfile = strcat(path_to_flacs,name,ext);
    [ y, Fs ] = audioread(flacfile);
   
    % Extract Log Mel Frequency Spectrum
    [ Log_MFCSpectrum, yphase ] = Mel_Spectrum_FromX(y * am_scale, AmFlag, FrameSize, FrameRate, FFT_SIZE, sr, minfreq, nfilts);
    
    % Skip this sequence if is too small
    [ mcc_dim seq_length ] = size(Log_MFCSpectrum);
    %if seq_length < 3;
    %    continue;
    %end;
    % Add Delta
    delta_Log_MFCSpectrum = dynamic_feature_ver1(Log_MFCSpectrum, 2);
    
    % Store data
    yphase_map(counter) = { yphase.' };
    fbank(counter) = {Log_MFCSpectrum.'};
    fbank_delta(counter) = {delta_Log_MFCSpectrum.'};
    
    % Data to word id Map
        
    parsed_name = strread(name,'%s','delimiter','_');
    word_id = parsed_name;
    
    index2word_map(counter,1) = {counter};
    %index2word_map(counter,2) = {str2num(word_id)};
    index2word_map(counter,2) = {strcat(name,ext)};

    %digit_labels = [ digit_labels, str2num(word_id) ];
    counter = counter + 1;
end;

% Save Data to word_id map

%{
index2word_map( all(cellfun(@isempty,index2word_map),2),:) = [];

save_csv_path = strcat(path_to_dir,'fbank.map');
fid = fopen(save_csv_path,'wt+');
[ nrow, ncol ] = size(index2word_map);
fprintf('Writing map...\n');
for k=1:nrow;
    [ idx, flac_idx ] = index2word_map{k,:}
    fprintf(fid,'%d,%s\n',idx,flac_idx);
end;
fclose(fid);
%}


% Save yphase, fbank, fbank_delta to csv files
number_of_features = length(fbank);
for k = 1:length(fbank);
    fprintf('Writing %d / %d\n',k,number_of_features);
    phase_name = strcat(int2str(k),'.csv');
    arr_name = strcat(int2str(k),'.csv');
    %disp(arr_name);
    
    curr_phase     = yphase_map{k};
    curr_arr       = fbank{k};
    curr_arr_delta = fbank_delta{k};
    
    path_phase = strcat(path_to_phase,phase_name);
    path_arr   = strcat(path_to_fbank,arr_name);
    path_delta = strcat(path_to_fbank_delta,arr_name);
    
    csvwrite(path_phase, curr_phase);
    csvwrite(path_arr,   curr_arr);
    csvwrite(path_delta, curr_arr_delta);
end;
