% 2015.2.26 constructed by Hwang, Hsin-Te
% 2016.1.26 modified by Wu, Ted
% 1.This function performs Maximum likelihood parameter generation (MLPG) algorithm to 
%   smooth ANN-mapping's output
function [converted_seq_mlpg]=generalized_MLPG_ver2(Input_seq,Cov,dynamic_flag,featureDIM)
converted_seq = zeros(featureDIM/(dynamic_flag+1),size(Input_seq,2));
parfor i=1:(featureDIM/(dynamic_flag+1))
    Cov_new = zeros(3,3);
    Input_seq_new = zeros(3,size(Input_seq,2));     
    Input_seq_new(1,:) = Input_seq(i,:);
    Input_seq_new(2,:) = Input_seq(i+(featureDIM/(dynamic_flag+1)),:);      
    Input_seq_new(3,:) = Input_seq(i+(featureDIM/(dynamic_flag+1))*2,:);      
    Cov_new(1,1) = Cov(i,i);
    Cov_new(2,2) = Cov(i+(featureDIM/(dynamic_flag+1)),...
        i+(featureDIM/(dynamic_flag+1)));
    Cov_new(3,3) = Cov(i+(featureDIM/(dynamic_flag+1))*2,...
        i+(featureDIM/(dynamic_flag+1))*2);                
    [converted_seq(i,:)] = generalized_MLPG_ver1(Input_seq_new,Cov_new,dynamic_flag); % mcc_dim by lenght, using dynamic features     
end
converted_seq_mlpg=converted_seq;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [output_sequence]=generalized_MLPG_ver1(Input_seq,Cov,dynamic_flag)
%% Input parameters:
% Input_seq: Input sequence obtained by ANN-mapping, 
% e.g., 3D-by-T -> ( Input_seq = (static+delta+delta^2) by T ), where D = dimension of static/delta/delta^2, T is total
% number of input frames. 

% Cov: A single Gaussian estimated covariance matrix, e.g., 3D by 3D 
% dynamic_flag: 1 -> using delta, 2-> using delta^2;

%% Output parameter:
% output_sequence: Generating smoothed output sequence


%% 1. Intial values setup for MLPG algorithm
sequence_length = size(Input_seq,2);
front_term=0;
back_term=0;
if dynamic_flag ==1
   static_dim=size(Input_seq,1)/2;
   w=[[0*eye(static_dim) 1*eye(static_dim) 0*eye(static_dim)];[-0.5*eye(static_dim) 0*eye(static_dim) 0.5*eye(static_dim)]]; % delta
else % dynamic_flag == 2
   static_dim=size(Input_seq,1)/3;
   w=[[0*eye(static_dim) 1*eye(static_dim) 0*eye(static_dim)];[-0.5*eye(static_dim) 0*eye(static_dim) 0.5*eye(static_dim)];[1*eye(static_dim) -2*eye(static_dim) 1*eye(static_dim)]]; %delta^2
end
w=sparse(w);

CCov_seq = zeros(size(Input_seq,1),size(Input_seq,1),sequence_length); % covariance sequence
for i = 1:sequence_length
    CCov_seq(:,:,i) = Cov(:,:);
end 


%% 2. MLPG algorithm
for j=1:sequence_length
       if j==1
          if dynamic_flag==1
           W=[w(:,static_dim+1:end),zeros(2*static_dim,(sequence_length-2)*static_dim)]; % delta
          elseif dynamic_flag==2
           W=[w(:,static_dim+1:end),zeros(3*static_dim,(sequence_length-2)*static_dim)]; % delta^2                   
          else
           W=[w(:,static_dim+1:end),zeros(2*static_dim,(sequence_length-2)*static_dim)]; % delta
          end
       elseif j==sequence_length
          if dynamic_flag==1
           W=[zeros(2*static_dim,(sequence_length-2)*static_dim),w(:,1:2*static_dim)]; % delta
          elseif dynamic_flag==2
           W=[zeros(3*static_dim,(sequence_length-2)*static_dim),w(:,1:2*static_dim)]; % delta^2     
          else
           W=[zeros(2*static_dim,(sequence_length-2)*static_dim),w(:,1:2*static_dim)]; % delta
          end           
       else
          if dynamic_flag==1
           W=[zeros(2*static_dim,(j-2)*static_dim),w,zeros(2*static_dim,(sequence_length-3-j+2)*static_dim)]; % delta
          elseif dynamic_flag==2
           W=[zeros(3*static_dim,(j-2)*static_dim),w,zeros(3*static_dim,(sequence_length-3-j+2)*static_dim)]; % delta^2    
          else
           W=[zeros(2*static_dim,(j-2)*static_dim),w,zeros(2*static_dim,(sequence_length-3-j+2)*static_dim)]; % delta
          end                  
       end
                E=Input_seq(:,j);
                D=CCov_seq(:,:,j);
                D=sparse(D);
                Q=W'*(D^-1);
                front_term=sparse(front_term)+Q*W;         
                back_term=sparse(back_term)+Q*E;
end
            est_t=front_term\back_term;
            output_sequence=full(reshape(est_t,static_dim,sequence_length));
            
