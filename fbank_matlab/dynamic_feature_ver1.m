% 2013.11.25 constructed by Hwang, Hsin-Te: 
% 2013.11.25 modified by Hwang, Hsin-Te
% Computing static and dynamic features

function output_sequence=dynamic_feature_ver1(static_feature_seq,dynamic_flag)
% static_feature_seq: static feature sequence
% dynamic_flag: 1=>delta, 2=> delta^2
% output_sequence: joint static and dynamic feature sequence
output_sequence=[];
[mcc_dim seq_length]=size(static_feature_seq);
if dynamic_flag==1
    w=[[0*eye(mcc_dim) 1*eye(mcc_dim) 0*eye(mcc_dim)];[-0.5*eye(mcc_dim) 0*eye(mcc_dim) 0.5*eye(mcc_dim)]];
elseif dynamic_flag==2
    w=[[0*eye(mcc_dim) 1*eye(mcc_dim) 0*eye(mcc_dim)];[-0.5*eye(mcc_dim) 0*eye(mcc_dim) 0.5*eye(mcc_dim)];[1*eye(mcc_dim) -2*eye(mcc_dim) 1*eye(mcc_dim)]];
else
    exit(1);
end         

for i=1:seq_length
    if i==1
       static=zeros(mcc_dim,3);
       static(:,2)=static_feature_seq(:,i);
       static(:,3)=static_feature_seq(:,i+1);
    elseif i==size(static_feature_seq,2)
       static=zeros(mcc_dim,3);
       static(:,1)=static_feature_seq(:,i-1);
       static(:,2)=static_feature_seq(:,i);           
    else
       static=static_feature_seq(:,i-1:i+1);
    end
      len=size(static,2);    
      y=reshape(static,mcc_dim*len,1);
      dynamic_feature_vector=w*y;    
      output_sequence=[output_sequence,dynamic_feature_vector];
end      