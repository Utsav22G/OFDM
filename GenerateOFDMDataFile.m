%% Establish Parameters
N_Training = 16;    % Sets of Training Data 
N_XData = 84;       % Sets of True Data 
Bits_per_set = 48;  % Bits per set of data
size_training_data = Bits_per_set*N_Training;       % Size of Training Data - 16 sets of 64 bits: 1024
size_X = Bits_per_set*N_XData;                      % Size of data - 84 sets of 64 bits: 5376 bits

%% Generate Data
range = [-1 1]; % List of wanted values: 
a = randi(2,size_X,1); %Randomly select index
b = randi(2,size_training_data,1); 

X_k = transpose(range(a)); %Creates array with range(a) values
X_train = transpose(range(b)); % Creates Training Data
message = [X_train ; X_k]; %Combines data



%%
for ii = numel(I_bit):-1:1
    Data(2*ii-[0 1]) = [Q_bit(ii), I_bit(ii)];
end
%%
Symbol_period = 50;

pulse = ones(Symbol_period, 1);

% spread out the values in "bits" by Symbol_period
% first create a vector to store the values
x = zeros(Symbol_period*length(Data),1);

% assign every Symbol_period-th sample to equal a value from bits
x(1:Symbol_period:end) = Data;

% now convolve the single generic pulse with the spread-out bits
x_tx = conv(pulse, x);

%%
% first create a vector to store the interleaved real and imaginary values
tmp = zeros(length(x_tx)*2, 1);

% then assign the real part of x_tx to every other sample and the imaginary
% part to the remaining samples. In this example, the imaginary parts are
% all zero since our original signal is purely real, but we still have to
% write the zero values 
tmp(1:2:end) = real(x_tx);
tmp(2:2:end) = imag(x_tx);

target_begin = ones(10000, 1);
target_end = ones(10000,1);

tmp = cat(1, target_begin, tmp(1:2:end));
tmp = cat(1, tmp(1:2:end),target_end);
figure
stem(tmp)
%%
% open a file to write in binary format 
f1 = fopen('tx3.dat', 'wb');
% write the values as a float32
fwrite(f1, tmp/2, 'float32');
% close the file
fclose(f1)