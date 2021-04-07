clc; clear all; close all;

% Creating X_k - Randomly choose between 1 and -1
size_training_data = 640;
size_X = 6400; %Size of data - 100 sets of 64 bits
range = [-1 1]; %List of wanted values
a = randi(2,size_X,1); %Randomly select index
b = randi(2,size_training_data,1); 
X_k = transpose(range(a)); %Creates array with range(a) values
X_train = transpose(range(b)); 

% IDFT - Takes inverse discrete fourier tranform of data
x_m = ifft(X_k); %Turns data points into impulses
x_train = ifft(X_train);

% Extending each point by 64 points
L = 64; %Length of Pulse - 64

pulse = ones(L, 1); %Creates pulse with L of 64 bits

x_0 = zeros(L*length(x_m),1);       % Creates "empty" vector that is length of desired data: 409600 bits
x_0(1:L:end) = x_m;                 % assign every Symbol_period-th sample to equal a value from bits
x_m_pulse = conv(pulse, x_0);       % Convolve vector by designed pulse
x_m_pulse= x_m_pulse(1:409600);     % Cut extra zeros caused by convolution - desired length of 409600 bits

x_t = zeros(L*length(x_train),1);       % Creates "empty" vector that is length of desired data: 409600 bits
x_t(1:L:end) = x_train;                 % assign every Symbol_period-th sample to equal a value from bits
x_train_pulse = conv(pulse, x_t);       % Convolve vector by designed pulse
x_train_pulse = x_train_pulse(1:L*length(x_train));     % Cut extra zeros caused by convolution - desired length of 409600 bits

figure(1)
stem(x_train_pulse)


% Plot data
figure;
stem(x_m_pulse)
hold on
ylabel('Amplitude')
xlabel('Bit')
title('Data Extended by Pulse')
hold off
%%  Adding cyclic prefix
n_data = size_X / L; % Calculates amount of sets of d-bit data chunks from original data

Len_pulse_data = length(x_m_pulse) + length(x_m_pulse)/4; % Calculates Desired final length of data w/ cyclic prefix

%Index 1st cyclic prefix - Done to ensure methodology works
cycPrefix = x_m_pulse(3073: 4096);                  % Determines index of first cyclic prefix - 1024 bits
x_m_load(1:1024) = cycPrefix;                       % Initializes array that will hold data w/ prefix, making first 1024 valus the cyclic prefix
x_m_load(1025:5120) = x_m_pulse(1: 4096);           % Loads first set of data, 64 bits -> 4096 bits 

%Index the Remaining Data
for R = 2:1:n_data   % Adds cyclic prefixes
    cycPrefix = x_m_pulse(4096*(R-1)+ 64*48+1: 4096*(R)); 
    x_m_load(80*64*(R-1)+1 : 5120*(R-1) + 1024) = cycPrefix;
    Post_chunk =  x_m_pulse(4096*(R-1)+1: 4096*(R));
    x_m_load(5120*(R-1) + 1024+1 : 5120*(R-1) + 1024 +4096) = Post_chunk;
end

% -------------------------TRAINING DATA--------------------------

Len_pulse_train = length(x_train_pulse) + length(x_train_pulse)/4; % Calculates Desired final length of data w/ cyclic prefix

%   For Training data Index 1st cyclic prefix - Done to ensure methodology works
cycPrefix_train = x_train_pulse(3073: 4096);                  % Determines index of first cyclic prefix - 1024 bits
x_train_load(1:1024) = cycPrefix;                       % Initializes array that will hold data w/ prefix, making first 1024 valus the cyclic prefix
x_train_load(1025:5120) = x_train_pulse(1: 4096);           % Loads first set of data, 64 bits -> 4096 bits 

%Index the Remaining Data
for R = 2:1:(size_training_data/L)   % Adds cyclic prefixes
    cycPrefix_train = x_train_pulse(4096*(R-1)+ 64*48+1: 4096*(R)); 
    x_train_load(80*64*(R-1)+1 : 5120*(R-1) + 1024) = cycPrefix_train;
    Post_chunk =  x_train_pulse(4096*(R-1)+1: 4096*(R));
    x_train_load(5120*(R-1) + 1024+1 : 5120*(R-1) + 1024 +4096) = Post_chunk;
end

%% Check Cyclic Prefix
% Downsample from x_m_load
Data_6400cyc = downsample(x_m_load,64);
%Visually Checked, Values are what they are supposed to be

%% Plot Data with cyclic Prefixes
stem(x_m_load)
hold on
title('Data with Cyclic Prefixes')
hold off

%% Transmit Data

%Transpose the Data - 1x51200 -> 51200x1
x_m_load = transpose(x_m_load);

% Transmit data through nonflat channel function
y_m = nonflat_channel(x_m_load);

% Determine Lag in Data
[y1x1,lag1] = xcorr(y_m,x_m_load); 
[~, Index1] = max(abs(y1x1));
shift1 = lag1(Index1);
%Comes out to be 9

%Cut Data from Lag
y_m = y_m(shift1:end);

% ---------------------TRAINING DATA----------------------------

%Transpose the Data - 1x51200 -> 51200x1
x_train_load = transpose(x_train_load);

% Transmit data through nonflat channel function
y_m_train = nonflat_channel(x_train_load);

% Determine Lag in Data
[y1x1_train,lag1] = xcorr(y_m_train,x_train_load); 
[~, Index1_train] = max(abs(y1x1_train));
shift1_train = lag1(Index1_train);
%Comes out to be 9

%Cut Data from Lag
y_m_train = y_m_train(shift1_train:end);

%% FFT - Fast Fourier Transform of Received Data
Y_k = fft(y_m,size_X);
Y_k_train = fft(y_m_train,size_training_data);

figure;
stem(Y_k);
hold on
title('Received Data')
hold off
%%  Channel Estimation 

% With OFDM, we are assuming flat fading model for every 64 bits of data,
% giving us 100 different transfer functions

H_k = zeros(100,1); %Empty vector to put transfer function values

for H = 1:1:n_data   % For each chunk of 64 -> 100 in total
    ChannelEst_n = mean(Y_k(1 + 64*(H-1): 64*(H))); % Calculates transfer function for every 64 bits (assuming flat fading model)
    H_k(H) = ChannelEst_n; %Place data into Transfer function matrix
end
figure;
stem(H_k);
hold on
title('Transfer Functions of Data');
hold off
%% Filtering out White Noise



%% Transmitted Vs Received Data
% Calculating Error

% diff = real((real(X_k) - (real(Y_k)./abs(real(Y_k)))))./real(X_k);
diff = real((real(X_k) - (real(Y_k)./abs(real(Y_k)))))./(2*real(X_k));
% num_error = length(diff(diff==2));
Error = abs(mean(diff)) *100

%Plotting Data
stem(x_m) % Plotting Transmitted Data
hold on
stem(Y_k)% Plotting Received Data
legend('Transmitted','Received')
hold off
