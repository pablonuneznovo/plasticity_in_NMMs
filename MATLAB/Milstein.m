% BTSP (Milstein et al. 2021) - Exact Weight-Dependent Model
% Implements Eqs. 8, 9, 16 from the paper.
clear; clc; close all;

% --- Parameters (from Milstein et al. 2021 Results) ---
tau_ET = 1000;      % ms (Eligibility Trace decay)
tau_IS = 500;       % ms (Instructive Signal decay)
W_max = 4.0;        % Maximum weight
k_plus = 2.0;       % Potentiation rate
k_minus = 0.5;      % Depression rate (lower than potentiation)

% Sigmoid Parameters for q+ and q- (Eqs 10-13)
alpha_plus = 0.2; beta_plus = 30; 
alpha_minus = 0.1; beta_minus = 30; % Depression has lower threshold

% --- Simulation Settings ---
T = 6000; dt = 1; time = 1:dt:T;

% 1. Input Spikes (Eligibility)
t_spike = 1500;
spikes = zeros(size(time)); spikes(t_spike) = 1;

% 2. Plateau Potential (Instructive Signal)
t_plat_onset = 2500; dur_plat = 300;
Plateau_Gate = zeros(size(time)); 
Plateau_Gate(t_plat_onset : t_plat_onset+dur_plat) = 1;

% --- Integration of Signal Dynamics ---
ET = zeros(size(time));
IS = zeros(size(time));

for t = 2:T
    % Eq. 8: Eligibility Trace
    dET = -ET(t-1)/tau_ET + spikes(t-1);
    ET(t) = ET(t-1) + dET*dt;
    
    % Eq. 9: Instructive Signal
    dIS = -IS(t-1)/tau_IS + Plateau_Gate(t-1);
    IS(t) = IS(t-1) + dIS*dt;
end
% Normalize IS for calculating overlap (conceptually)
IS = IS / max(IS); 

% Calculate Signal Overlap
Overlap = ET .* IS;

% --- Integration of Synaptic Weight (Eq. 16) ---
% We simulate 3 synapses with different INITIAL weights to show the rule
W_weak   = zeros(size(time)); W_weak(1) = 0.5;
W_strong = zeros(size(time)); W_strong(1) = 3.5;

for t = 2:T
    % q functions (Sigmoidal dependence on Overlap)
    % Using generic sigmoid function s(x) = 1 / (1 + exp(-beta*(x-alpha)))
    q_plus  = 1 / (1 + exp(-beta_plus  * (Overlap(t-1) - alpha_plus)));
    q_minus = 1 / (1 + exp(-beta_minus * (Overlap(t-1) - alpha_minus)));
    
    % Eq. 16: Weight Dynamics
    % Weak Synapse
    dW_w = (W_max - W_weak(t-1))*k_plus*q_plus - W_weak(t-1)*k_minus*q_minus;
    W_weak(t) = W_weak(t-1) + dW_w * dt * 0.001; % Scale time
    
    % Strong Synapse
    dW_s = (W_max - W_strong(t-1))*k_plus*q_plus - W_strong(t-1)*k_minus*q_minus;
    W_strong(t) = W_strong(t-1) + dW_s * dt * 0.001;
end

% --- PLOTTING ---
figure('Color','w', 'Position', [100 100 600 800]); 

% Panel 1: Signals
subplot(3,1,1); hold on;
plot(time, ET, 'Color', [0.2 0.6 0.8], 'LineWidth', 2);
plot(time, IS, 'Color', [0.8 0.3 0.3], 'LineWidth', 2);
fill([t_plat_onset t_plat_onset+dur_plat t_plat_onset+dur_plat t_plat_onset], ...
     [0 0 1 1], [0.8 0.3 0.3], 'FaceAlpha', 0.1, 'EdgeColor', 'none');
legend({'Eligibility Trace', 'Instructive Signal'}, 'Box', 'off');
ylabel('Amplitude'); xlim([0 T]);
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XTick', [],'FontSize', 16);

% Panel 2: Overlap
subplot(3,1,2);
area(time, Overlap, 'FaceColor', [0.1 0.5 0.3], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
ylabel('Overlap'); xlim([0 T]); ylim([0 0.4]);
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XTick', [],'FontSize', 16);

% Panel 3: Weight Change
subplot(3,1,3); hold on;
plot(time, W_weak, 'Color', [0.2 0.6 0.2], 'LineWidth', 2.5); % Potentiation
plot(time, W_strong, 'Color', [0.8 0.2 0.2], 'LineWidth', 2.5); % Depression
yline(W_max, ':', 'Color', [0.5 0.5 0.5]);

xlabel('Simulation Time', 'FontSize', 10);
ylabel('Synaptic Weight');
legend({'Weak \rightarrow Potentiates', 'Strong \rightarrow Depresses'}, 'Box', 'off', 'Location', 'East');
xlim([0 T]); ylim([0 4.5]);
set(gca, 'LineWidth', 1.2, 'Box', 'off','FontSize', 16);