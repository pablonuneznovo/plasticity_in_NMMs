% Panel N: Stasinski et al. (2024) - Whole-Brain dFIC Dynamics
% Simulation Time: 12 seconds (Focused on the transient convergence)
% Reference: Stasinski et al. (2024) PLoS Comput Biol.
clear; clc; close all;

% --- 1. Load Connectome ---
try
    load('SC_sample.mat', 'C'); 
    C = double(C); 
    C(1:size(C,1)+1:end) = 0; 
    C = C / max(C(:)); 
catch
    % Fallback if file missing
    N = 90; C = rand(N); C = C - diag(diag(C)); C = C/max(C(:));
end
N = size(C, 1);

% --- 2. Jansen-Rit Parameters (Table 1) ---
A = 3.25;       % Max amplitude EPSP (mV)
B = 22.0;       % Max amplitude IPSP (mV)
a = 0.1;        % Inv. time constant excitatory (1/ms)
b = 0.05;       % Inv. time constant inhibitory (1/ms)

C1 = 135;       
C2 = 108;       % 0.8 * C1
C3 = 33.75;     % 0.25 * C1
C4 = 33.75;     % 0.25 * C1

v_max = 0.0025; % Max firing rate (1/ms)
v_0   = 6.0;    % Firing threshold (mV)
r     = 0.56;   % Steepness (1/mV)

% Global Parameters
G = 21.0;       % Optimal range from paper
I_mean = 0.11;  % Sub-bistable boundary input

% --- 3. dFIC Control Parameters ---
eta    = 0.005; % Learning rate
tau_d  = 1000;  % Detector time constant (ms)
Target = 0.01;  % Target y0 (mV)

% --- 4. Simulation Setup ---
dt = 0.5;
T_total = 12000; % 12 Seconds (Zoomed in)
Steps = T_total / dt;

% State Variables (N x 6)
y = rand(N, 6) * 0.01; 

% dFIC Variables
wFIC = ones(N, 1) * 1.0;    % Start at 1.0
y0_d = zeros(N, 1);         
y2_d = zeros(N, 1);         

% History Arrays
Hist_y0   = zeros(N, Steps/50); 
Hist_wFIC = zeros(N, Steps/50);
Hist_Time = zeros(1, Steps/50);
idx_plot = 1;

% Identify Hub and Leaf for visualization
Deg = sum(C, 2);
[~, idx_Hub]  = max(Deg);
[~, idx_Leaf] = min(Deg);

fprintf('Running Stasinski dFIC (G=%.1f, Target=%.2f mV)...\n', G, Target);

% --- 5. Simulation Loop ---
for t = 1:Steps
    
    % Sigmoid Helper Function
    sig = @(v) 2 * v_max ./ (1 + exp(r * (v_0 - v)));
    
    % Input Calculation
    sig_y1_y2 = sig(y(:,2) - y(:,3)); 
    Network_Input = G * (C * sig_y1_y2);
    Total_Input   = I_mean + Network_Input; 
    
    % Current State
    y0 = y(:,1); y1 = y(:,2); y2 = y(:,3);
    y3 = y(:,4); y4 = y(:,5); y5 = y(:,6);
    
    % Derivatives (Jansen-Rit)
    % Eq 1B modified: S[y1 - wFIC * y2]
    dy3 = A * a * sig(y1 - wFIC .* y2) - 2*a*y3 - a^2*y0; 
    dy4 = A * a * (Total_Input + C2 * sig(C1 * y0)) - 2*a*y4 - a^2*y1; 
    dy5 = B * b * (C4 * sig(C3 * y0)) - 2*b*y5 - b^2*y2; 
    
    % Integration (Euler)
    y(:,1) = y(:,1) + y(:,4) * dt;      
    y(:,2) = y(:,2) + y(:,5) * dt;      
    y(:,3) = y(:,3) + y(:,6) * dt;      
    y(:,4) = y(:,4) + dy3 * dt;
    y(:,5) = y(:,5) + dy4 * dt;
    y(:,6) = y(:,6) + dy5 * dt;
    
    % dFIC Detector Dynamics (Eq 4A, 4B)
    dy0_d = (y(:,1) - y0_d) / tau_d;
    y0_d  = y0_d + dy0_d * dt;
    
    dy2_d = (y(:,3) - y2_d) / tau_d;
    y2_d  = y2_d + dy2_d * dt;
    
    % dFIC Weight Update (Eq 4C)
    dwFIC = eta * y2_d .* (y0_d - Target);
    wFIC  = wFIC + dwFIC * dt;
    
    % Store Data
    if mod(t, 50) == 0
        Hist_y0(:, idx_plot)   = y(:,1); 
        Hist_wFIC(:, idx_plot) = wFIC;
        Hist_Time(idx_plot)    = t * dt / 1000; 
        idx_plot = idx_plot + 1;
    end
end

% --- 6. PLOTTING ---
figure('Color','w', 'Position', [50 50 800 800]);

% Trim history
Hist_Time = Hist_Time(1:idx_plot-1);
Hist_wFIC = Hist_wFIC(:, 1:idx_plot-1);
Hist_y0   = Hist_y0(:, 1:idx_plot-1);

% PANEL A: Whole-Network Weight Adaptation
subplot(2,1,1); hold on;

% 1. Plot Background (All Nodes) - Faint Grey
plot(Hist_Time, Hist_wFIC', 'Color', [0.7 0.7 0.7, 0.4], 'LineWidth', 0.5); 

% 2. Plot Highlights (Hub/Leaf) - Thick Colors
plot(Hist_Time, Hist_wFIC(idx_Hub, :), 'Color', [0.5 0 0.5], 'LineWidth', 2.5); % Purple
plot(Hist_Time, Hist_wFIC(idx_Leaf, :), 'Color', [0.4 0.8 0.4], 'LineWidth', 2.5); % Green

% 3. Dummy Handles for Correct Legend Colors
h_bg   = plot(nan, nan, 'Color', [0.6 0.6 0.6], 'LineWidth', 1);
h_hub  = plot(nan, nan, 'Color', [0.5 0 0.5], 'LineWidth', 2.5);
h_leaf = plot(nan, nan, 'Color', [0.4 0.8 0.4], 'LineWidth', 2.5);

%title('A. Adaptation of Inhibitory Weights (wFIC)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Inhibitory Weight (wFIC)', 'FontSize', 11);
xlabel('Simulation Time', 'FontSize', 11);
legend([h_bg, h_hub, h_leaf], {'All Nodes', 'Hub Node', 'Leaf Node'}, ...
    'Location', 'Best', 'Box', 'off');
grid on; set(gca, 'Box', 'off', 'LineWidth', 1.2);
xlim([0 12]); 

% PANEL B: PSP Convergence
subplot(2,1,2); hold on;

% 1. Plot Background
plot(Hist_Time, Hist_y0', 'Color', [0.7 0.7 0.7, 0.4], 'LineWidth', 0.5);

% 2. Plot Highlights
plot(Hist_Time, Hist_y0(idx_Hub, :), 'Color', [0.5 0 0.5], 'LineWidth', 2.0);
plot(Hist_Time, Hist_y0(idx_Leaf, :), 'Color', [0.4 0.8 0.4], 'LineWidth', 2.0);

% 3. Target Line
yline(Target, 'k--', 'LineWidth', 2, 'Label', 'Target 0.01 mV');

%title('B. Convergence of Pyramidal Activity (y_0)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Pyramidal PSP (mV)', 'FontSize', 11); 
xlabel('Simulation Time', 'FontSize', 11);
grid on; set(gca, 'Box', 'off', 'LineWidth', 1.2);
xlim([0 12]); 

% Fix Y-Axis: Ensure initial transient is visible
max_val = max(Hist_y0(:));
ylim([0 max_val * 1.1]); 

%sgtitle('Stasinski et al. (2024): Whole-Brain dFIC Dynamics', 'FontSize', 14, 'FontWeight', 'bold');