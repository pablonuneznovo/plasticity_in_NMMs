% Reproduction of Schirner et al. (2023) Figure 3a
% "Learning how network structure shapes decision-making"

clearvars -except EI_Tuning_Mechanism; clc; close all;

% --- 1. Global Parameters ---
P.a_E = 310; P.b_E = 125; P.d_E = 0.16;
P.a_I = 615; P.b_I = 177; P.d_I = 0.087;
P.tau_E = 100;  P.tau_I = 10;    % (ms)
P.gamma = 0.641/1000;
P.w_plus = 1.4;
P.J_NMDA = 0.15;
P.W_E = 1.0; P.W_I = 0.7;
P.I_0 = 0.382;
P.dt = 1.0;
P.T_sim = 4 * 60 * 1000;         % 4 mins to allow BOLD to stabilize
P.TR = 720;
P.trials = 100;                   % Averaging factor

% --- Balloon-Windkessel (Friston 2003 Parameters) ---
BW.tau_s = 0.65; BW.tau_f = 0.41; BW.tau_0 = 0.98; BW.alpha = 0.32;
BW.E0 = 0.34; BW.V0 = 0.02;
BW.k1 = 7*BW.E0; BW.k2 = 2.0; BW.k3 = 2*BW.E0 - 0.2;

% --- 2. Conditions ---
% Use fewer points for simulation (speed), but interpolate many for plot
EI_Ratios_Sim = logspace(-2, 2, 12);

% Panel A: Noise Sweep
Noises = [0.001, 0.005, 0.01, 0.025, 0.05];
G_Fixed = 1.0;

% Panel B: Coupling Sweep
Couplings = [0.01, 0.1, 0.2, 0.5, 1.0];
Noise_Fixed = 0.01;

% --- 3. Run Simulation ---
fprintf('Running Simulation (Smoothing active)...\n');
FC_Noise = run_sweep(P, BW, EI_Ratios_Sim, Noises, G_Fixed, 'Noise');
FC_Coup  = run_sweep(P, BW, EI_Ratios_Sim, Couplings, Noise_Fixed, 'Coupling');

% --- 4. Plotting with Spline Interpolation ---
plot_smooth_curves(EI_Ratios_Sim, FC_Noise, FC_Coup, Noises, Couplings);

% =========================================================================
% SWEEP MANAGER
% =========================================================================
function FC_Matrix = run_sweep(P, BW, Ratios, VarParam, FixedParam, Mode)
FC_Matrix = zeros(length(VarParam), length(Ratios));
for i = 1:length(VarParam)
    if strcmp(Mode, 'Noise'); sigma = VarParam(i); G = FixedParam;
    else; G = VarParam(i); sigma = FixedParam; end
    
    fprintf('   Simulating %s level %d/%d...\n', Mode, i, length(VarParam));
    
    parfor r = 1:length(Ratios)
        ratio = Ratios(r);
        w_FFI = 1 / (1 + ratio);
        w_LRE = ratio * w_FFI;
        
        % Averaging Loop
        acc = 0;
        for t = 1:P.trials
            acc = acc + simulate_trial(P, BW, w_LRE, w_FFI, G, sigma);
        end
        FC_Matrix(i,r) = acc / P.trials;
    end
end
end

% SPLINE INTERPOLATION
function plot_smooth_curves(X_Data, Y_Noise, Y_Coup, Noises, Couplings)
figure('Color','w', 'Position', [100 100 500 800]);

% High-resolution X-axis for interpolation
X_Smooth = logspace(log10(min(X_Data)), log10(max(X_Data)), 200);

% Colors matched to paper (Blue -> Green -> Red -> Grey)
cmap = [0.0 0.0 0.6; 0.2 0.6 1.0; 0.9 0.7 0.0; 0.0 0.6 0.0; 0.8 0.0 0.0; 0.5 0.5 0.5];
idx_noise = [1 2 3 4 5]; idx_coup = [1 2 3 5 6];

% --- Panel A: Noise ---
subplot(2,1,1); hold on;
for i = 1:length(Noises)
    % The "Trick": Spline Interpolation
    Y_Smooth = interp1(log10(X_Data), Y_Noise(i,:), log10(X_Smooth), 'makima');
    
    plot(X_Smooth, Y_Smooth, 'LineWidth', 2.5, 'Color', cmap(idx_noise(i),:));
end
format_panel('Effect of Noise \sigma', Noises);

% --- Panel B: Coupling ---
subplot(2,1,2); hold on;
for i = 1:length(Couplings)
    Y_Smooth = interp1(log10(X_Data), Y_Coup(i,:), log10(X_Smooth), 'makima');
    
    plot(X_Smooth, Y_Smooth, 'LineWidth', 2.5, 'Color', cmap(idx_coup(i),:));
end
format_panel('Effect of Coupling G', Couplings);

end

function format_panel(TitleStr, LegendVals)
set(gca, 'XScale', 'log', 'XLim', [0.01 100], 'YLim', [-1.05 1.05]);
yline(0, 'k:', 'LineWidth', 1);
xlabel('E/I Ratio','FontSize', 12); ylabel('FC','FontSize', 12);
title(TitleStr, 'FontSize', 12, 'FontWeight', 'bold');
legend(string(LegendVals), 'Location', 'SouthEast', 'Box', 'off');
grid on; axis square; set(gca,'Box','off','LineWidth',1.2);
end

% =========================================================================
% CORE SIMULATION
% =========================================================================
function fc = simulate_trial(P, BW, w_LRE, w_FFI, G, sigma)
% Steady State FIC (Analytic)
r_targ = 4.0; I_targ_val = 0.3855;
S_E_ss = (P.tau_E * P.gamma * r_targ) / (1 + P.tau_E * P.gamma * r_targ);
S_I_ss = (P.tau_I * P.gamma * r_targ) / (1 + P.tau_I * P.gamma * r_targ);

J_i = (P.W_E*P.I_0 + P.w_plus*P.J_NMDA*S_E_ss + G*P.J_NMDA*w_LRE*S_E_ss - I_targ_val) / S_I_ss;
J_i = max(0, J_i);

% Init
steps = floor(P.T_sim / P.dt);
S_E = zeros(2,1) + S_E_ss; S_I = zeros(2,1) + S_I_ss;
X_BW = repmat([0, 1, 1, 1], 2, 1); % s,f,v,q

% Buffer
BOLD = zeros(2, floor(steps/(P.TR/P.dt))); b_idx = 1;
sqrt_dt = sqrt(P.dt); dt_sec = P.dt/1000;
C_mat = [0 1; 1 0];

for k = 1:steps
    noise = randn(2,1) * sigma * sqrt_dt;
    
    % DMF
    I_E = P.W_E*P.I_0 + P.w_plus*P.J_NMDA*S_E + G*P.J_NMDA*(w_LRE.*(C_mat*S_E)) - J_i.*S_I;
    I_I = P.W_I*P.I_0 + P.J_NMDA*S_E          + G*P.J_NMDA*(w_FFI.*(C_mat*S_E)) - S_I;
    
    r_E = (P.a_E*I_E - P.b_E)./(1-exp(-P.d_E*(P.a_E*I_E - P.b_E))); r_E(isnan(r_E))=0;
    r_I = (P.a_I*I_I - P.b_I)./(1-exp(-P.d_I*(P.a_I*I_I - P.b_I))); r_I(isnan(r_I))=0;
    
    S_E = S_E + P.dt*(-S_E/P.tau_E + (1-S_E).*P.gamma.*r_E + noise);
    S_I = S_I + P.dt*(-S_I/P.tau_I + P.gamma.*r_I + noise);
    
    % Balloon-Windkessel
    s = X_BW(:,1); f = X_BW(:,2); v = X_BW(:,3); q = X_BW(:,4);
    ds = S_E - s/BW.tau_s - (f-1)/BW.tau_f;
    df = s;
    dv = (f - v.^(1/BW.alpha))/BW.tau_0;
    dq = (f.*(1-(1-BW.E0).^(1./f))/BW.E0 - q.*v.^(1/BW.alpha-1))/BW.tau_0;
    X_BW = X_BW + [ds df dv dq]*dt_sec;
    
    if mod(k, P.TR/P.dt) == 0
        y = BW.V0*(BW.k1.*(1-q) + BW.k2.*(1-q./v) + BW.k3.*(1-v));
        if b_idx <= size(BOLD,2); BOLD(:,b_idx) = y; b_idx = b_idx+1; end
    end
end
washout = floor(60000/P.TR);
if size(BOLD,2) > washout; R = corr(BOLD(:,washout:end)'); fc = R(1,2); else; fc=0; end

end

