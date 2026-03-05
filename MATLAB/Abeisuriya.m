% Panel G: Abeysuriya et al. (2018) - ISP with Local Gating (Fixed)
% Rule: dc_ie/dt = eta * I(t) * (E(t) - rho)
clear; clc; close all;

% --- Visual Style ---
c_Exc    = [0.2 0.6 0.8];   % Teal
c_Inh    = [0.8 0.3 0.3];   % Red
c_Target = [0.5 0.5 0.5];   % Grey
c_Input  = [0.9 0.6 0.2];   % Orange
c_Ax     = [0.15 0.15 0.15];% Soft Black

% --- Simulation Parameters ---
T_sim = 2000; dt = 0.5; time = 0:dt:T_sim;

% Wilson-Cowan Parameters
tau = 10; 
c_ee = 16; c_ei = 12; 
c_ii = 3; 
c_ie = zeros(size(time)); c_ie(1) = 10;

% ISP Parameters
rho = 0.15;     % Target Rate
eta = 0.1;      % Learning Rate

% Input Perturbation
P = ones(size(time)) * 1.5; 
P(time > 800) = 4.0; % Strong Step Input

% State Variables
E = zeros(size(time)); E(1) = rho;
I = zeros(size(time)); I(1) = 0.1;

% Sigmoid Function
S = @(x) 1 ./ (1 + exp(-x));

% --- Integration ---
for t = 2:length(time)
    % 1. Neural Dynamics
    dE = (-E(t-1) + S(c_ee*E(t-1) - c_ie(t-1)*I(t-1) + P(t-1))) / tau;
    dI = (-I(t-1) + S(c_ei*E(t-1) - c_ii*I(t-1))) / tau;
    
    E(t) = E(t-1) + dE * dt;
    I(t) = I(t-1) + dI * dt;
    
    % 2. Local Plasticity Rule
    % Note: I(t) gates the plasticity (Hebbian-like local rule)
    dc_ie = eta * I(t-1) * (E(t-1) - rho);
    c_ie(t) = c_ie(t-1) + dc_ie * dt;
    
    % Safety: Prevent negative weights (Biophysical constraint)
    if c_ie(t) < 0
        c_ie(t) = 0;
    end
end

% --- PLOTTING ---
figure('Color','w', 'Position', [100 100 500 600]);

% Panel A: Excitatory Rate
subplot(3,1,1); hold on;
yline(rho, '--', 'Color', c_Target, 'LineWidth', 1.5);
plot(time, E, 'Color', c_Exc, 'LineWidth', 2);
text(100, rho+0.1, 'Target \rho', 'Color', c_Target, 'FontSize', 10);
%title('Excitatory Rate E(t)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', c_Ax);
ylabel('Firing Rate', 'FontSize', 11, 'Color', c_Ax);

% FIX: Increased Limit to 1.1 because E(t) can spike up to 1.0
xlim([0 T_sim]); ylim([0 1.1]); 
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XColor', c_Ax, 'YColor', c_Ax, 'XTick', []);

% Panel B: Inhibitory Weight
subplot(3,1,2); hold on;
plot(time, c_ie, 'Color', c_Inh, 'LineWidth', 2.5);
%title('Inhibitory Weight c_{ie}(t)', 'FontSize', 12, 'FontWeight', 'bold', 'Color', c_Ax);
ylabel('Inhibitory Weight c_{ie}(t)', 'FontSize', 11, 'Color', c_Ax);
xlim([0 T_sim]); 
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XColor', c_Ax, 'YColor', c_Ax, 'XTick', []);

% Panel C: Input
subplot(3,1,3);
area(time, P, 'FaceColor', c_Input, 'EdgeColor', 'none', 'FaceAlpha', 0.2);
hold on; plot(time, P, 'Color', c_Input, 'LineWidth', 1.5);
%title('Input Perturbation', 'FontSize', 12, 'FontWeight', 'bold', 'Color', c_Ax);
xlabel('Simulation Time', 'FontSize', 11, 'Color', c_Ax);
ylabel('Input Perturbation', 'FontSize', 11, 'Color', c_Ax);
xlim([0 T_sim]); ylim([0 5]);
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XColor', c_Ax, 'YColor', c_Ax);

%sgtitle('G. Abeysuriya et al. (2018): ISP Response', 'FontSize', 13, 'FontWeight', 'bold', 'Color', c_Ax);