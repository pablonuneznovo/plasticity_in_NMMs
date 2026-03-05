% Mean-field approximations of QIF networks with STP
% Multi-population Approximation (FRE_MPA) - Gast et al. (2021)

clear; clc; close all;

% --- Colour Palette ---
c_Rate    = [0.2 0.2 0.2];     % Dark Grey (r)
c_Input   = [0.6 0.6 0.6];     % Light Grey (Input Pulse)
c_Volt    = [0.3 0.5 0.3];     % Muted Green (v)
c_Facil   = [0.8 0.3 0.3];     % Red (u) - Utilization
c_Depress = [0.2 0.4 0.8];     % Blue (x) - Resources
c_Memory  = [0.6 0.4 0.8];     % Muted Purple (Effective Weight)
c_Ax      = [0.15 0.15 0.15];  % Soft Black

%% 1. Global Parameters (Fig 5/6 configuration)
P.M       = 100;               % Number of subpopulations
P.tau     = 1.0;               % Membrane time constant
P.Delta   = 2.0;               % HWHM of global eta distribution
P.eta_bar = -3.0;              % Center of global eta distribution
P.J       = 15.0 * sqrt(P.Delta); % Global coupling strength
P.tau_x   = 50.0;              % Depression time constant
P.tau_u   = 20.0;              % Facilitation time constant
P.U0      = 0.2;               % Baseline synaptic efficacy
P.alpha   = 0.1;               % Depression strength

% 2. Subpopulation Parameterization (Eq. 22a, 22b)
m_idx = 1:P.M;
P.eta_m = P.eta_bar + P.Delta * tan(pi * (2*m_idx - P.M - 1) / (2*(P.M + 1)));
P.Delta_m = P.Delta * (tan(pi * (2*m_idx - P.M - 0.5) / (2*(P.M + 1))) - ...
            tan(pi * (2*m_idx - P.M - 1.5) / (2*(P.M + 1))));

%% 3. Simulation Setup
P.I_amp    = 3.0;              % Input pulse amplitude 
P.t1_start = 20.0; P.t1_end = 40.0;
P.t2_start = 60.0; P.t2_end = 80.0;

tspan = [0 100];
% Initial conditions: [r_m; v_m; x_m; u_m] x M
y0 = [repmat(0.2, P.M, 1); repmat(-1.5, P.M, 1); repmat(0.5, P.M, 1); repmat(0.3, P.M, 1)];

options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
[t, y] = ode45(@(t,y) gast_mpa_ode(t, y, P), tspan, y0, options);

% Average across subpopulations for plotting 
r_avg = mean(y(:, 1:P.M), 2);
v_avg = mean(y(:, P.M+1 : 2*P.M), 2);
x_avg = mean(y(:, 2*P.M+1 : 3*P.M), 2);
u_avg = mean(y(:, 3*P.M+1 : 4*P.M), 2);

% Reconstruct Stimulus Vector for Plotting
I_S = zeros(size(t));
I_S((t >= P.t1_start & t <= P.t1_end) | (t >= P.t2_start & t <= P.t2_end)) = P.I_amp;

%% 4. Visualization
figure('Color', 'w', 'Position', [100, 100, 700, 900]);

% --- Panel A: Firing Rate & Input Pulse ---
subplot(3,1,1);

% Right Axis: Input Stimulus (Background Area)
yyaxis right;
area(t, I_S, 'FaceColor', c_Input, 'EdgeColor', 'none', 'FaceAlpha', 0.2);
ylabel('Input I(t)', 'FontSize', 16, 'Color', c_Input);
ylim([0 P.I_amp * 1.5]); 
set(gca, 'YColor', c_Input, 'FontSize', 14);

% Left Axis: Firing Rate (Foreground Line)
yyaxis left;
plot(t, r_avg, 'Color', c_Rate, 'LineWidth', 2.5);
ylabel('Firing Rate r', 'FontSize', 16, 'Color', c_Rate);
ylim([0 1]); xlim([0 100]);
set(gca, 'YColor', c_Rate, 'FontSize', 14, 'Box', 'off', 'XTickLabel', [], 'LineWidth', 1.5);

% --- Panel B: Membrane Potential ---
subplot(3,1,2);
plot(t, v_avg, 'Color', c_Volt, 'LineWidth', 2.0);
ylabel('Potential v', 'FontSize', 16);
ylim([-2 0.5]); xlim([0 100]);
set(gca, 'FontSize', 14, 'Box', 'off', 'XTickLabel', [], 'LineWidth', 1.5, 'YColor', c_Ax);

% --- Panel C: Combined Synaptic Dynamics (Dual Axis) ---
subplot(3,1,3);

% 1. Effective Weight (Background Shade)
eff_weight = u_avg .* x_avg;
yyaxis left;
hold on;
area(t, eff_weight, 'FaceColor', c_Memory, 'EdgeColor', 'none', 'FaceAlpha', 0.15);

% 2. Left Axis: Resources (x) - Blue
plot(t, x_avg, 'Color', c_Depress, 'LineWidth', 2.5);
ylabel('Resources (x)', 'FontSize', 16, 'Color', c_Depress);
ylim([0.2 0.8]); 
set(gca, 'YColor', c_Depress, 'FontSize', 14, 'LineWidth', 1.5);

% 3. Right Axis: Utilization (u) - Red
yyaxis right;
plot(t, u_avg, 'Color', c_Facil, 'LineWidth', 2.5);
ylabel('Utilization (u)', 'FontSize', 16, 'Color', c_Facil);
ylim([0.2 0.8]); 
set(gca, 'YColor', c_Facil, 'FontSize', 14, 'LineWidth', 1.5);

xlabel('Simulation Time', 'FontSize', 16);
xlim([0 100]);
grid off;

%% 5. MPA ODE Function
function dydt = gast_mpa_ode(t, y, P)
    M = P.M;
    r = y(1:M);
    v = y(M+1 : 2*M);
    x = y(2*M+1 : 3*M);
    u = y(3*M+1 : 4*M);
    
    I_ext = 0;
    if (t >= P.t1_start && t <= P.t1_end) || (t >= P.t2_start && t <= P.t2_end)
        I_ext = P.I_amp;
    end
    
    % Effective network input (Weighted sum across subpopulations)
    r_eff = (P.J * P.tau / M) * sum(x .* u .* r);
    
    % Dynamics for each subpopulation m
    drdt = (P.Delta_m' / (pi * P.tau) + 2 * r .* v) / P.tau; 
    dvdt = (v.^2 + P.eta_m' + I_ext + r_eff - (pi * r * P.tau).^2) / P.tau; 
    dxdt = (1 - x) / P.tau_x - P.alpha * u .* x .* r; 
    dudt = (P.U0 - u) / P.tau_u + P.U0 * (1 - u) .* r; 
    
    dydt = [drdt; dvdt; dxdt; dudt];
end