% Exact Neural Mass Model with Short-Term Plasticity (STP)
% Based on Taher et al., PLOS Computational Biology (2020)
% Implements Equations (14a-14d) and Figure 1 parameters.

clear; clc; close all;

% --- Colour Palette ---
c_Rate    = [0.2 0.2 0.2];      % Dark Grey (r)
c_Input   = [0.6 0.6 0.6];      % Light Grey (Input Pulse)
c_Volt    = [0.3 0.5 0.3];      % Muted Green (v)
c_Facil   = [0.8 0.3 0.3];      % Red (u) - Utilization
c_Depress = [0.2 0.4 0.8];      % Blue (x) - Resources
c_Memory  = [0.6 0.4 0.8];      % Muted Purple (Effective Weight)
c_Ax      = [0.15 0.15 0.15];   % Soft Black

%% 1. Model Parameters (Fig 1)
% Time constants are converted to seconds (paper uses ms, but rates are Hz)
P.tau_m   = 0.015;      % Membrane time constant (15 ms) [cite: 193]
P.tau_d   = 0.200;      % Depression time constant (200 ms) [cite: 149]
P.tau_f   = 1.500;      % Facilitation time constant (1500 ms) [cite: 149]

% Neural Parameters
P.Delta   = 0.25;       % HWHM of Lorentzian excitability [cite: 193]
P.H       = 0.0;        % Median excitability [cite: 193]
P.J       = 15.0;       % Synaptic weight strength [cite: 193]
P.I_B     = -1.0;       % Background current [cite: 193]

% STP Parameters
P.U0      = 0.2;        % Baseline utilization factor [cite: 149]

%% 2. Simulation Setup
% Stimulus: Two pulses as in Fig 1
P.I_stim_amp = 2.0;    
P.t1_start   = 0.2;  P.t1_end = 0.35; % First Pulse (0.15s duration)
P.t2_start   = 0.5;  P.t2_end = 0.65; % Second Pulse (after 0.15s gap)

tspan = [0 1.0]; % 1 second simulation
% Initial conditions: [r; v; x; u]
% Start at low activity equilibrium
y0 = [0.1; -2.0; 1.0; P.U0]; 

options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t, y] = ode45(@(t,y) taher_mass_ode(t, y, P), tspan, y0, options);

% Extract variables
r = y(:, 1);  % Firing Rate (Hz)
v = y(:, 2);  % Mean Membrane Potential
x = y(:, 3);  % Available Resources
u = y(:, 4);  % Utilization Factor

% Reconstruct Stimulus Vector for Plotting
I_S = zeros(size(t));
I_S((t >= P.t1_start & t <= P.t1_end) | (t >= P.t2_start & t <= P.t2_end)) = P.I_stim_amp;

%% 3. Visualization
figure('Color', 'w', 'Position', [100, 100, 700, 900]);

% --- Panel A: Firing Rate & Input Pulse (Dual Axis) ---
subplot(3,1,1);

% Right Axis: Input Stimulus (Background Area)
yyaxis right;
area(t, I_S, 'FaceColor', c_Input, 'EdgeColor', 'none', 'FaceAlpha', 0.2);
ylabel('Input Current', 'FontSize', 12, 'Color', c_Input);
ylim([0 P.I_stim_amp * 1.5]); 
set(gca, 'YColor', c_Input, 'FontSize', 12);

% Left Axis: Firing Rate (Foreground Line)
yyaxis left;
plot(t, r, 'Color', c_Rate, 'LineWidth', 2.0);
ylabel('Firing Rate', 'FontSize', 12, 'Color', c_Rate);
ylim([0 max(r)*1.2]); xlim([0 1.0]);
set(gca, 'YColor', c_Rate, 'FontSize', 12, 'Box', 'off', 'XTickLabel', [], 'LineWidth', 1.5);

% --- Panel B: Mean Membrane Potential ---
subplot(3,1,2);
plot(t, v, 'Color', c_Volt, 'LineWidth', 1.5);
ylabel('Membrane potential', 'FontSize', 12);
ylim([min(v)-1 max(v)+1]); xlim([0 1.0]);
set(gca, 'FontSize', 12, 'Box', 'off', 'XTickLabel', [], 'LineWidth', 1.5, 'YColor', c_Ax);

% --- Panel C: Combined Synaptic Dynamics (Dual Axis) ---
subplot(3,1,3);

% 1. Effective Synaptic Gating (Background Shade)
% In this model, synaptic strength is proportional to u(t)*x(t)
eff_weight = u .* x;
yyaxis left;
hold on;
area(t, eff_weight, 'FaceColor', c_Memory, 'EdgeColor', 'none', 'FaceAlpha', 0.15);

% 2. Left Axis: Resources (x) - Blue
plot(t, x, 'Color', c_Depress, 'LineWidth', 2.0);
ylabel('Resources (q)', 'FontSize', 12, 'Color', c_Depress);
ylim([0 1.1]); 
set(gca, 'YColor', c_Depress, 'FontSize', 12, 'LineWidth', 1.5);

% 3. Right Axis: Utilization (u) - Red
yyaxis right;
plot(t, u, 'Color', c_Facil, 'LineWidth', 2.0);
ylabel('Utilization (u)', 'FontSize', 12, 'Color', c_Facil);
ylim([0 1.0]); 
set(gca, 'YColor', c_Facil, 'FontSize', 12, 'LineWidth', 1.5);

xlabel('Time (s)', 'FontSize', 12);
xlim([0 1.0]);
grid off;

%% 4. Exact Neural Mass ODE Function
function dydt = taher_mass_ode(t, y, P)
    % Unpack variables
    r = y(1); % Firing rate
    v = y(2); % Mean membrane potential
    x = y(3); % Depression variable
    u = y(4); % Facilitation variable
    
    % External Stimulus
    I_S = 0;
    if (t >= P.t1_start && t <= P.t1_end) || (t >= P.t2_start && t <= P.t2_end)
        I_S = P.I_stim_amp;
    end
    
    % --- Exact Neural Mass Equations (Eq 14 in Methods / Eq 1 & 2 in Results) ---
    
    % 1. Firing Rate Dynamics (Eq 14a / 1a)
    % tau_m * r_dot = Delta/(pi*tau_m) + 2*r*v
    drdt = (P.Delta / (pi * P.tau_m) + 2 * r * v) / P.tau_m;
    
    % 2. Mean Membrane Potential Dynamics (Eq 14b / 1b)
    % tau_m * v_dot = v^2 + H + I_B + I_S - (pi*tau_m*r)^2 + tau_m * J_eff * r
    % Synaptic coupling J_eff = J * u * x
    synaptic_input = P.J * u * x * r; 
    dvdt = (v^2 + P.H + P.I_B + I_S - (pi * P.tau_m * r)^2 + P.tau_m * synaptic_input) / P.tau_m;
    
    % 3. Depression Dynamics (Eq 14c / 2a)
    % x_dot = (1-x)/tau_d - u * x * r
    dxdt = (1 - x) / P.tau_d - u * x * r;
    
    % 4. Facilitation Dynamics (Eq 14d / 2b)
    % u_dot = (U0 - u)/tau_f + U0 * (1 - u) * r
    dudt = (P.U0 - u) / P.tau_f + P.U0 * (1 - u) * r;
    
    dydt = [drdt; dvdt; dxdt; dudt];
end