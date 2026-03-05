% MATLAB Script: Kilpatrick-Bressloff (2010) - Stationary Bump with Depression
% Sources: Kilpatrick & Bressloff (2010)

clear; clc; close all;

% --- 1. Visual Style ---
c_U     = [0.2 0.6 0.8];   % Teal (Used for Activity U)
c_Q     = [0.8 0.3 0.3];   % Brick Red (Used for Resources Q)
c_Thresh= [0.5 0.5 0.5];   % Grey (Threshold theta)
c_Ax    = [0.15 0.15 0.15];% Soft Black
c_Fill  = [0.2 0.6 0.8];   % Teal fill for activity

% --- 2. Parameters (Kilpatrick & Bressloff 2010) ---
% Parameters from Fig 1 caption
alpha = 20;       % Recovery time constant
beta  = 0.01;     % Depression strength
theta = 0.2;      % Firing threshold

% --- 3. Functions ---
% Weight integral function W(x) for Mexican Hat
% w(x) = (1-|x|)exp(-|x|) -> W(x) = x*exp(-|x|)
W_func = @(x) x .* exp(-abs(x));

% Synaptic Drive U(x)
% U(x) = 1/(1+alpha*beta) * [W(x+a) - W(x-a)]
U_func = @(x, a, alp, bet) (1 / (1 + alp * bet)) .* (W_func(x + a) - W_func(x - a));

% Depression Variable Q(x)
% Q(x) = 1 - (alpha*beta)/(1+alpha*beta) * Heaviside(U(x) - theta)
Q_func = @(u_val, alp, bet, th) 1 - ((alp * bet) / (1 + alp * bet)) .* (u_val > th);

% --- 4. Solve and Calculate ---
% Solve for bump half-width 'a' using threshold condition U(a) = theta
% Implicit equation: 2*a*exp(-2*a)/(1+alpha*beta) - theta = 0
target_eq = @(a) (2 * a .* exp(-2 * a)) ./ (1 + alpha * beta) - theta;
options = optimset('Display','off');
a_sol = fzero(target_eq, 1.0, options); % Initial guess ~1.0

% Generate spatial domain
x = linspace(-6, 6, 1200); % Slightly wider X range for context

% Compute Profiles
U_vals = U_func(x, a_sol, alpha, beta);
Q_vals = Q_func(U_vals, alpha, beta, theta);

% --- 5. Plotting ---
figure('Color','w', 'Position', [100 100 700 550]);
hold on;

% === Plot Elements ===

% 1. Active Region Shading (Under U(x))
% Identify region where U > theta
active_region = U_vals > theta;

% 2. Threshold Line
yline(theta, '--', 'Color', c_Thresh, 'LineWidth', 1.5);
text(-5.5, theta + 0.06, ['\theta'], ...
    'Color', c_Thresh, 'FontSize', 16, 'FontWeight', 'bold');

% 3. Synaptic Drive U(x) (Teal)
p1 = plot(x, U_vals, '-', 'Color', c_U, 'LineWidth', 3);

% 4. Depression Q(x) (Red, Dashed)
p2 = plot(x, Q_vals, '--', 'Color', c_Q, 'LineWidth', 2.5);

% === Annotations and Styling ===

% Axis Limits - EXPANDED Y-AXIS
xlim([-6 6]);
ylim([-0.4 1.3]); % Expanded from [-0.2 1.1] to give more vertical space

% Labels
xlabel('x', 'Color', c_Ax, 'FontSize', 16);

% Custom Text Labels (Repositioned for new limits)
text(0, 0.7, 'Synaptic Drive U(x)', 'Color', c_U, ...
    'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'BackgroundColor', 'w', 'Margin', 1);

text(0.5, 1.075, 'Available Resources Q(x)', 'Color', c_Q, ...
    'FontSize', 16, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

% Axis Properties
set(gca, 'Box', 'off', 'LineWidth', 1.2, 'XColor', c_Ax, 'YColor', c_Ax);
set(gca, 'YTick', [-0.2 0 theta 0.5 1],'FontSize', 16);

hold off;