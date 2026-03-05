% Wilson (2016) BCM Model - Weight Change vs Calcium
% Sources: Wilson et al. (2016), Shouval et al. (2002)

clear; clc; close all;

% --- 1. Visual Style ---
c_LTP   = [0.2 0.6 0.8];   % Teal
c_LTD   = [0.8 0.3 0.3];   % Brick Red
c_Basal = [0.5 0.5 0.5];   % Grey
c_Net   = [0.1 0.5 0.3];   % Deep Emerald
c_Ax    = [0.15 0.15 0.15];% Soft Black

% --- 2. Parameters (Wilson et al. 2016) ---
% Thresholds: "LTD (0.15-0.5), LTP (>0.5)"
theta_d = 0.15e-6;  % Depression Threshold (0.15 uM)
theta_p = 0.5e-6;   % Potentiation Threshold (0.50 uM)
slope   = 0.05e-6;  % Sigmoid transition width

% --- 3. Functions ---
% Helper: Sigmoid
sigmoid = @(x, th, k) 1 ./ (1 + exp(-(x - th)/k));

% Omega (Target Weight): 
% 0.5 (Low) -> Drops to 0 (Medium/LTD) -> Rises to 1 (High/LTP)
omega_f = @(c) 0.5 - 0.5*sigmoid(c, theta_d, slope) + 1.0*sigmoid(c, theta_p, slope);

% Eta (Learning Rate): 
% "Low at low levels... high at moderate to high" 
% Modeled as turning on at the first threshold (theta_d)
eta_f   = @(c) 10 * sigmoid(c, theta_d, slope);

% --- 4. Calculate Curves ---
Ca_range = linspace(0, 1e-6, 1000); 
Omega_vals = omega_f(Ca_range);
Eta_vals   = eta_f(Ca_range);

% Drift Calculation (dW/dt)
% dW/dt = Eta * (Omega - W_current)
% W_current = 0.5 (Assuming synapse is at basal strength)
W_current = 0.5; 
Drift = Eta_vals .* (Omega_vals - W_current); 

% --- 5. Plotting ---
figure('Color','w', 'Position', [100 100 600 700]);

% === Subplot 1: Control Functions ===
subplot(2,1,1); hold on;

% Patches
x_LTD = [theta_d theta_p theta_p theta_d] * 1e6;
y_LTD = [0 0 1.3 1.3];
patch(x_LTD, y_LTD, c_LTD, 'FaceAlpha', 0.1, 'EdgeColor', 'none');

x_LTP = [theta_p 1.0 1.0 theta_p] * 1e6;
y_LTP = [0 0 1.3 1.3];
patch(x_LTP, y_LTP, c_LTP, 'FaceAlpha', 0.1, 'EdgeColor', 'none');

% Plot Omega (Left Axis)
yyaxis left
plot(Ca_range*1e6, Omega_vals, '-', 'Color', c_Ax, 'LineWidth', 2.5);
ylabel('\Omega (Target Weight)', 'Color', c_Ax, 'FontSize', 12);
set(gca, 'YColor', c_Ax, 'YTick', [0 0.5 1]);
ylim([0 1.3]);

% Plot Eta (Right Axis)
yyaxis right
plot(Ca_range*1e6, Eta_vals, '--', 'Color', c_Basal, 'LineWidth', 2);
ylabel('\eta (Learning Rate)', 'Color', c_Basal, 'FontSize', 12);
set(gca, 'YColor', c_Basal);
ylim([0 13]);

xlim([0 1]);
xlabel('Intracellular Calcium (\muM)', 'Color', c_Ax, 'FontSize', 12);
set(gca, 'Box', 'off', 'LineWidth', 1.2, 'XColor', c_Ax);

text(0.32, 3.15, 'LTD Zone', 'Color', c_LTD, 'FontWeight', 'bold', 'FontSize', 11, 'HorizontalAlignment', 'center');
text(0.75, 3.15, 'LTP Zone', 'Color', c_LTP, 'FontWeight', 'bold', 'FontSize', 11, 'HorizontalAlignment', 'center');

% === Subplot 2: Plasticity vs Calcium ===
subplot(2,1,2); hold on;

% Zero Line
yline(0, '-', 'Color', c_Ax, 'LineWidth', 1);

% Fill Areas
area(Ca_range(Drift<0)*1e6, Drift(Drift<0), 'FaceColor', c_LTD, 'FaceAlpha', 0.3, 'EdgeColor', 'none');
area(Ca_range(Drift>0)*1e6, Drift(Drift>0), 'FaceColor', c_LTP, 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Plot Drift Curve
plot(Ca_range*1e6, Drift, 'Color', c_Net, 'LineWidth', 3);

ylabel('Weight Change (ds/dt)', 'Color', c_Net, 'FontSize', 12);
xlabel('Intracellular Calcium (\muM)', 'Color', c_Ax, 'FontSize', 12);
xlim([0 1]);
ylim([min(Drift)*1.2, max(Drift)*1.2]);
set(gca, 'Box', 'off', 'LineWidth', 1.2, 'XColor', c_Ax, 'YColor', c_Net);

text(0.32, min(Drift)/2, 'Depression (-)', 'Color', c_LTD, 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(0.75, max(Drift)/2, 'Potentiation (+)', 'Color', c_LTP, 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');