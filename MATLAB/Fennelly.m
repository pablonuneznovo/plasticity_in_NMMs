% Fennelly et al. (2025) - Synchrony-Driven Adaptive Coupling
clear; clc; close all;

% --- Visual Style ---
c_Target = [0.5 0.3 0.7];   % Purple (Target Function)
c_K      = [0.2 0.6 0.8];   % Teal (Coupling k)
c_Sync   = [0.8 0.4 0.2];   % Orange (Synchrony |z|)
c_Ax     = [0.15 0.15 0.15];% Soft Black

% --- Parameters ---
alpha = 2.0;    
epsilon = 0.1;  

% --- Simulation ---
T_sim = 200; dt = 0.1; time = 0:dt:T_sim;

% 1. Input: Synchrony |z|
z_mag = zeros(size(time));
z_mag(time < 50) = 0.1;             
z_mag(time >= 50 & time < 120) = 0.9; 
z_mag(time >= 120) = 0.4;           

% 2. Dynamics: dk/dt = epsilon * (-k + alpha * |z|^2)
k = zeros(size(time));
k(1) = 0.1; 

for i = 2:length(time)
    dk = epsilon * (-k(i-1) + alpha * z_mag(i-1)^2);
    k(i) = k(i-1) + dk * dt;
end

% --- PLOTTING ---
figure('Color','w', 'Position', [100 100 400 600]); 

% Panel 1: The Rule
subplot(2,1,1); hold on;
r_range = linspace(0, 1, 100);
Target_Curve = alpha * r_range.^2; 

plot(r_range, Target_Curve, 'Color', c_Target, 'LineWidth', 3);
area(r_range, Target_Curve, 'FaceColor', c_Target, 'FaceAlpha', 0.1, 'EdgeColor', 'none');

title('Macroscopic Rule', 'FontSize', 12, 'FontWeight', 'bold', 'Color', c_Ax);
ylabel('Target Coupling \alpha |z|^2', 'FontSize', 11, 'Color', c_Ax);
xlabel('Synchrony |z|', 'FontSize', 11, 'Color', c_Ax); 
xlim([0 1]); ylim([0 alpha*1.1]);
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XColor', c_Ax, 'YColor', c_Ax);

% Panel 2: Dynamics
subplot(2,1,2); hold on;

% Synchrony (Left Axis)
yyaxis left
plot(time, z_mag, ':', 'Color', c_Sync, 'LineWidth', 2);
ylabel('Synchrony |z|(t)', 'FontSize', 11, 'Color', c_Sync, 'FontWeight', 'bold');
set(gca, 'YColor', c_Sync);
ylim([0 1.1]);

% Coupling k (Right Axis)
yyaxis right
plot(time, k, 'Color', c_K, 'LineWidth', 2.5);
ylabel('Mean Coupling k(t)', 'FontSize', 12, 'Color', c_K, 'FontWeight', 'bold');
set(gca, 'YColor', c_K);
ylim([0 alpha]);

title('Adaptation Dynamics', 'FontSize', 12, 'FontWeight', 'bold', 'Color', c_Ax);
xlabel('Simulation Time', 'FontSize', 11, 'Color', c_Ax);
xlim([0 T_sim]);
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XColor', c_Ax);