% Panel D1: Structural Plasticity (Diaz-Pier et al., 2016)
% Clean version: Exact Gaussian Growth Rule (Eqs 2-3 in paper)
clear; clc; close all;

% --- Color Palette ---
c_Ca      = [0.2 0.6 0.8];   % Teal (Calcium)
c_Struct  = [0.5 0.3 0.7];   % Purple (Synaptic Elements z)
c_Zone    = [0.2 0.8 0.4];   % Green (Growth Window)
c_Ax      = [0.15 0.15 0.15];% Soft Black

% --- Parameters (Diaz-Pier et al., 2016) ---
% nu: Growth rate. Set to 4.0 x 10^-4 elements/ms
nu = 0.0004;      
% eta: Minimum activity threshold. Set to 0.0
eta = 0.0;        
% epsilon: Target calcium concentration. Set to 0.2 (Inhibitory target)
epsilon = 0.2;    

% Derived Gaussian Parameters
xi = (eta + epsilon) / 2;             % Center of window
zeta = (epsilon - eta) / (2 * sqrt(log(2))); % Width parameter

% --- Simulation Setup ---
T = 25000; 
time = 1:T;

% 1. Calcium Sweep (0 to 0.6)
Ca = linspace(0, 0.6, T); 

% 2. Integration of Structural Elements (z)
z = zeros(size(time));
z(1) = 0.8; 

for t = 2:T
    Current_Ca = Ca(t);
    
    % Exact Gaussian Equation
    growth_speed = nu * ( 2 * exp( -((Current_Ca - xi)/zeta)^2 ) - 1 );
    
    % Update z (Forward Euler)
    z(t) = z(t-1) + growth_speed;
    z(t) = max(0.0, z(t)); 
end

% --- PLOTTING ---
figure('Color','w', 'Position', [100 100 400 500]); 

% Subplot 1: The Gaussian Growth Rule (Function of Calcium)
subplot(2,1,1); hold on;

% Theoretical Curve for visualization
Ca_range = linspace(0, 0.6, 200);
Growth_Curve = nu * ( 2 * exp( -((Ca_range - xi)/zeta).^2 ) - 1 );

% Zero Line
yline(0, '-', 'Color', c_Ax, 'LineWidth', 1);

% Fill Growth Zone (Positive Growth)
area(Ca_range(Growth_Curve>0), Growth_Curve(Growth_Curve>0), ...
    'FaceColor', c_Zone, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot Curve
plot(Ca_range, Growth_Curve, 'k', 'LineWidth', 2);

% Annotations
xline(eta, '--', 'Color', c_Zone, 'LineWidth', 1.5);
xline(epsilon, '--', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5);

% --- FIX: Text Offset ---
% Added +0.02 to x position and changed alignment to 'left'
text(epsilon + 0.02, nu*1.1, '\epsilon (Target)', 'Color', [0.8 0.2 0.2], ...
     'FontSize', 9, 'HorizontalAlignment', 'left');

ylabel('Growth Rate dz/dt', 'FontSize', 10, 'FontWeight', 'bold', 'Color', c_Ax);
xlabel('Calcium concentration', 'FontSize', 10, 'FontWeight', 'bold', 'Color', c_Ax);
title('Gaussian Growth Rule', 'FontSize', 12, 'FontWeight', 'bold', 'Color', c_Ax);
xlim([0 0.6]); 
ylim([-nu nu*1.3]); 
set(gca, 'Box', 'off', 'XColor', c_Ax, 'YColor', c_Ax, 'LineWidth', 1.2, 'YTick', []);

% Subplot 2: Temporal Evolution (Result)
subplot(2,1,2); hold on;

% Visualize the Calcium Input (Background)
yyaxis left
plot(time, Ca, 'Color', c_Ca, 'LineWidth', 2, 'LineStyle', ':');
ylabel('Intracellular Ca^{2+}', 'FontSize', 10, 'Color', c_Ca);
set(gca, 'YColor', c_Ca);
ylim([0 0.6]);

% Visualize the Synaptic Elements (Foreground)
yyaxis right
plot(time, z, 'Color', c_Struct, 'LineWidth', 2.5);
ylabel('Synaptic Elements z(t)', 'FontSize', 10, 'FontWeight', 'bold', 'Color', c_Struct);
set(gca, 'YColor', c_Struct);

% Auto-scale Y-axis
ymax = max(z);
if ymax == 0, ymax = 1; end
ylim([0 ymax * 1.1]); 

xlabel('Simulation Time', 'FontSize', 10, 'FontWeight', 'bold', 'Color', c_Ax);
xlim([0 T]);
set(gca, 'Box', 'off', 'XColor', c_Ax, 'LineWidth', 1.2);


grid on;
