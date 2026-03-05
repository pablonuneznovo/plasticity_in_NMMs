% Analytical vs Iterative FIC (Comparison)
clear; clc; close all;

% --- 1. Load Data ---
try
    load('SC_sample.mat', 'C'); 
catch
    error('SC_sample.mat not found. Please upload it.');
end
C = double(C); N = size(C,1);
C(1:N+1:end) = 0; 
if max(C(:))>0; C = C/max(C(:))*0.2; end

% --- 2. Parameters ---
a_E = 310; b_E = 125; d_E = 0.16;
tau_S = 100; gamma = 0.641/1000;
w_plus = 1.4; J_NMDA = 0.15; I_0 = 0.382;
G = 6.0; 

% =========================================================================
% METHOD A: ANALYTICAL SOLUTION 
% =========================================================================
fprintf('--- Method A: Analytical ---\n');

Threshold = b_E / a_E;      
Target_Offset = -0.026;     
I_target = Threshold + Target_Offset; 

num = a_E * I_target - b_E;
Target_Rate = num / (1 - exp(-d_E * num)); 
S_target = (tau_S * gamma * Target_Rate) / (1 + tau_S * gamma * Target_Rate);

Term_Self = w_plus * J_NMDA * S_target;
Term_Net  = G * J_NMDA * (C * (ones(N,1) * S_target));
Term_Bias = I_0;

J_Analytical = (Term_Self + Term_Net + Term_Bias - I_target) / S_target;
fprintf('Analytical Calculation: Done.\n');

% =========================================================================
% METHOD B: ITERATIVE SOLUTION 
% =========================================================================
fprintf('\n--- Method B: Iterative ---\n');

J_Iter = ones(N,1); 
Delta = 0.05;       
Max_Epochs = 25;

% Find hub and leaf nodes
Idx_Hub = find(sum(C,2) == max(sum(C,2)), 1);
Idx_Leaf = find(sum(C,2) == min(sum(C,2)), 1);

Hist_Hub = [];
Hist_Leaf = [];

for epoch = 1:Max_Epochs
    
    % Simulation Loop (Mean Field)
    S = ones(N,1) * S_target; 
    Avg_Input = zeros(N,1);
    
    steps = 1000; dt = 0.5;
    for t = 1:steps
        I_net = G * J_NMDA * (C * S);
        I_E = w_plus * J_NMDA * S - J_Iter .* S + I_net + I_0;
        
        num = a_E * I_E - b_E;
        r = num ./ (1 - exp(-d_E * num));
        r(abs(num)<1e-9) = 1/d_E; r(r<0)=0;
        
        dS = -S/tau_S + (1-S).*gamma.*r;
        S = S + dS * dt;
        
        if t > 200; Avg_Input = Avg_Input + I_E; end
    end
    Avg_Input = Avg_Input / (steps-200);
    
    % Store History
    Hist_Hub(epoch) = J_Iter(Idx_Hub);
    Hist_Leaf(epoch) = J_Iter(Idx_Leaf);
    
    % Update Rule
    diff = Avg_Input - I_target;
    J_Iter(diff > 0.001)  = J_Iter(diff > 0.001)  + Delta;
    J_Iter(diff < -0.001) = J_Iter(diff < -0.001) - Delta;
    
    fprintf('Epoch %d: Hub J = %.3f\n', epoch, J_Iter(Idx_Hub));
end

% =========================================================================
% PLOTTING (2x1 Layout)
% =========================================================================
figure('Color','w', 'Position', [100 50 600 800]); % Taller figure

% Panel A: Convergence of J
subplot(2,1,1); hold on;

% Analytical Benchmarks (Dashed Lines)
yline(J_Analytical(Idx_Hub), '--', 'Color', [0.5 0 0.5], 'LineWidth', 2);
yline(J_Analytical(Idx_Leaf), '--', 'Color', [0.4 0.8 0.4], 'LineWidth', 2);

% Iterative Trajectories
plot(Hist_Hub, '-o', 'Color', [0.5 0 0.5], 'MarkerFaceColor', [0.5 0 0.5], 'LineWidth', 1.5);
plot(Hist_Leaf, '-s', 'Color', [0.4 0.8 0.4], 'MarkerFaceColor', [0.4 0.8 0.4], 'LineWidth', 1.5);

ylabel('Inhibitory Weight J_i', 'FontSize', 11); 
xlabel('Optimization Epoch', 'FontSize', 11);

legend({'Theoretical target (Hub)', 'Theoretical target (Leaf)', 'Hub (Simulation)', 'Leaf (Simulation)'}, ...
    'Location', 'East', 'FontSize', 9, 'Box', 'off');

grid on; axis square; 
set(gca, 'Box', 'off', 'LineWidth', 1.2);

% Panel B: Resulting Rates
subplot(2,1,2); hold on;
bar(r, 'FaceColor', [0.8 0.3 0.3], 'EdgeColor', 'none');
yline(Target_Rate, 'k--', 'LineWidth', 2, 'Label', sprintf('Target %.2f Hz', Target_Rate));

title('Final Firing Rates', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Firing Rate (Hz)', 'FontSize', 11); 
xlabel('Node Index', 'FontSize', 11);
ylim([0 4.5]); xlim([0 N+1]);
set(gca, 'Box', 'off', 'LineWidth', 1.2);
