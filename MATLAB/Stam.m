% Stam et al. (2010) - Adaptive Rewiring
clear; clc; close all;

% --- Visual Style ---
c_GDP    = [0.2 0.6 0.8];   % Teal (Distance Rule)
c_SDP    = [0.8 0.3 0.3];   % Red (Sync Rule)
c_Ax     = [0.15 0.15 0.15];% Dark Grey (Axes/Text)
colormap_style = flipud(gray); 

% --- Simulation Parameters ---
N = 40;             
T_sim = 600;        
dt = 0.05;

% 1. Initial Topology (Random)
W = rand(N) > 0.80; 
W = double(W - diag(diag(W))); 
W = (W + W')/2; 
W(W>0) = 0.5;
W_start = W; 

% 2. Dynamics Setup (Kuramoto)
theta = 2*pi*rand(N, 1);     
omega = 1.0 + 0.1*randn(N,1);
coupling_k = 2.0;            

% 3. Plasticity Parameters
alpha_SDP = 0.02;   
threshold_SDP = 0.4;
Target_Deg = 4;     

% --- Evolution Loop ---
for t = 1:T_sim
    % Fast Dynamics
    phase_diff = theta' - theta; 
    interaction = sum(W .* sin(phase_diff), 2);
    theta = theta + (omega + coupling_k/N * interaction) * dt;
    
    % Slow Dynamics
    if mod(t, 5) == 0 
        Sync = cos(phase_diff); 
        
        % A. SDP (Hebbian)
        dW = alpha_SDP * (Sync - threshold_SDP);
        mask = W > 0;
        W(mask) = W(mask) + dW(mask);
        W(W < 0.05) = 0; % Pruning
        W = (W + W') / 2; 
        
        % B. GDP (Homeostatic)
        degrees = sum(W > 0, 2);
        for i = 1:N
            if degrees(i) < Target_Deg
                dist_vec = abs((1:N) - i);
                dist_vec = min(dist_vec, N - dist_vec); 
                prob = exp(-dist_vec / 4); 
                prob(i) = 0; prob(W(i,:) > 0) = 0; 
                if sum(prob) > 0
                    target = randsample(N, 1, true, prob);
                    W(i, target) = 0.5; W(target, i) = 0.5; 
                end
            end
        end
    end
end

% --- PLOTTING ---
figure('Color','w', 'Position', [100 100 600 600]); 

% Panel 1: GDP Rule (Distance)
subplot(2,2,1); hold on;
d = 0:15; 
target_w = exp(-0.2 * d); % Stam Eq. 3
plot(d, target_w, 'Color', c_GDP, 'LineWidth', 2.5);
title('GDP Rule', 'FontWeight', 'bold', 'Color', c_Ax);
xlabel('Node Distance', 'Color', c_Ax); 
ylabel('Target Strength', 'Color', c_Ax);
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XColor', c_Ax, 'YColor', c_Ax);
xlim([0 15]); ylim([0 1]);

% Panel 2: SDP Rule (Synchronization)
subplot(2,2,2); hold on;
r = linspace(0, 2, 100);
Hill = (r.^2)./(r.^2 + 1) - 0.5; % Stam Eq. 4
plot(r, Hill, 'Color', c_SDP, 'LineWidth', 2.5);
yline(0, '-', 'Color', [0.7 0.7 0.7]);
xline(1, '--', 'Color', [0.7 0.7 0.7]);
title('SDP Rule', 'FontWeight', 'bold', 'Color', c_Ax);
xlabel('Synchronization (r)', 'Color', c_Ax); 
ylabel('\Delta Weight', 'Color', c_Ax);
set(gca, 'LineWidth', 1.2, 'Box', 'off', 'XColor', c_Ax, 'YColor', c_Ax);
xlim([0 2]); ylim([-0.6 0.6]);

% Panel 3: Start Matrix
subplot(2,2,3);
imagesc(W_start); colormap(gca, colormap_style);
axis square; axis off;
title('Network Weights (Start)', 'FontWeight', 'bold', 'Color', c_Ax);

% Panel 4: End Matrix
subplot(2,2,4);
[~, idx] = sort(theta); 
imagesc(W(idx, idx)); colormap(gca, colormap_style);
axis square; axis off;
title('Network Weights (End)', 'FontWeight', 'bold', 'Color', c_Ax);