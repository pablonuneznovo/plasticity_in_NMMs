% Gallinaro et al. (2022) structural-plasticity mean-field figure.
% Translates the released Mathematica notebook time-series-fig4.nb from:
% https://github.com/juliavg/engram_structural_plasticity
%
% The script solves the deterministic mean-field ODE system used for
% Gallinaro et al., Fig. 4: population rates, calcium traces, and the 2-by-2
% excitatory connectivity matrix. Stochastic NEST connectivity samples
% distributed with the authors' public code are loaded from
% gallinaro_nest_connectivity_data.mat and overlaid as dots in panel C.

clear; clc; close all;

% --- Colour Palette ---
c_Rate    = [0.2 0.2 0.2];
c_Input   = [0.6 0.6 0.6];
c_Trace   = [0.3 0.5 0.3];
c_Facil   = [0.8 0.3 0.3];
c_Depress = [0.2 0.4 0.8];
c_Memory  = [0.6 0.4 0.8];
c_Ax      = [0.15 0.15 0.15];

%% 1. Parameters from Gallinaro et al. Fig. 4 notebook
P.J           = 0.1;
P.tau_m       = 0.02;
P.tau_r       = 0.002;
P.theta       = 20.0;
P.v_reset     = 10.0;
P.g           = 8.0;
P.epsilon     = 0.1;
P.n_e1        = 1000.0;
P.n_e2        = 9000.0;
P.n_x         = 10000.0;
P.nu_ext      = 15.0;
P.n_i         = 2500.0;
P.tau_rate    = 0.0002;
P.tau_ca      = 10.0;
P.beta_a      = 2.0;
P.beta_d      = 2.0;
P.k_a         = 1.0;
P.k_d         = 1.0;
P.t0          = 0.0;
P.tf          = 6150.0;
P.tf_plot     = 1150.0;
P.t_on        = 500.0;
P.t_off       = 650.0;
P.target_rate = 8.0;
P.n_pop       = [P.n_e1; P.n_e2];

% Numerical primitive of exp(u^2)*erfc(-u), the Brunel/Siegert integral
% in Gallinaro Eq. 13. This accelerates the same transfer function.
P.u_grid = linspace(-8.0, 8.0, 20001)';
integrand = exp(min(P.u_grid.^2, 700.0)) .* erfc(-P.u_grid);
P.primitive = cumtrapz(P.u_grid, integrand);

%% 2. Solve mean-field ODE
% State vector:
% [phi_E1; phi_E2; W_E1E1; W_E1E2; W_E2E1; W_E2E2; r_E1; r_E2; r_I]
initial = [ones(2,1) * P.target_rate; ones(4,1) * 0.1; ones(3,1) * P.target_rate];
y0 = 0.001 * initial;

options = odeset('RelTol', 1e-5, 'AbsTol', 1e-7, 'MaxStep', 2.0);
[t, y] = ode15s(@(t,y) gallinaro_mean_field_ode(t, y, P), [P.t0 P.tf], y0, options);

phi = y(:, 1:2);
w11 = y(:, 3);
w12 = y(:, 4);
w21 = y(:, 5);
w22 = y(:, 6);
rates = y(:, 7:9);
input_e1 = zeros(size(t));
for ii = 1:numel(t)
    stim_vec = stimulus(t(ii), P);
    input_e1(ii) = stim_vec(1);
end

%% 3. NEST connectivity samples from the authors' public code repository
this_file = mfilename('fullpath');
out_dir = fileparts(this_file);
data_file = fullfile(out_dir, 'gallinaro_nest_connectivity_data.mat');
if ~exist(data_file, 'file')
    error('Could not find gallinaro_nest_connectivity_data.mat. Keep it in the same folder as this script.');
end
nest_data = load(data_file);
nest_t = nest_data.sim_steps(:) / 1000.0;
nest_c11 = nest_data.c11(:);
nest_c12 = nest_data.c12(:);
nest_c21 = nest_data.c21(:);
nest_c22 = nest_data.c22(:);

%% 4. Visualization
fig = figure('Color', 'w', 'Position', [100, 100, 700, 900]);

% --- Panel A: firing-rate dynamics and stimulation ---
subplot(3,1,1);
yyaxis right;
area(t, input_e1, 'FaceColor', c_Input, 'EdgeColor', 'none', 'FaceAlpha', 0.2);
ylabel('$s_{E_1}(t)$', 'Interpreter', 'latex', 'FontSize', 12, 'Color', c_Input);
ylim([0 0.15]);
set(gca, 'YColor', c_Input, 'FontSize', 12);

yyaxis left;
plot(t, rates(:,1), 'Color', c_Rate, 'LineWidth', 2.0); hold on;
plot(t, rates(:,2), 'Color', c_Input, 'LineWidth', 1.6);
plot([P.t0 P.tf], [P.target_rate P.target_rate], '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 1.1);
ylabel('$r_Y(t)$ (Hz)', 'Interpreter', 'latex', 'FontSize', 12, 'Color', c_Rate);
ylim([0 max(max(rates(:,1:2))) * 1.18]);
xlim([P.t0 P.tf_plot]);
xticks([0 50 200 350 500 650 800 950 1100]);
yl = ylim;
plot([P.t_on P.t_on], yl, 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
plot([P.t_off P.t_off], yl, 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
ylim(yl);
legend({'$r_{E_1}(t)$','$r_{E_2}(t)$'}, ...
    'Interpreter', 'latex', 'Location', 'northeast', 'Box', 'off');
set(gca, 'YColor', c_Rate, 'FontSize', 12, 'Box', 'off', 'XTickLabel', [], 'LineWidth', 1.5);

% --- Panel B: calcium trace Eq. 14 ---
subplot(3,1,2);
plot(t, phi(:,1), 'Color', c_Trace, 'LineWidth', 2.0); hold on;
plot(t, phi(:,2), 'Color', c_Input, 'LineWidth', 1.6);
plot([P.t0 P.tf], [P.target_rate P.target_rate], '--', 'Color', [0.55 0.55 0.55], 'LineWidth', 1.1);
ylabel('$\phi_Y(t)$', 'Interpreter', 'latex', 'FontSize', 12);
ylim([0 max(max(phi)) * 1.14]);
xlim([P.t0 P.tf_plot]);
xticks([0 50 200 350 500 650 800 950 1100]);
yl = ylim;
plot([P.t_on P.t_on], yl, 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
plot([P.t_off P.t_off], yl, 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
ylim(yl);
legend({'$\phi_{E_1}(t)$','$\phi_{E_2}(t)$'}, ...
    'Interpreter', 'latex', 'Location', 'northeast', 'Box', 'off');
set(gca, 'FontSize', 12, 'Box', 'off', 'XTickLabel', [], 'LineWidth', 1.5, 'YColor', c_Ax);

% --- Panel C: connectivity matrix Eq. 15, with released NEST samples ---
subplot(3,1,3);
plot(t, w11, 'Color', c_Facil, 'LineWidth', 2.0); hold on;
plot(t, w12, 'Color', c_Depress, 'LineWidth', 2.0);
plot(t, w21, 'Color', c_Memory, 'LineWidth', 2.0);
plot(t, w22, 'Color', c_Rate, 'LineWidth', 2.0);
plot(nest_t, nest_c11, 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', c_Facil, 'LineWidth', 0.9);
plot(nest_t, nest_c12, 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', c_Depress, 'LineWidth', 0.9);
plot(nest_t, nest_c21, 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', c_Memory, 'LineWidth', 0.9);
plot(nest_t, nest_c22, 'o', 'MarkerSize', 3.5, 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', c_Rate, 'LineWidth', 0.9);
ylabel('$\bar{C}_{Y,Z}(t)$', 'Interpreter', 'latex', 'FontSize', 12);
xlabel('Time (s)', 'FontSize', 12);
ylim([0 max([max(w11), max(nest_c11)]) * 1.08]);
xlim([P.t0 P.tf_plot]);
xticks([0 50 200 350 500 650 800 950 1100]);
yl = ylim;
plot([P.t_on P.t_on], yl, 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
plot([P.t_off P.t_off], yl, 'Color', [0.45 0.45 0.45], 'LineWidth', 1.0);
ylim(yl);
legend({'$\bar{C}_{E_1,E_1}(t)$','$\bar{C}_{E_1,E_2}(t)$', ...
    '$\bar{C}_{E_2,E_1}(t)$','$\bar{C}_{E_2,E_2}(t)$'}, ...
    'Interpreter', 'latex', 'Location', 'northoutside', ...
    'Orientation', 'horizontal', 'Box', 'off');
set(gca, 'FontSize', 12, 'Box', 'off', 'LineWidth', 1.5, 'YColor', c_Ax);

%% Local functions
function dydt = gallinaro_mean_field_ode(t, y, P)
    phi = max(y(1:2), 0.0);
    W = max([y(3) y(4); y(5) y(6)], 1e-12);
    rates = max(y(7:9), 0.0);
    r_e = rates(1:2);
    r_i = rates(3);

    k_in = max(W * P.n_pop, 1e-12);
    k_out = max(W' * P.n_pop, 1e-12);

    dend_create = max((P.target_rate - phi) * P.k_d / P.beta_d, 0.0);
    axon_create = max((P.target_rate - phi) * P.k_a / P.beta_a, 0.0);
    dend_delete = max((phi - P.target_rate) * P.k_d / P.beta_d, 0.0);
    axon_delete = max((phi - P.target_rate) * P.k_a / P.beta_a, 0.0);

    dendritic_loss = W .* ((dend_delete ./ k_in) * ones(1,2));
    axonal_loss = W .* (ones(2,1) * (axon_delete ./ k_out)');

    free_dendrites = dend_create + sum(axonal_loss .* (ones(2,1) * P.n_pop'), 2);
    free_axons = axon_create + sum(dendritic_loss .* (P.n_pop * ones(1,2)), 1)';

    total_free_dendrites = sum(P.n_pop .* free_dendrites);
    total_free_axons = sum(P.n_pop .* free_axons);
    if total_free_dendrites > 0.0 && total_free_axons > 0.0
        pairing_rate = min(total_free_dendrites, total_free_axons);
        lambda_pair = pairing_rate / (total_free_dendrites * total_free_axons);
    else
        lambda_pair = 0.0;
    end

    dW = lambda_pair * (free_dendrites * free_axons') - dendritic_loss - axonal_loss;
    dphi = (r_e - phi) / P.tau_ca;

    stim_vec = stimulus(t, P);
    e_input = (W * (P.n_pop .* r_e)) + (1.0 + stim_vec) * P.epsilon * P.n_x * P.nu_ext;
    mu_e = P.tau_m * P.J * (e_input - P.g * P.epsilon * P.n_i * r_i);
    sigma_e = P.J * sqrt(max(P.tau_m * (e_input + P.g^2 * P.epsilon * P.n_i * r_i), 1e-12));
    target_e = brunel_rate(mu_e, sigma_e, P);

    i_input = P.epsilon * sum(P.n_pop .* r_e);
    mu_i = P.tau_m * P.J * (i_input - P.g * P.epsilon * P.n_i * r_i + P.epsilon * P.n_x * P.nu_ext);
    sigma_i = P.J * sqrt(max(P.tau_m * (i_input + P.g^2 * P.epsilon * P.n_i * r_i + P.epsilon * P.n_x * P.nu_ext), 1e-12));
    target_i = brunel_rate(mu_i, sigma_i, P);

    dr = (-rates + [target_e; target_i]) / P.tau_rate;
    dydt = [dphi; dW(1,1); dW(1,2); dW(2,1); dW(2,2); dr];
end

function rate = brunel_rate(mu, sigma, P)
    sigma = max(sigma, 1e-9);
    a = (P.v_reset - mu) ./ sigma;
    b = (P.theta - mu) ./ sigma;
    integral = primitive_value(b, P) - primitive_value(a, P);
    rate = 1.0 ./ (P.tau_r + P.tau_m * sqrt(pi) .* max(integral, 0.0));
    rate(a >= P.u_grid(end)) = 0.0;
    rate(b <= P.u_grid(1)) = 1.0 / P.tau_r;
    rate(b >= P.u_grid(end)) = 0.0;
end

function values = primitive_value(x, P)
    x_clipped = min(max(x, P.u_grid(1)), P.u_grid(end));
    values = interp1(P.u_grid, P.primitive, x_clipped, 'linear');
    mask = x < P.u_grid(1);
    if any(mask)
        values(mask) = -log((-x(mask)) / (-P.u_grid(1))) / sqrt(pi);
    end
end

function stim_vec = stimulus(t, P)
    z_on = min(max(1000.0 * (t - P.t_on), -60.0), 60.0);
    z_off = min(max(1000.0 * (P.t_off - t), -60.0), 60.0);
    gate = (1.0 / (1.0 + exp(-z_on))) * (1.0 / (1.0 + exp(-z_off)));
    stim_vec = [0.10 * gate; 0.0];
end
