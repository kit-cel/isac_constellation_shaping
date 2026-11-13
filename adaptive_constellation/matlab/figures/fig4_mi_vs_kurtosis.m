% =========================================================================
% Fig. 4: Communication performance of the adaptive constellation
%
% Compares the mutual information (MI) over an AWGN channel of the
% pretrained adaptive constellation family with the upper bound on the MI
% under a kurtosis constraint, Eq. (11) in [1], for several SNR values.
% The MI is plotted over the achieved kurtosis of the constellations.
%
% Reference:
% [1] B. Geiger, L. Hampel, and L. Schmalen, "Joint power and
%     adaptive-constellation optimization for OFDM-ISAC systems," in Proc.
%     IEEE Conf. Integr. Sens. Commun. (ISAC), Lisbon, Portugal, Nov. 2026.
%     arXiv:2611.12166
%
% License: MIT (see LICENSE in repository root)
% =========================================================================

clear; close all; clc;

snr_dB = [10, 6, 2];                        % communication SNR 1/sigma_c^2
kappa_target = 1:0.01:2;                    % target kurtosis values
matlab_dir = fileparts(fileparts(mfilename("fullpath")));   % .../matlab
addpath(matlab_dir, fullfile(matlab_dir, "functions"));
weights = load(fullfile(matlab_dir, "adaptive_constellation_weights.mat"));

% Generate constellations once
M = weights.M;
x = zeros(M, numel(kappa_target));
p = zeros(M, numel(kappa_target));
for k = 1:numel(kappa_target)
    [x(:, k), p(:, k)] = generate_adaptive_constellation(kappa_target(k), weights);
end
kappa_achieved = sum(p .* abs(x).^4, 1);

% MI of the adaptive constellations (Gauss-Hermite quadrature, see MI.m)
% and upper bound
mi_adaptive = zeros(numel(snr_dB), numel(kappa_target));
mi_upper_bound = zeros(numel(snr_dB), numel(kappa_target));
for s = 1:numel(snr_dB)
    for k = 1:numel(kappa_target)
        mi_adaptive(s, k) = MI(x(:, k), p(:, k), snr_dB(s));
    end
    mi_upper_bound(s, :) = mi_upper_bound_kurtosis(kappa_target, snr_dB(s));
end

%% Plot
colors = [0.85 0.15 0.15; 0.20 0.45 0.75; 1.00 0.50 0.05];
figure("Name", "Fig. 4: MI vs. kurtosis"); hold on;
h = gobjects(numel(snr_dB), 1);
for s = 1:numel(snr_dB)
    plot(kappa_target, mi_upper_bound(s, :), "--", "Color", colors(s, :), ...
         "LineWidth", 2.5, "HandleVisibility", "off");
    h(s) = plot(kappa_achieved, mi_adaptive(s, :), "-", "Color", colors(s, :), ...
                "LineWidth", 1.5, "DisplayName", sprintf("%d dB", snr_dB(s)));
end
h_ub = plot(nan, nan, "--", "Color", [0.5 0.5 0.5], "LineWidth", 2.5, ...
            "DisplayName", "Upper bound");
h_ad = plot(nan, nan, "-k", "LineWidth", 1.5, "DisplayName", "Novel adaptive");
grid on; box on; xlim([1 2]);
xlabel("Kurtosis \kappa_n"); ylabel("MI I (bit/symbol)");
legend([h; h_ub; h_ad], "Location", "east");
