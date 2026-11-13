function [x, p] = generate_adaptive_constellation(kappa_target, weights)
% GENERATE_ADAPTIVE_CONSTELLATION  Kurtosis-adaptive constellation for OFDM-ISAC.
%
%   [x, p] = generate_adaptive_constellation(kappa_target)
%   [x, p] = generate_adaptive_constellation(kappa_target, weights)
%
%   Generates the geometrically and probabilistically shaped constellation
%   (M = 64 points) of the pretrained kurtosis-adaptive constellation family
%   for the target kurtosis kappa_target in [1, 2], see Sec. IV and
%   Eqs. (12)-(14) of [1].
%
%   Inputs:
%     kappa_target - scalar target kurtosis, 1 <= kappa_target <= 2
%                    (1: unit-modulus / sensing-optimal,
%                     2: Gaussian-like / communication-optimal)
%     weights      - (optional) struct loaded from
%                    adaptive_constellation_weights.mat; pass it when calling
%                    the function many times to avoid reloading the file
%
%   Outputs:
%     x - [M x 1] complex constellation points, zero mean and unit power
%     p - [M x 1] probabilities of the constellation points (sum(p) = 1)
%
%   The achieved kurtosis is sum(p .* abs(x).^4).
%
%   Reference:
%   [1] B. Geiger, L. Hampel, and L. Schmalen, "Joint power and
%       adaptive-constellation optimization for OFDM-ISAC systems," in Proc.
%       IEEE Conf. Integr. Sens. Commun. (ISAC), Lisbon, Portugal, Nov. 2026.
%       arXiv:2611.12166
%
%   License: MIT (see LICENSE in repository root)

arguments
    kappa_target (1,1) double {mustBeInRange(kappa_target, 1, 2)}
    weights struct = struct()
end

if isempty(fieldnames(weights))
    weights = load(fullfile(fileparts(mfilename('fullpath')), ...
                            'adaptive_constellation_weights.mat'));
end

M = weights.M;

%% Radii, Eq. (12): unit modulus for kappa = 1, radial shaping for kappa > 1
radial_out = mlp_forward(kappa_target, weights.radial_nn);              % [1 x M]
r = max(1 + sqrt(kappa_target - 1) * tanh(radial_out), weights.eps_R);

%% Phases, Eq. (13): bounded offset added to a uniform angular grid
phase_out = mlp_forward(kappa_target, weights.phase_nn);                % [1 x M]
theta_grid = 2*pi*(0:M-1)/M;
theta = theta_grid + pi * tanh(phase_out);

%% Probabilities, Eq. (14): uniform for kappa = 1, shaped for kappa > 1
logits = mlp_forward(kappa_target, weights.prob_nn);                    % [1 x M]
logits = min((kappa_target - 1) / weights.eps_Pr, 1) * logits;
p = exp(logits - max(logits));
p = p.' / sum(p);                                                       % [M x 1]

%% Constellation points, normalized to zero mean and unit average power
x = (r .* exp(1j*theta)).';                                             % [M x 1]
x = x - sum(p .* x);
x = x / sqrt(sum(p .* abs(x).^2));
end


function out = mlp_forward(in, nn)
% Fully connected NN with ReLU activations in the hidden layers and a
% linear output layer. Weights are stored as W_l [out x in], b_l [1 x out].
num_layers = numel(fieldnames(nn)) / 2;
out = in;
for l = 1:num_layers
    out = out * nn.("W" + l).' + nn.("b" + l);
    if l < num_layers
        out = max(out, 0);
    end
end
end
