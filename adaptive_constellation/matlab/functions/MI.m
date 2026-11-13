function mi = MI(x, p, snr_dB, num_nodes)
% MI  Mutual information of a discrete constellation over the AWGN channel.
%
%   mi = MI(x, p, snr_dB)
%   mi = MI(x, p, snr_dB, num_nodes)
%
%   Computes the mutual information I(X;Y) (bit/symbol) of the constellation
%   x with probabilities p over the AWGN channel y = x + w, w ~ CN(0, sigma^2),
%   with SNR = 1/sigma^2 (x has unit average power):
%
%     I(X;Y) = -sum_i p_i E_w{ log2( sum_j p_j
%                 exp((|w|^2 - |x_i - x_j + w|^2) / sigma^2) ) }.
%
%   The expectation over the complex Gaussian noise w is evaluated with a
%   two-dimensional Gauss-Hermite quadrature (num_nodes x num_nodes nodes).
%   In contrast to a Monte Carlo estimate, the result is deterministic and
%   free of sampling noise.
%
%   Inputs:
%     x         - [M x 1] complex constellation points (unit average power)
%     p         - [M x 1] probabilities of the constellation points
%     snr_dB    - SNR 1/sigma^2 in dB
%     num_nodes - (optional) number of Gauss-Hermite nodes per real
%                 dimension, default 32
%
%   License: MIT (see LICENSE in repository root)

arguments
    x {mustBeNumeric, mustBeVector}
    p {mustBeNumeric, mustBeVector, mustBeNonnegative}
    snr_dB (1,1) double
    num_nodes (1,1) double {mustBeInteger, mustBePositive} = 32
end

x = x(:);
p = p(:) / sum(p);
sigma2 = 10^(-snr_dB/10);

% Gauss-Hermite nodes t and weights g for int exp(-t^2) f(t) dt
% (Golub-Welsch: eigenvalues/eigenvectors of the Jacobi matrix)
beta = sqrt((1:num_nodes-1) / 2);
[V, D] = eig(diag(beta, 1) + diag(beta, -1));
t = diag(D);
g = sqrt(pi) * V(1, :).'.^2;

% Quadrature points w = sigma*(t_k + 1j*t_l) of CN(0, sigma^2), stored as
% [Re{w}, Im{w}], with weights g_k*g_l/pi (summing to one)
[t_re, t_im] = ndgrid(t, t);
w = sqrt(sigma2) * [t_re(:), t_im(:)];                        % [K^2 x 2]
q = (g * g.') / pi;
q = q(:);                                                     % [K^2 x 1]

log_p = log(p.');                                             % [1 x M]
mi = 0;
for i = find(p > 0).'
    % Exponents (|w|^2 - |d_ij + w|^2)/sigma^2 + log(p_j) with
    % d_ij = x_i - x_j, using |w|^2 - |d + w|^2 = -|d|^2 - 2 Re{d^* w}
    d = x(i) - x.';                                           % [1 x M]
    e = (-abs(d).^2 - 2 * (w * [real(d); imag(d)])) / sigma2 + log_p;
    e_max = max(e, [], 2);                                    % [K^2 x 1]
    log_sum = e_max + log(sum(exp(e - e_max), 2));            % log-sum-exp
    mi = mi - p(i) * (q.' * log_sum);
end
mi = mi / log(2);
end
