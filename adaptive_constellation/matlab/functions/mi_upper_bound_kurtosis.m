function mi_ub = mi_upper_bound_kurtosis(kappa, snr_dB)
% MI_UPPER_BOUND_KURTOSIS  Upper bound on the MI under a kurtosis constraint.
%
%   mi_ub = mi_upper_bound_kurtosis(kappa, snr_dB)
%
%   Evaluates the upper bound on the mutual information (bit/symbol) of the
%   complex AWGN channel y = x + w, w ~ CN(0, sigma^2), for a unit-power
%   input x with kurtosis E{|x|^4} = kappa, see Eqs. (8)-(11) of [1] with
%   h = 1 and P = 1, and Eqs. (23)-(32) of [2].
%
%   The bound is obtained from the maximum-entropy output PDF
%   exp(gamma0 + gamma2*|y|^2 + gamma4*|y|^4) that satisfies the second- and
%   fourth-order moment constraints of the received signal. This is the
%   same computation as in mi_bounds_kurtosis_constraint.m in the
%   repository root, packaged as a function for multiple SNR values.
%
%   Inputs:
%     kappa  - vector of kurtosis values, 1 <= kappa <= 2
%     snr_dB - communication SNR 1/sigma^2 in dB
%
%   References:
%   [1] B. Geiger, L. Hampel, and L. Schmalen, "Joint power and
%       adaptive-constellation optimization for OFDM-ISAC systems," in Proc.
%       IEEE Conf. Integr. Sens. Commun. (ISAC), Lisbon, Portugal, Nov. 2026.
%       arXiv:2611.12166
%   [2] B. Geiger et al., "Constellation shaping for OFDM-ISAC systems:
%       From theoretical bounds to practical implementation," IEEE Trans.
%       Commun., 2026. arXiv:2509.04055
%
%   License: MIT (see LICENSE in repository root)

arguments
    kappa double {mustBeInRange(kappa, 1, 2)}
    snr_dB (1,1) double
end

sigma2 = 10^(-snr_dB/10);
entropy_awgn = log2(pi*exp(1)*sigma2);

% Moment constraints of the received signal, Eqs. (8) and (9)
C0 = 1;
C2 = 1 + sigma2;
C4 = kappa + 4*sigma2 + 2*sigma2^2;

mi_ub = zeros(size(kappa));
for i = 1:numel(kappa)
    if kappa(i) >= 2 - 1e-9
        % Gaussian input: maximum-entropy output is Gaussian -> capacity
        mi_ub(i) = log2(1 + 1/sigma2);
        continue;
    end

    % Nonlinear equation in gamma2 (Eq. (29) in [2]); erfcx(z) = exp(z^2)*erfc(z)
    % is used for numerical stability
    f = @(g2) (C2*(g2*C2 + 1)/C4(i) - g2) ...
        .* sqrt(pi*C4(i)/(2*(g2*C2 + 1))) ...
        .* erfcx(-g2*sqrt(C4(i)/(2*(g2*C2 + 1)))) - C0;

    % Bracket the root. gamma2 > -1/C2 keeps the square root real; f tends
    % to zero at this boundary (trivial solution), so the bracket is found
    % by scanning logarithmically spaced offsets from the boundary for the
    % first sign change of f from positive to negative.
    g2_hi = 10;
    while f(g2_hi) > 0
        g2_hi = 2*g2_hi;
    end
    g2_grid = -1/C2 + logspace(-8, log10(g2_hi + 1/C2), 400);
    f_grid = arrayfun(f, g2_grid);
    idx_pos = find(f_grid > 1e-14, 1);                % skip round-off region
    idx = idx_pos - 1 + find(f_grid(idx_pos:end) < 0, 1);
    if isempty(idx_pos) || isempty(idx)
        error('mi_upper_bound_kurtosis:bracket', ...
              'No valid bracket found for kappa = %g.', kappa(i));
    end
    gamma2 = fzero(f, g2_grid([idx-1, idx]));

    % gamma0 and gamma4 (Eqs. (30) and (31) in [2])
    gamma0 = log(1/pi*(C2*(gamma2*C2 + 1)/C4(i) - gamma2));
    gamma4 = -(gamma2*C2 + 1)/(2*C4(i));

    % Differential entropy of the output (Eq. (32) in [2]) and bound, Eq. (11)
    entropy_out = -(gamma0 + C2*gamma2 + C4(i)*gamma4)/log(2);
    mi_ub(i) = entropy_out - entropy_awgn;
end
end
