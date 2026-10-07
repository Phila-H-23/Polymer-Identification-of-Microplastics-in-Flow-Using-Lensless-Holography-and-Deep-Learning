function [eta, theta] = birefringence_eta_theta(I_0, I_45, I_90, I_135)

% BIREFRINGENCE_ETA_THETA

I_0   = double(I_0);
I_45  = double(I_45);
I_90  = double(I_90);
I_135 = double(I_135);

% Opposing analyzer-pair sums.
D0  = I_0 + I_90;
D45 = I_45 + I_135;

% Reject invalid denominators.
D0(D0 <= 0)   = NaN;
D45(D45 <= 0) = NaN;

% Four-frame algorithm without extinction setting.
A = (I_0   - I_90) ./ D0;
B = (I_135 - I_45) ./ D45;

% Magnitude term.
p = hypot(A, B);

% Physically valid range for asin().
valid = isfinite(p) & (p <= 1);
p(~valid) = NaN;

% Retardance, radians.
delta = asin(p);

% Slow-axis azimuth, modulo pi.
theta = mod(0.5 * atan2(A, B), pi);

% Orientation is undefined at zero retardance.
theta(~valid | p == 0) = NaN;

% Normalize.
eta   = delta / (pi/2);
theta = theta / pi;

end