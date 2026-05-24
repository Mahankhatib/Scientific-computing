% 1. Define your plant coefficients (as + b) / (s^4 + cs^3 + ds^2 + es + f)
a = 1e07;   b = 7.26e08;
c = 5879;   d = 3.34e06;   e = 1.65e07;   f = 1.192e08;

% 2. Set your design target (Damping Ratio)
z = 0.707;  % Standard target for 5% overshoot

% 3. Define the coefficients of your derived wn equation
% w^4(4az^2-a) + w^3(4zb-2acz-8bz^3) + w^2(ad+4bcz^2-bc) + w^1(-2bdz) + w^0(eb-fa) = 0
coeffs = [
    a * (4 * z^2 - 1), ...                                 % w^4
    (4 * z * b) - (2 * a * c * z) - (8 * b * z^3), ...     % w^3
    (a * d) + (4 * b * c * z^2) - (b * c), ...             % w^2
    (-2 * b * d * z), ...                                  % w^1
    (e * b) - (f * a) ...                                  % w^0
];

% 4. Solve for roots
roots_wn = roots(coeffs);

fprintf('Potential natural frequencies (wn):\n');

% 5. Filter for real, positive frequencies and back-calculate parameters
for i = 1:length(roots_wn)
    r = roots_wn(i);
    
    % Check if the root is purely real and positive
    if isreal(r) && r > 0
        wn = real(r);
        
        % Calculate intermediate polynomial variables q and p
        q = d - (2 * z * c * wn) + (wn^2 * (4 * z^2 - 1));
        p = c - (2 * z * wn);
        
        % Back-calculate K from the s^0 matching equation
        K = (((wn^2)/b)*(d-(wn^2)-(2*c*z*wn)+(2*z*wn)^2))-(f/b)
        
        fprintf('-> Found wn = %.2f rad/s | Resulting K = %.4f\n', wn, K);
    end
end