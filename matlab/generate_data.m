line = 2; % 1 = line, 0 = ring

N = 4; % number of classes
R = 10 ; % radius
S = 50; % number of points per class
D = 10; % number of additional (noise) dimensions 
sigma = 4; % sigma for the normrnd spread

% the class centres
if line
    X = R * (1:N);
    Y = X;
else
    alphas = 0:(360/N):359;
    X = R * cos(deg2rad(alphas));
    Y = R * sin(deg2rad(alphas));
end


% feature vectors for the data and vector for the label
fvec = zeros(S*N, D+2);
lbl = zeros(S*N,1);

for i = 1:N
    rows = (i-1)*S+1:i*S;
    fvec(rows,1) = normrnd(X(i), sigma, [S,1]); % first dim
    fvec(rows,2) = normrnd(Y(i), sigma, [S,1]); % second dim
    fvec(rows,3:D+2) = normrnd(0, 1, [S,D]);    % additional D noise dims
    lbl(rows) = i;
end

figure;
hold on;
for i = 1:N
    rows = (i-1)*S+1:i*S;
    scatter(fvec(rows,1), fvec(rows,2));
end
hold off;

save('circle_ordinal.mat','fvec','lbl')