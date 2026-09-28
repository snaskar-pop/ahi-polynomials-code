% ahi_benchmark.m
% Benchmarks for "The AHI family of sum of squares polynomials" (numerical section).
%
%   E1: single tight AHI polynomial      -> Level B vs SOS / Sparse SOS / DSOS   (cost comparison; all methods certify)
%   E2: k-fold products of binomial      -> Level B vs SOS / Sparse SOS / SOS on the Level-A basis / DSOS.
%       squares  prod_l (x_{2l-1}^2 - x_{2l}^2)^2   DSOS is provably infeasible here (Prop. 4.10 of the paper for k = 2).
%   E3: random products of two           -> DSOS membership as a function of the tightness parameter theta, plus costs.
%       overlapping AHI factors
%
% Test performed everywhere: certify f >= 0 (feasibility, gamma = 0).
% Tightness parameter:  c = sum(lambda.^2) + theta*((sum lambda)^2 - sum(lambda.^2)),
%   theta = 0 : every coefficient of the factor is >= 0 (trivially DSOS);  theta = 1 : AM-HM boundary, eq. (4) of the paper.
%
% Methods (all via YALMIP + MOSEK):
%   Level B      : one small SOS problem per factor, on the factor's half-support (Gram basis of size n(n-1)+1).
%   SOS          : YALMIP solvesos on the expanded product with Newton-polytope reduction and congruence block-diagonalization.
%   Sparse SOS   : as SOS plus YALMIP's correlative sparsity (sos.csp = 1).
%   SOS+LevelA   : generic SDP on the Minkowski-sum basis B_1+...+B_k (NOT the closed-form Level A certificate, which needs no solver).
%   DSOS         : diagonally dominant Gram matrix, built by hand as an LP (YALMIP has no built-in dsos), on (a) the reduced basis returned
%                  by solvesos ('DSOS red.') and (b) the full homogeneous basis ('DSOS full', only for tiny cases).
%
% Output: console table, ahi_benchmark_results.csv, and a checkpoint ALL_partial.mat after every case.
% Runtime with the defaults below (Intel i7 2.8 GHz, 16 GB): roughly 20-30 minutes, dominated by E2 at k = 4 (DSOS, ~6 min) and k = 5 (SOS, ~6 min).
% Guards (cfg.*) skip methods whose basis would be too large; "not attempted" in the output means skipped by a guard, NOT an out-of-memory crash.
% Requires: MATLAB R2023b, YALMIP (release 20250626 used), MOSEK.

clear; clc; rng(1);
RUN = [1 1 1];          % which experiments to run: [E1 E2 E3]
E2_k = [2 3 4 5 6 7];   % number of binomial-square factors; k >= 8 runs Level B only (extend to [.. 8 10 12] if desired)

cfg.tlimit      = 300;  % MOSEK time limit (s) per SOS/DSOS solve
cfg.maxTerms    = 1e5;  % do not expand f if the estimated number of monomials exceeds this
cfg.maxK_full   = 200;  % DSOS on the full homogeneous basis only if its size <= this (cross-check)
cfg.maxK_dsos   = 1500; % skip DSOS (reduced basis) if the largest block exceeds this (LP has ~K^2 variables)
cfg.maxProdB    = 300;  % generic SOS / sparse SOS / DSOS only if prod_l |B_l| <= this
cfg.maxK_levelA = 800;  % 'SOS+LevelA' only if prod_l |B_l| <= this (2187 at k=7 needs hours)

cfg.ops     = sdpsettings('solver','mosek','verbose',0);
cfg.ops_gen = sdpsettings('solver','mosek','verbose',0,'sos.newton',1,'sos.congruence',1);
cfg.ops_csp = sdpsettings('solver','mosek','verbose',0,'sos.newton',1,'sos.congruence',1,'sos.csp',1);
cfg.ops.mosek.MSK_DPAR_OPTIMIZER_MAX_TIME     = cfg.tlimit;
cfg.ops_gen.mosek.MSK_DPAR_OPTIMIZER_MAX_TIME = cfg.tlimit;
cfg.ops_csp.mosek.MSK_DPAR_OPTIMIZER_MAX_TIME = cfg.tlimit;

ALL = [];

% =====================================================================
% E1: single tight AHI polynomial (n squared monomials in n variables, degree 2d), as in Table 1 of the paper
% =====================================================================
if RUN(1)
    fprintf('\n################ E1: single AHI polynomial (theta = 1) ################\n');
    %        n  2d  tryGeneric   (set tryGeneric = 1 to also attempt SOS/Sparse/DSOS; large cases may hang or run out of memory)
    cfg1 = [ 4  16  1;
             4  24  1;
             6  24  1;
             7  28  1;
             8  32  1;
             5  30  1;
             6  36  1;
             4  40  1;
            10  40  0 ];
    for r = 1:size(cfg1,1)
        n = cfg1(r,1); deg2d = cfg1(r,2); tg = cfg1(r,3);
        yalmip('clear');
        x = sdpvar(n,1);
        alpha  = rand_alpha(n, n, deg2d/n);
        lambda = rand(n,1); lambda = lambda/sum(lambda);
        [p, B] = make_ahi(x, alpha, lambda, tight_c(lambda, 1));
        R = run_case(sprintf('E1: AHI n=%d, deg=%d', n, deg2d), n, {p}, {B}, cfg, x, tg);
        ALL = [ALL; tag_rows(R, 'E1', 1, NaN)]; save('ALL_partial.mat','ALL'); %#ok<AGROW>
    end
end

% =====================================================================
% E2: products of binomial squares (Thm 4.8 / Prop 4.10): PiAHI, provably NOT SDSOS, hence not DSOS
% =====================================================================
if RUN(2)
    fprintf('\n################ E2: PiAHI not in DSOS (products of binomial squares) ################\n');
    for k = E2_k
        yalmip('clear');
        N = 2*k;  x = sdpvar(N,1);
        factors = cell(k,1); bases = cell(k,1);
        for l = 1:k
            alpha = zeros(2,N);  alpha(1,2*l-1) = 2;  alpha(2,2*l) = 2;
            [factors{l}, bases{l}] = make_ahi(x, alpha, [1;1], 4);       % (x_a^2 - x_b^2)^2
        end
        R = run_case(sprintf('E2: k=%d binomial squares, N=%d', k, N), N, factors, bases, cfg, x, 1);
        ALL = [ALL; tag_rows(R, 'E2', NaN, NaN)]; save('ALL_partial.mat','ALL'); %#ok<AGROW>
    end
end

% =====================================================================
% E3: random products of overlapping AHI factors; DSOS membership decided by the DSOS LP
% =====================================================================
if RUN(3)
    fprintf('\n################ E3: random PiAHI products, DSOS membership sweep ################\n');
    %         N  m  ds  k
    cfg3 = [  4  3  2   2;
              5  4  2   2;
              6  4  2   2 ];
    thetas = [0 0.5 1];
    nseed  = 3;
    mem = NaN(size(cfg3,1), numel(thetas), nseed);
    for r = 1:size(cfg3,1)
        N = cfg3(r,1); m = cfg3(r,2); ds = cfg3(r,3); k = cfg3(r,4);
        for ti = 1:numel(thetas)
            for sd = 1:nseed
                yalmip('clear');
                x = sdpvar(N,1);
                factors = cell(k,1); bases = cell(k,1);
                for l = 1:k
                    alpha  = rand_alpha(m, N, ds);
                    lambda = rand(m,1); lambda = lambda/sum(lambda);
                    [factors{l}, bases{l}] = make_ahi(x, alpha, lambda, tight_c(lambda, thetas(ti)));
                end
                nm = sprintf('E3: N=%d m=%d ds=%d k=%d theta=%.2f seed=%d', N, m, ds, k, thetas(ti), sd);
                R = run_case(nm, N, factors, bases, cfg, x, 1);
                mem(r,ti,sd) = dsos_class(R);
                ALL = [ALL; tag_rows(R, 'E3', thetas(ti), sd)]; save('ALL_partial.mat','ALL'); %#ok<AGROW>
            end
        end
    end
    fprintf('\nE3 summary: number of random products that are DSOS-feasible (out of %d seeds)\n', nseed);
    fprintf('%-22s', 'config (N,m,ds,k)');
    for ti = 1:numel(thetas), fprintf('theta=%-6.2f', thetas(ti)); end
    fprintf('\n');
    for r = 1:size(cfg3,1)
        fprintf('%-22s', mat2str(cfg3(r,:)));
        for ti = 1:numel(thetas)
            v = squeeze(mem(r,ti,:));
            fprintf('%d/%-10d', nansum(v), sum(~isnan(v)));
        end
        fprintf('\n');
    end
end

% ---------------- save ----------------
if ~isempty(ALL)
    T = struct2table(ALL);
    writetable(T, 'ahi_benchmark_results.csv');
    fprintf('\nSaved %d rows to ahi_benchmark_results.csv\n', height(T));
end

% =====================================================================
function R = run_case(name, N, factors, bases, cfg, x, tryGeneric)
    R = struct('case',{},'method',{},'status',{},'total',{},'solver',{},'K',{});
    k  = numel(factors);
    nt = cellfun(@(p) numel(getvariables(p)), factors);
    Dtot = 0; for l = 1:k, Dtot = Dtot + degree(factors{l}); end
    Dh = Dtot/2;
    n_even    = exp(gammaln(N+Dh) - gammaln(Dh+1) - gammaln(N));
    est_terms = min(prod(nt), n_even);
    prodB     = prod(cellfun(@(v) length(v), bases));
    fprintf('\n=== %s ===\n', name);
    fprintf('terms per factor: %s | deg f = %d | est. #monomials of f <= %.3g | prod|B_l| = %g\n', ...
            mat2str(nt(:)'), Dtot, est_terms, prodB);
    fprintf('%-12s | %-34s %-9s %-9s %-7s\n', 'method','status','total(s)','solver(s)','K');
    fprintf('%s\n', repmat('-',1,80));

    % ---------------- Level B: k small subproblems ----------------
    tot = 0; sv = 0; Kmax = 0; status = 'OK';
    for l = 1:k
        [st, t, ts, K] = try_sos(sos(factors{l}), cfg.ops, bases{l});
        tot = tot + t; sv = sv + ts; Kmax = max(Kmax, K);
        if ~strcmp(st,'OK'), status = st; end
    end
    R = addrow(R, name, 'Level B', status, tot, sv, Kmax);

    need_f = prodB <= max(cfg.maxProdB, cfg.maxK_levelA);
    if est_terms > cfg.maxTerms || ~need_f
        msg = sprintf('not attempted (prod|B_l|=%.3g)', prodB);
        if k > 1, R = addrow(R, name, 'SOS+LevelA', msg, NaN, NaN, NaN); end
        R = skip_generic(R, name, msg);
        return
    end

    % ---------------- expand the product ----------------
    try
        f = factors{1};
        for l = 2:k, f = f * factors{l}; end
    catch ME
        R = addrow(R, name, 'expand f', classify(ME), NaN, NaN, NaN);
        return
    end

    % ---------------- SOS with the structure-aware basis B_1+...+B_k (Level A) ----------------
    if k > 1
        if prodB <= cfg.maxK_levelA
            [st, t, ts, K] = sos_levelA(f, x, N, bases, cfg.ops_gen);
            R = addrow(R, name, 'SOS+LevelA', st, t, ts, K);
        else
            R = addrow(R, name, 'SOS+LevelA', sprintf('skipped (prod|B_l|=%.3g)', prodB), NaN, NaN, prodB);
        end
    end

    if ~tryGeneric || prodB > cfg.maxProdB
        R = skip_generic(R, name, 'not attempted (guard / tryGeneric=0)');
        return
    end

    % ---------------- generic SOS ----------------
    [st, t, ts, K, Bas] = try_sos(sos(f), cfg.ops_gen, []);
    R = addrow(R, name, 'SOS', st, t, ts, K);

    % ---------------- sparse SOS (csp) ----------------
    [st, t, ts, K] = try_sos(sos(f), cfg.ops_csp, []);
    R = addrow(R, name, 'Sparse SOS', st, t, ts, K);

    % ---------------- DSOS on the reduced (Newton + congruence) basis returned by solvesos ----------------
    if isempty(Bas)
        R = addrow(R, name, 'DSOS red.', 'skipped (SOS gave no basis)', NaN, NaN, NaN);
    elseif max(cellfun(@(v) length(v), Bas)) > cfg.maxK_dsos
        R = addrow(R, name, 'DSOS red.', sprintf('skipped (K=%d)', max(cellfun(@(v) length(v), Bas))), NaN, NaN, NaN);
    else
        [st, t, ts, Kred] = dsos_reduced(f, x, N, Bas, cfg.ops);
        R = addrow(R, name, 'DSOS red.', st, t, ts, Kred);
    end

    % ---------------- DSOS on the full homogeneous basis (cross-check, tiny cases only) ----------------
    Kfull = round(exp(gammaln(N+Dh) - gammaln(Dh+1) - gammaln(N)));
    if Kfull > cfg.maxK_full
        R = addrow(R, name, 'DSOS full', sprintf('skipped (K=%d)', Kfull), NaN, NaN, Kfull);
    else
        [st, t, ts] = dsos_blocks(f, x, N, {hom_exponents(N, Dh)}, cfg.ops);
        R = addrow(R, name, 'DSOS full', st, t, ts, Kfull);
    end
end

function R = skip_generic(R, name, msg)
    R = addrow(R, name, 'SOS',        msg, NaN, NaN, NaN);
    R = addrow(R, name, 'Sparse SOS', msg, NaN, NaN, NaN);
    R = addrow(R, name, 'DSOS red.',  msg, NaN, NaN, NaN);
    R = addrow(R, name, 'DSOS full',  msg, NaN, NaN, NaN);
end

function R = addrow(R, name, method, status, t, ts, K)
    fprintf('%-12s | %-34s %-9.3f %-9.3f %-7g\n', method, status, t, ts, K);
    R(end+1) = struct('case',name,'method',method,'status',status,'total',t,'solver',ts,'K',K);
end

function R = tag_rows(R, expname, theta, seed)
    for i = 1:numel(R)
        R(i).exp = expname; R(i).theta = theta; R(i).seed = seed;
    end
    R = R(:);
end

function c = tight_c(lambda, theta)
    % c such that K = sum(l.^2) - c = -2*theta*sum_{i<j} l_i l_j  (theta = 1: AM-HM boundary)
    c = sum(lambda.^2) + theta*(sum(lambda)^2 - sum(lambda.^2));
end

function c = dsos_class(R)
    % 1 = DSOS-feasible, 0 = DSOS-infeasible, NaN = not decided
    c = NaN;
    for i = 1:numel(R)
        if strcmp(R(i).method, 'DSOS red.')
            if strcmp(R(i).status, 'OK'), c = 1;
            elseif strncmp(R(i).status, 'no cert', 7), c = 0; end
        end
    end
end

function [st, t, ts, K] = sos_levelA(f, x, N, bases, ops)
    % SOS on the basis B = B_1 + ... + B_k (Minkowski sum of the factors' half-supports)
    st = 'OK'; t = NaN; ts = NaN; K = NaN;
    try
        tic;
        Eb = cellfun(@(v) exps_of(v, x, N), bases, 'UniformOutput', false);
        E = Eb{1};
        for l = 2:numel(Eb)
            F = Eb{l};
            [I,J] = ndgrid(1:size(E,1), 1:size(F,1));
            E = unique(E(I(:),:) + F(J(:),:), 'rows');
        end
        vc = cell(size(E,1),1);
        for r = 1:size(E,1), vc{r} = mono(x, E(r,:)); end
        v = cat(1, vc{:});
        t0 = toc;
        [st, t, ts, K] = try_sos(sos(f), ops, v);
        t = t + t0;
    catch ME
        st = classify(ME);
    end
end

function alpha = rand_alpha(m, N, ds)
    % m distinct squared monomials of degree ds in N variables (even exponents)
    while true
        alpha = zeros(m,N);
        for i = 1:m
            for u = 1:ds/2
                kk = randi(N);
                alpha(i,kk) = alpha(i,kk) + 2;
            end
        end
        if size(unique(alpha,'rows'),1) == m, break; end
    end
end

function [p, B] = make_ahi(x, alpha, lambda, c)
    % AHI polynomial (paper eq. (1)) with s_i = x^alpha(i,:), and its Level-B Gram basis (half-support)
    [m,N] = size(alpha);
    s = cell(m,1);
    for i = 1:m, s{i} = mono(x, alpha(i,:)); end
    t1 = 0; t2 = 0; P = 1;
    for i = 1:m
        t1 = t1 + lambda(i)*s{i};
        others = 1;
        for j = 1:m, if j ~= i, others = others * s{j}; end, end
        t2 = t2 + lambda(i)*others;
        P = P * s{i};
    end
    p = t1*t2 - c*P;
    B = [];
    for i = 1:m
        for j = (i+1):m
            sk = zeros(1,N);
            for kk = 1:m, if kk~=i && kk~=j, sk = sk + alpha(kk,:); end, end
            B = [B; mono(x, alpha(i,:) + 0.5*sk); mono(x, alpha(j,:) + 0.5*sk)]; %#ok<AGROW>
        end
    end
    B = [B; mono(x, sum(alpha,1)/2)];
    B = unique(B);
end

function m = mono(x, e)
    % monomial prod_v x(v)^e(v) with integer exponents (sdpvar has no vector power)
    e = round(e(:));
    m = 1;
    for v = 1:numel(e)
        if e(v) > 0, m = m * x(v)^e(v); end
    end
end

function [status, t, ts, K, Bas] = try_sos(Fsos, opts, basis)
    % Bas: cell array of the monomial-basis blocks used by solvesos (empty on failure)
    status = 'OK'; t = NaN; ts = NaN; K = NaN; Bas = {};
    try
        tic;
        if isempty(basis)
            [sol,m] = solvesos(Fsos, [], opts);
        else
            [sol,m] = solvesos(Fsos, [], opts, [], basis);
        end
        t = toc;
        if isfield(sol,'solvertime'), ts = sol.solvertime; end
        Bas = flatten_basis(m);
        if ~isempty(Bas)
            K = max(cellfun(@(v) length(v), Bas));
        end
        if sol.problem ~= 0
            status = ['no cert: ' strtrim(sol.info)];
        end
    catch ME
        status = classify(ME);
    end
end

function L = flatten_basis(m)
    L = {};
    if iscell(m)
        for i = 1:numel(m), L = [L, flatten_basis(m{i})]; end %#ok<AGROW>
    elseif ~isempty(m)
        L = {m};
    end
end

function E = exps_of(v, x, n)
    % exponent matrix (length(v) x n) of a vector v of monomials (coefficient 1) in x
    Bv = getbase(v);  vv = getvariables(v);
    mt = yalmip('monomtable');  xi = getvariables(x);
    k = size(Bv,1);  E = zeros(k,n);
    for r = 1:k
        row = full(Bv(r,2:end));
        idx = find(row);
        if numel(idx) > 1 || (~isempty(idx) && abs(row(idx)-1) > 1e-9)
            error('basis element is not a plain monomial');
        end
        if ~isempty(idx)
            E(r,:) = full(mt(vv(idx), xi));
        end
    end
end

function [status, t, ts, Kmax] = dsos_reduced(p, x, n, Bas, opts)
    status = 'OK'; t = NaN; ts = NaN; Kmax = NaN;
    try
        tic;
        Eb = cellfun(@(v) exps_of(v, x, n), Bas, 'UniformOutput', false);
        Kmax = max(cellfun(@(E) size(E,1), Eb));
        t0 = toc;
        [status, t, ts] = dsos_blocks(p, x, n, Eb, opts);
        t = t + t0;
    catch ME
        status = classify(ME);
    end
end

function [status, t, ts] = dsos_blocks(p, x, n, Eb, opts)
    % DSOS certificate:  p = sum_b v_b' Q_b v_b,  every Q_b diagonally dominant (=> PSD).
    % Eb{b} = exponent matrix of block b's monomial basis. Built numerically as an LP.
    status = 'OK'; t = NaN; ts = NaN;
    try
        tic;
        po = 0; ro = 0;
        Sc = {}; wc = {}; Rr = []; Rc = []; dg = [];
        for b = 1:numel(Eb)
            E = Eb{b}; Kb = size(E,1);
            [I,J] = find(triu(true(Kb)));
            Pb  = numel(I);
            Sc{end+1} = E(I,:) + E(J,:);          %#ok<AGROW>
            wc{end+1} = 1 + double(I ~= J);       %#ok<AGROW>
            isd = (I == J);  od = find(~isd);
            dg  = [dg; po + find(isd)];           %#ok<AGROW>  ordered as block rows 1..Kb
            Rr  = [Rr; ro + I(od); ro + J(od)];   %#ok<AGROW>
            Rc  = [Rc; po + od; po + od];         %#ok<AGROW>
            po  = po + Pb;  ro = ro + Kb;
        end
        S = vertcat(Sc{:});  w = vertcat(wc{:});  P = po;  Ktot = ro;
        [U,~,ic] = unique(S,'rows');
        M = sparse(ic, (1:P)', w, size(U,1), P);  % coefficient-matching matrix

        % numeric coefficients / exponents of p via YALMIP's internal monomial table
        Bp = getbase(p);
        pv = getvariables(p);
        mt = yalmip('monomtable');
        xi = getvariables(x);
        c  = full(Bp(1, 2:end)).';
        Ep = full(mt(pv, xi));
        if abs(full(Bp(1,1))) > 0
            Ep = [Ep; zeros(1,n)]; c = [c; full(Bp(1,1))];
        end
        if ~all(ismember(Ep, U, 'rows'))
            status = 'no cert: monomial outside basis products';
            t = toc; return
        end
        [tf,loc] = ismember(U, Ep, 'rows');
        target = zeros(size(U,1),1);
        target(tf) = c(loc(tf));

        R  = sparse(Rr, Rc, 1, Ktot, P);
        q  = sdpvar(P,1);
        tt = sdpvar(P,1);
        Fc = [M*q == target, tt >= q, tt >= -q, q(dg) >= R*tt];
        sol = optimize(Fc, [], opts);
        t = toc;
        if isfield(sol,'solvertime'), ts = sol.solvertime; end
        if sol.problem ~= 0
            status = ['no cert: ' strtrim(sol.info)];
        end
    catch ME
        status = classify(ME);
    end
end

function E = hom_exponents(n, d)
    % all exponent vectors in N^n with sum d (stars and bars)
    if n == 1, E = d; return; end
    C = nchoosek(1:n+d-1, n-1);
    K = size(C,1);
    E = zeros(K,n);
    prev = zeros(K,1);
    for k = 1:n-1
        E(:,k) = C(:,k) - prev - 1;
        prev = C(:,k);
    end
    E(:,n) = (n+d-1) - prev;
end

function s = classify(ME)
    m = lower(ME.message);
    if contains(m,'memory') || contains(m,'array size') || contains(m,'exceeds')
        s = 'out of memory';
    else
        loc = '';
        if ~isempty(ME.stack)
            loc = sprintf(' @%s:%d', ME.stack(1).name, ME.stack(1).line);
        end
        s = ['error: ' ME.message(1:min(100,end)) loc];
    end
end

