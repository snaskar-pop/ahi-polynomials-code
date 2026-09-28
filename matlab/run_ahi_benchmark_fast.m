function run_ahi_benchmark_fast()
    clc; clear; yalmip('clear');
    
    % The cases from your paper
    cases = [
        4, 16;
        4, 24;
        6, 24;
        7, 28;
        8, 32;
        5, 30;
        6, 36;
        4, 40;
        10, 40
    ];

    fprintf('%-6s %-6s | %-10s %-10s %-10s | %-10s %-10s %-10s\n', ...
        'n', '2d', 'Dense N', 'Dense K', 'Time(s)', 'AHI N', 'AHI K', 'Time(s)');
    fprintf('%s\n', repmat('-', 1, 85));

    for k = 1:size(cases, 1)
        n = cases(k, 1);
        deg_total = cases(k, 2);
        
        % --- Reconstruct p(x) ---
        yalmip('clear');
        x = sdpvar(n, 1);
        power_per_var = deg_total / n;
        
        % Check for integer powers
        if mod(power_per_var, 1) ~= 0
            fprintf('%d, %d skipped (power issue)\n', n, deg_total);
            continue;
        end
        
        a = cell(n, 1);
        for i = 1:n, a{i} = x(i)^power_per_var; end
        lambda = (1/n) * ones(n, 1);
        sum_lambda_a = sum(lambda .* [a{:}].'); 
        prod_all_a = prod([a{:}]);
        sum_lambda_prod_not_i = 0;
        for i = 1:n
            prod_not_i = 1;
            for j = 1:n
                if j ~= i, prod_not_i = prod_not_i * a{j}; end
            end
            sum_lambda_prod_not_i = sum_lambda_prod_not_i + lambda(i) * prod_not_i;
        end
        p = sum_lambda_a * sum_lambda_prod_not_i - prod_all_a;

        % ====================================================
        % METHOD 1: Standard SOS (Dense)
        % ====================================================
        % SAFETY CHECK: Only run Dense if matrix size is small (< 600)
        basis_size = nchoosek(n + deg_total/2, deg_total/2);
        
        if basis_size < 600
            try
                tic;
                v_d = monolist(x, deg_total/2);
                K_dense = length(v_d);
                Q_dense = sdpvar(K_dense, K_dense);
                gamma = sdpvar(1);
                sos_poly = v_d' * Q_dense * v_d;
                F = [coefficients(p - gamma - sos_poly, x) == 0, Q_dense >= 0];
                optimize(F, -gamma, sdpsettings('solver','sdpt3','verbose',0));
                dense_time = toc;
                
                d_N_str = num2str(K_dense*(K_dense+1)/2); 
                d_K_str = num2str(K_dense);
                d_Time_str = sprintf('%.2f', dense_time);
            catch
                d_N_str = "Err"; d_K_str = "Err"; d_Time_str = "Err";
            end
        else
            % Too big, skip immediately
            d_N_str = "-"; d_K_str = "-"; d_Time_str = "OOM";
        end
        
        % ====================================================
        % METHOD 2: AHI Method (Sparse)
        % ====================================================
        % Clean up YALMIP to free memory from Dense attempt
        yalmip('clear'); 
        x = sdpvar(n, 1); % Re-declare variables
        % Re-build p (identical code to above)
        for i = 1:n, a{i} = x(i)^power_per_var; end
        sum_lambda_a = sum(lambda .* [a{:}].'); 
        prod_all_a = prod([a{:}]);
        sum_lambda_prod_not_i = 0;
        for i = 1:n
            prod_not_i = 1;
            for j = 1:n
                if j ~= i, prod_not_i = prod_not_i * a{j}; end
            end
            sum_lambda_prod_not_i = sum_lambda_prod_not_i + lambda(i) * prod_not_i;
        end
        p = sum_lambda_a * sum_lambda_prod_not_i - prod_all_a;

        tic;
        [~, monos] = coefficients(p, x);
        basis_exponents = [];
        for m_idx = 1:length(monos)
            degs = degree(monos(m_idx), x);
            basis_exponents = [basis_exponents; degs/2];
        end
        basis_exponents = unique(basis_exponents, 'rows');
        v_custom = [];
        for b_idx = 1:size(basis_exponents, 1)
            term = 1;
            for v_idx = 1:n
                term = term * x(v_idx)^basis_exponents(b_idx, v_idx);
            end
            v_custom = [v_custom; term];
        end
        
        K_ahi = length(v_custom);
        Q_ahi = sdpvar(K_ahi, K_ahi);
        gamma = sdpvar(1);
        sos_poly = v_custom' * Q_ahi * v_custom;
        F = [coefficients(p - gamma - sos_poly, x) == 0, Q_ahi >= 0];
        optimize(F, -gamma, sdpsettings('solver','sdpt3','verbose',0));
        ahi_time = toc;
        ahi_N = K_ahi*(K_ahi+1)/2;
        
        % Print Result
        fprintf('%-6d %-6d | %-10s %-10s %-10s | %-10d %-10d %-10.2f\n', ...
            n, deg_total, d_N_str, d_K_str, d_Time_str, ahi_N, K_ahi, ahi_time);
    end
end