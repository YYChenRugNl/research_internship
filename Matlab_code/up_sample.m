function [ output_args1, output_args2 ] = up_sample(x, y)
        [init_shape, dim] = size(x);
        [C,ia,ic] = unique(y);
        a_counts = accumarray(ic,1);
        value_counts = [C, a_counts];
        max_count = max(value_counts(:, 2));
        new_x = zeros(init_shape, dim);
        new_y = zeros(init_shape, 1);

        for idx = 1:size(C,1)
            cls = C(idx);
            subset_x = x(y == cls, :);
            subset_x = datasample(subset_x, max_count, 'Replace', true);
            subset_y = ones(max_count, 1) * cls;
            
            sub_idx = 1;
            for index = 1+(idx-1)*max_count : idx*max_count
                new_x(index, :) = subset_x(sub_idx, :);
                sub_idx = sub_idx + 1;
            end
            new_y(1+(idx-1)*max_count : idx*max_count) = subset_y(:);
        end
        
        output_args1=  new_x;
        output_args2 = new_y;
end

