function [ output_args ] = relabel_data( labels, n_classes )
%RELABEL_DATA Summary of this function goes here
%   Detailed explanation goes here
process_data = labels;
unit = 100 / n_classes;
current_rank = 0;
rank_begin_label = Inf;
rank_end_label = Inf;
for i = 1:n_classes
    % percentile of boundary in the beginning and in the end
    bound_begin = (i - 1) * unit;
    bound_end = i * unit;
    if bound_end > 100
        bound_end = 100;
    end
    label_begin = prctile(labels, bound_begin);
    label_end = prctile(labels, bound_end);

    if rank_begin_label ~= label_begin && rank_end_label ~= label_end
        current_rank = current_rank + 1;
        rank_begin_label = label_begin;
        rank_end_label = label_end;
    end

    if label_begin == label_end
        process_data(labels == label_end) = current_rank;
    elseif i == 1
        process_data((labels <= label_end) & (labels >= label_begin)) = current_rank;
    else
        process_data((labels <= label_end) & (labels > label_begin)) = current_rank;
    end
end

output_args = uint8(process_data);
end

