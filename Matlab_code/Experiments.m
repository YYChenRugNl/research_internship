data=load('C:\Users\Yukki\Desktop\RIntern\benchmark\Machine-Cpu\machine.data');
[len, dim] = size(data);
data(:,dim) = relabel_data(data(:,dim), 10);
[data, labels] = up_sample(data, data(:,dim));

nb_folds = 5;
Indices = crossvalind('Kfold',len,nb_folds);
testError1_arr = zeros(nb_folds,1);
testError_arr = zeros(nb_folds,1);

for i_fold=1:nb_folds
    train_set = data(Indices ~= i_fold, :);
    test_set = data(Indices == i_fold, :);
    
    train_dataset = train_set(:,1:dim-1);
    train_label = train_set(:,dim);
        
    test_dataset = test_set(:,1:dim-1);
    test_label = test_set(:,dim);
    
%     GMLVQ_model = GMLVQ_train(train_dataset, train_label, 'PrototypesPerClass', 5, 'nb_epochs', 2500, 'Display', 'off');
%     estimatedTrainLabels = GMLVQ_classify(train_dataset, aOGMLVQ_model);
%     testError1 = mean( abs(train_label - estimatedTrainLabels ));
    
    aOGMLVQ_model = OGMLVQ_train_modified(train_dataset, train_label, 'PrototypesPerClass', 3, 'nb_epochs', 250, 'Lmin', 1);
    estimatedTrainLabels = GMLVQ_classify(train_dataset, aOGMLVQ_model);
%     testError = mean( test_label ~= estimatedTrainLabels );
    testError = mean( abs(train_label - estimatedTrainLabels ));

%     testError1_arr(i_fold) = testError1;
    testError_arr(i_fold) = testError;
    
%     disp(testError1);
    disp(testError);
end
% mean(testError1_arr)
mean(testError_arr)



