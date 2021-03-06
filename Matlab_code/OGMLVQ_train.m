function [model, varargout] = OGMLVQ_train(trainSet, trainLab, varargin)
%GMLVQ_trai.m - trains the Generalized Matrix LVQ algorithm
%NOTE: minimal requirement version 7.4.0.336 (R2007a) 
%  example for usage:
%  trainSet = [1,2,3;4,5,6;7,8,9];
%  trainLab = [1;1;2];
%  GMLVQ_model=GMLVQ_train(trainSet,trainLab); % minimal parameters required
%  estimatedTrainLabels = GMLVQ_classify(trainSet, GMLVQ_model);
%  trainError = mean( trainLab ~= estimatedTrainLabels );
%
% input: 
%  trainSet : matrix with training samples in its rows
%  trainLab : vector with the labels of the training set
% optional parameters:
%  PrototypesPerClass: (default=1) the number of prototypes per class used. This could
%  be a number or a vector with the number for each class
%  initialPrototypes : (default=[]) a set of prototypes to start with. If not given initialization near the class means
%  initialMatrix     : the matrix omega to start with. If not given random
%  initialization for rectangular matrices and Unity for squared omega
%  dim               : (default=nb of features for training) the maximum rank or projection dimension
%  regularization    : (default=0) values usually between 0 and 1 treat with care. 
%  Regularizes the eigenvalue spectrum of omega'*omega to be more homogeneous
%  testSet           : (default=[]) an optional test set used to compute
%  the test error. The last column is expected to be a label vector
%  comparable        : (default=0) a flag which resets the random generator
%  to produce comparable results if set to 1
%  optimization      : (default=fminlbfgs) indicates which optimization is used: sgd or fminlbfgs
% parameter for the stochastic gradient descent sgd
%  nb_epochs             : (default=100) the number of epochs for sgd
%  learningRatePrototypes: (default=[]) the learning rate for the prototypes. 
%  Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  learningRateMatrix    : (default=[]) the learning rate for the matrix.
%  Could be the start and end value used for a sigmoidal spectrum or a vector of length nb_epochs
%  MatrixStart           : (default=1) the epoch to start the matrix training
% 
%
% output: the GMLVQ model with prototypes w their labels c_w and the matrix omega 
%  optional output:
%  initialization : a struct containing the settings
%  trainError     : error in the training set
%  testError      : error in the training set (only computed if 'testSet' is given)
%  costs          : the output of the cost function
% 
% Citation information:
% Petra Schneider, Michael Biehl, Barbara Hammer: 
% Adaptive Relevance Matrices in Learning Vector Quantization. Neural Computation 21(12): 3532-3561 (2009)
% 
% K. Bunte, P. Schneider, B. Hammer, F.-M. Schleif, T. Villmann and M. Biehl, 
% Limited Rank Matrix Learning - Discriminative Dimension Reduction and Visualization, 
% Neural Networks, vol. 26, nb. 4, pp. 159-173, 2012.
% 
% P. Schneider, K. Bunte, B. Hammer and M. Biehl, Regularization in Matrix Relevance Learning, 
% IEEE Transactions on Neural Networks, vol. 21, nb. 5, pp. 831-840, 2010.
% 
% Kerstin Bunte (modified based on the code of Marc Strickert http://www.mloss.org/software/view/323/ and Petra Schneider)
% uses the Fast Limited Memory Optimizer fminlbfgs.m written by Dirk-Jan Kroon available at the MATLAB central
% kerstin.bunte@googlemail.com
% Fri Nov 09 14:13:52 CEST 2012
%
% Conditions of GNU General Public License, version 2 and BSD License apply.
% See file 'license-gpl2.txt' and 'BSD_license.txt' enclosed in this package.
% Programs are not for use in critical applications!
%
nout = max(nargout,1)-1;
p = inputParser;   % Create an instance of the class.
p.addRequired('trainSet', @isfloat);
p.addRequired('trainLab', @(x) length(x)==size(trainSet,1) & isnumeric(x));
p.addParamValue('PrototypesPerClass', ones(1,length(unique(trainLab))), @(x)(sum(~(x-floor(x)))/length(x)==1 && (length(x)==length(unique(trainLab)) || length(x)==1)));
p.addParamValue('initialPrototypes',[], @(x)(size(x,2)-1==size(trainSet,2) && isfloat(x)));
p.addParamValue('initialMatrix',[], @(x)(size(x,2)==size(trainSet,2) && isfloat(x)));
p.addParamValue('dim',size(trainSet,2), @(x)(~(x-floor(x)) && x<=size(trainSet,2) && x>0));
p.addParamValue('regularization',0, @(x)(isfloat(x) && x>=0));
p.addOptional('testSet', [], @(x)(size(x,2)-1)==size(trainSet,2) & isfloat(x));
p.addOptional('comparable', 0, @(x)(~(x-floor(x))));

% parameter for the stochastic gradient descent
p.addOptional('nb_epochs', 100, @(x)(~(x-floor(x))));
p.addParamValue('learningRatePrototypes', [], @(x)(isfloat(x) || isa(x,'function_handle'))); % && (length(x)==2 || length(x)==p.Results.epochs)
p.addParamValue('learningRateMatrix', [], @(x)(isfloat(x)  || isa(x,'function_handle')));
p.addOptional('MatrixStart', 1, @(x)(~(x-floor(x))));

p.addParamValue ('decay_factor',0,@(x)x>=0 & isfloat(x));

p.addOptional('Lmin',1,@(x)isfloat(x));

p.CaseSensitive = true;
p.FunctionName = 'OGMLVQ';
% Parse and validate all input arguments.
p.parse(trainSet, trainLab, varargin{:});

%%% check if results should be comparable
if p.Results.comparable,
    rng('default');
end
%%% set useful variables
nb_samples = size(trainSet,1);
nb_features = size(trainSet,2);
% labels should be a row vector
if size(trainLab,1)~=nb_samples, trainLab = trainLab';end

classes = unique(trainLab);
nb_classes = length(classes);
dim = p.Results.dim;
MatrixStart = p.Results.MatrixStart;
testSet = p.Results.testSet;
% global regularization;
regularization = p.Results.regularization;
if regularization, disp(['Regularize the eigenvalue spectrum of omega''*omega with ',num2str(regularization)]);end

initialization = rmfield(p.Results, 'trainSet');
initialization.trainSet = [num2str(nb_samples),'x',num2str(nb_features),' matrix'];
initialization = rmfield(initialization, 'trainLab');
initialization.trainLab = ['vector of length ',num2str(length(trainLab))];
initialization.Lmin = p.Results.Lmin;
if ~isempty(testSet)
    initialization = rmfield(initialization, 'testSet');
    initialization.testSet = [num2str(size(testSet,1)),'x',num2str(size(testSet,2)),' matrix'];
end




% Display all arguments.
disp 'Settings for OGMLVQ:'
disp(initialization);

%%% check the number of prototypes per class if one integer is given and turn
%%% it into a vector
nb_ppc = p.Results.PrototypesPerClass;
if length(nb_ppc)~=nb_classes,
    nb_ppc = ones(1,nb_classes)*nb_ppc;
end

%%% initialize the prototypes
if isempty(p.Results.initialPrototypes)
    % initialize near the class centers
    w = zeros(sum(nb_ppc),nb_features);
    c_w = zeros(sum(nb_ppc),1);
    actPos = 1;
    for actClass=1:nb_classes
        nb_prot_c = nb_ppc(actClass);
        classMean = mean(trainSet(trainLab==classes(actClass),:),1);
        % set the prototypes to the class mean and add a random variation between -0.1 and 0.1
        w(actPos:actPos+nb_prot_c-1,:) = classMean(ones(nb_prot_c,1),:)+(rand(nb_prot_c,nb_features)*2-ones(nb_prot_c,nb_features))/10;
        %%
        c_w(actPos:actPos+nb_prot_c-1) = classes(actClass);
        actPos = actPos+nb_prot_c;
    end


%     for actClass=1:nb_classes
%         nb_prot_c = nb_ppc(actClass);
%         classMean = mean(trainSet(trainLab==classes(actClass),:),1);
%         % set the prototypes to the class mean
%         w(actPos:actPos+nb_prot_c-1,:) = classMean(ones(nb_prot_c,1),:);
%         %%
%         c_w(actPos:actPos+nb_prot_c-1) = classes(actClass);
%         actPos = actPos+nb_prot_c;
%     end


%     
        %%% initialize strategy changed by Fengzhen Tang 15/05/2015
        %%% randomly select 50% of samples of each class; compute mean value      
%      for actClass=1:nb_classes
%         nb_prot_c = nb_ppc(actClass);
%         A = find (trainLab == classes(actClass));
%         n_mean_components = floor(length(A)/2);
%         for j=1:nb_prot_c
%           shuffle = randperm (length (A));
%           B = trainSet (A (shuffle(1:n_mean_components)),:);
%           classMean = mean(B,1);
%           w(actPos,:) = classMean;
%           c_w(actPos) = classes(actClass);
%          
%           actPos = actPos + 1;
%         end
%       end

else
    % initialize with given w
    w = p.Results.initialPrototypes(:,1:end-1);
    c_w = p.Results.initialPrototypes(:,end);
end
%%% initialize the matrix
if isempty(p.Results.initialMatrix)
    if(p.Results.dim==nb_features)
        %omega = eye(nb_features);
        omega = eye(nb_features)/sqrt(nb_features);%%changed by Fengzhen 13/05/2015
    else % initialize with random numbers between -1 and 1
        omega = rand(dim,nb_features)*2-ones(dim,nb_features);
    end
else
    omega = p.Results.initialMatrix;
end
% normalize the matrix
omega = omega / sqrt(sum(diag(omega'*omega)));
model = struct('w',w,'c_w',c_w,'omega',omega);
clear w c_w omega;

sigma = 200/max(trainLab) + randn();
sigma1 = 80/max(trainLab) + randn();
sigma2 = 400/max(trainLab) + randn();


Lmin = p.Results.Lmin;
param.Lmin = Lmin;
param.sigma = sigma;
param.sigma1 = sigma1;
param.sigma2 = sigma2;
param.PrototypesPerClass = p.Results.PrototypesPerClass;
%%% gradient descent variables
nb_epochs = p.Results.nb_epochs;
% compute the vector of nb_epochs learning rates alpha for the prototype learning
if isa(p.Results.learningRatePrototypes,'function_handle')
    % with a given function specified from the user
    alphas = arrayfun(p.Results.learningRatePrototypes, 1:nb_epochs);
elseif length(p.Results.learningRatePrototypes)>2
    if length(p.Results.learningRatePrototypes)==nb_epochs
        alphas = p.Results.learningRatePrototypes;
    else
        disp('The learning rate vector for the prototypes does not fit the nb of epochs');
        return;
    end
elseif length(p.Results.learningRatePrototypes)==1%%added by Fengzhen Tang 15/05/2015
    alpha_start  = p.Results.learningRatePrototypes;
    alphas = alpha_start ./ (1+((1:nb_epochs)-1)*p.Results.decay_factor);
     
else
    % or use an decay with a start and a decay value
    if isempty(p.Results.learningRatePrototypes)
        initialization.learningRatePrototypes = [nb_features/1000, nb_features/100000];
    end
    alpha_start = initialization.learningRatePrototypes(1);
    alpha_end = initialization.learningRatePrototypes(2);
    alphas = arrayfun(@(x) alpha_start * (alpha_end/alpha_start)^(x/nb_epochs), 1:nb_epochs);
%     alphas = arrayfun(@(x) alpha_start / (1+(x-1)*alpha_end), 1:nb_epochs);
end
% compute the vector of nb_epochs learning rates epsilon for the Matrix learning
epsilons = zeros(1,nb_epochs);
if isa(p.Results.learningRateMatrix,'function_handle')
    % with a given function specified from the user
% 	epsilons = arrayfun(p.Results.learningRateMatrix, 1:nb_epochs);
    epsilons(MatrixStart:nb_epochs) = arrayfun(p.Results.learningRateMatrix, MatrixStart:nb_epochs);
elseif length(p.Results.learningRateMatrix)>2
    if length(p.Results.learningRateMatrix)==nb_epochs
        epsilons = p.Results.learningRateMatrix;
    else
        disp('The learning rate vector for the Matrix does not fit the nb of epochs');
        return;
    end
elseif length(p.Results.learningRateMatrix)==1%%added by Fengzhen Tang 15/05/2015
     eps_start = p.Results.learningRateMatrix;
     epsilons(MatrixStart:nb_epochs) = eps_start ./ (1+((MatrixStart:nb_epochs)-MatrixStart)*p.Results.decay_factor);
else
    % or use an decay with a start and a decay value
    if isempty(p.Results.learningRateMatrix)
        initialization.learningRateMatrix = [nb_features/10000, nb_features/1000000];
    end
    eps_start = initialization.learningRateMatrix(1);
    eps_end = initialization.learningRateMatrix(2);
%     epsilons = arrayfun(@(x) eps_start * (eps_end/eps_start)^(x/nb_epochs), 1:nb_epochs);
    epsilons(MatrixStart:nb_epochs) = arrayfun(@(x) eps_start * (eps_end/eps_start)^((x-MatrixStart)/(nb_epochs-MatrixStart)), MatrixStart:nb_epochs);
end

%%% initialize requested outputs
trainError = [];
costs = [];
testError = [];
if nout>=2,
    % train error requested
    trainError = ones(1,nb_epochs+1);
    estimatedLabels = GMLVQ_classify(trainSet, model); % error after initialization
    trainError(1) = sum( abs(trainLab - estimatedLabels) )/nb_samples;
    if nout>=3,
        % test error requested
        if isempty(testSet)
            testError = [];
            disp('The test error is requested, but no labeled test set given. Omitting the computation.');
        else
            testError = ones(1,nb_epochs+1);
            estimatedLabels = GMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
            testError(1) = sum( abs(testSet(:,end) - estimatedLabels) )/length(estimatedLabels);
        end        
        if nout>=4,
            % costs requested
%                 LabelEqPrototype = trainLab*ones(1,numel(model.c_w)) == (model.c_w*ones(1,nb_samples))';
            disp('The computation of the costs is an expensive operation, do it only if you really need it!');
            costs = ones(1,nb_epochs+1);
            costs(1) = OGMLVQ_costfun(trainSet, trainLab, model, param, regularization);
%                 costs(1) = sum(arrayfun(@(idx) GLVQ_costfun(min(dist(idx,model.c_w == trainLab(idx))),...
%                                                             min(dist(idx,model.c_w ~= trainLab(idx))))-regTerm, 1:size(dist,1)));
        end
    end
end

%     figure;
%     plotData(trainSet,trainLab);
%     plotPrototypes(model.w,model.c_w);
%     hold off;
PrototypesPerClass = p.Results.PrototypesPerClass;
%%% optimize with stochastic gradient descent
for epoch=1:nb_epochs
    if mod(epoch,100)==0, disp(epoch); end
    % generate order to sweep through the trainingset
    order = randperm(nb_samples);	
    % perform one sweep through trainingset
    for i=1:nb_samples
        % select one training sample randomly
        xi = trainSet(order(i),:);
        c_xi = trainLab(order(i));
        dist = sum((bsxfun(@minus, xi, model.w)*model.omega').^2, 2);

        %% find the correct prototype sets 

        %%%%%find the prototypes with correct labels 
        idxCorrectLab = find (abs(model.c_w - c_xi)<=Lmin);

        %%%%%%choose closetest one for each class
        nIdxCorrectLab= length(idxCorrectLab);
        nCorrect = nIdxCorrectLab/PrototypesPerClass;
        correct = zeros(nCorrect,1);
        distcorrect = zeros(nCorrect,1);

        pp = 1;
        for j = 1:PrototypesPerClass:nIdxCorrectLab
            idxPcorr = idxCorrectLab(j:j+PrototypesPerClass-1);
            [distcorrect(pp), idx] = min (dist(idxPcorr));
            correct(pp) = idxPcorr(idx);
         pp = pp+1;
        end


        %% find the wrong prototype sets

        %%%%%find the prototypes with wrong labels 
        idxWrongLab = find(abs(model.c_w - c_xi)> Lmin);
        %%%%% determine the wrong prototypes 
        distWrongAll = dist(idxWrongLab);
        Avg_dist=2*median(distWrongAll);         
        idx = distWrongAll<=Avg_dist;

    
       
        
        wrong = idxWrongLab(idx);
        distwrong= dist(wrong);
        %%%choose the same size of correct
        [sortedDistCorr,sortedIdxC] = sort(distcorrect);
        [sortedDistWrong,sortedIdxW] = sort(distwrong);
        nPairs  = min(length(distcorrect),length(distwrong));
        
        distcorrect  = sortedDistCorr(1:nPairs);
        distwrong = sortedDistWrong(1:nPairs);
     
        correct = correct(sortedIdxC);
        correct = correct(1:nPairs);
        wrong = wrong(sortedIdxW);
        wrong = wrong(1:nPairs);
   
        
        %%%weighting
        diffCorrect = (model.c_w(correct)-c_xi).^2;
        alpha_plus = exp(-diffCorrect/(2*sigma^2)  );
       
        y = max (abs (model.c_w - c_xi));
        diffWrong = (y-abs(c_xi - model.c_w(wrong))).^2;

        alpha_minus = exp(- diffWrong/(2*sigma1^2) )...
            .*exp(- distwrong/(2*sigma2^2));

        % prototype update 


        wJ = model.w(correct,:);
        wK = model.w(wrong,:);

        norm_correct = alpha_plus'*distcorrect;
        norm_wrong = alpha_minus'*distwrong;

        norm_factor = (norm_correct + norm_wrong)^2;

        DJ = bsxfun(@minus, xi, wJ);
        DK = bsxfun(@minus, xi, wK);

        oo = model.omega'*model.omega;

        wJ_factor = 2*alpha_plus*(norm_wrong/norm_factor);
        
        mu_minus = 2*alpha_minus*(norm_correct/norm_factor);
        
        wK_factor = mu_minus.*(1-distwrong/(2*sigma2^2));
        %wK_factor = mu_minus; 
        if sum(wK_factor <0)
            fprintf('wrong')
        end

        dwJ = bsxfun(@times,(2*wJ_factor)',(oo*DJ'));
        dwK = bsxfun(@times,(2*wK_factor)',(oo*DK'));

         model.w(correct,:) = wJ + alphas(epoch) * dwJ';
         model.w(wrong,:) = wK - alphas(epoch) * dwK';
%                 % update matrices
        if epsilons(epoch)>0, % epoch >= MatrixStart

            DJa =  bsxfun(@times,wJ_factor, DJ);
            DKa =  bsxfun(@times,wK_factor, DK);

            f1 = 2*(model.omega*DJ')*(DJa);
            f2 = 2*(model.omega*DK')*(DKa);


            % update omega
            if regularization,
                f3 = (pinv(model.omega))';                
            else
                f3 = 0;
            end
            %model.omega = model.omega-0.001 * (f1-f2  - regularization * f3);
            model.omega = model.omega-epsilons(epoch) * (f1-f2  - regularization * f3);
            % normalization
            model.omega = model.omega / sqrt(sum(diag(oo)));
        end

        %%%updating
        dSigma =  wJ_factor'*(distcorrect.*diffCorrect/(sigma^3));

        sigma = sigma - 10*alphas(epoch)*dSigma;

%         dSigma1 = wK_factor'*(distwrong.*diffWrong/sigma1^2);
% 
%         sigma1 = sigma1 - alphas(epoch)*dSigma1;
% 
%         dSigma2 = wK_factor'*(distwrong.*distwrong/sigma2^2);
% 
%         sigma2 = sigma2 - alphas(epoch)*dSigma2;


        dSigma1 = mu_minus'*(distwrong.*diffWrong/(sigma1^3));


        sigma1 = sigma1 + 10*alphas(epoch)*dSigma1;

        dSigma2 = mu_minus'*(distwrong.*distwrong/(sigma2^3));

        sigma2 = sigma2 + 10*alphas(epoch)*dSigma2;



    end
    %check the performance
    %[estimatedTrainLabels]= GMLVQ_classify(trainSet, model);
    %trainMAE(epoch) = mean( abs(trainLab - estimatedTrainLabels) );

    if nout>=2,
        % train error requested
        estimatedLabels = GMLVQ_classify(trainSet, model); % error after epoch
        trainError(epoch+1) = sum( abs(trainLab - estimatedLabels) )/nb_samples;
        if nout>=3,
            % test error requested
            if ~isempty(testSet)
                estimatedLabels = GMLVQ_classify(testSet(:,1:end-1), model); % error after initialization
                testError(epoch+1) = sum( abs(testSet(:,end) - estimatedLabels) )/length(estimatedLabels);
            end 
            if nout>=4,
                % costs requested
                costs(epoch+1) = OGMLVQ_costfun(trainSet, trainLab, model, param, regularization);
                %costs1(epoch+1) = GMLVQ_costfun(trainSet, trainLab, model, regularization);
%                     costs(epoch+1) = sum(arrayfun(@(idx) GLVQ_costfun(min(dist(idx,model.c_w == trainLab(idx))),...
%                                                                       min(dist(idx,model.c_w ~= trainLab(idx))))-regTerm, 1:size(dist,1)));
            end
        end
    end
end
%%% output of the training
varargout = cell(nout);
for k=1:nout
	switch(k)
		case(1)
			varargout(k) = {initialization};
		case(2)
			varargout(k) = {trainError};
		case(3)
			varargout(k) = {testError};
		case(4)
            varargout(k) = {costs};
	end
end
