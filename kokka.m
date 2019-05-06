rng(20)

%read test and train data
train1 = dlmread('1trn.SSV');
test1 = dlmread('1tst.SSV');
train2 = dlmread('2trn.SSV');
test2 = dlmread('2tst.SSV');
train3 = dlmread('3trn.SSV');
test3 = dlmread('3tst.SSV');
train4 = dlmread('4trn.SSV');
test4 = dlmread('4tst.SSV');
train5 = dlmread('5trn.SSV');
test5 = dlmread('5tst.SSV');
train6 = dlmread('6trn.SSV');
test6 = dlmread('6tst.SSV');
train7 = dlmread('7trn.SSV');
test7 = dlmread('7tst.SSV');

%change classification value
train1(train1(:,17)==255,17) = 1;
test1(test1(:,17) == 255,17) = 1;
train2(train2(:,17)==255,17) = 1;
test2(test2(:,17) == 255,17) = 1;
train3(train3(:,17)==255,17) = 1;
test3(test3(:,17) == 255,17) = 1;
train4(train4(:,17)==255,17) = 1;
test4(test4(:,17) == 255,17) = 1;
train5(train5(:,17)==255,17) = 1;
test5(test5(:,17) == 255,17) = 1;
train6(train6(:,17)==255,17) = 1;
test6(test6(:,17) == 255,17) = 1;
train7(train7(:,17)==255,17) = 1;
test7(test7(:,17) == 255,17) = 1;

 %resampling
 
 train1 = SMOTE(train1,900,3);
 train2 = SMOTE(train2,900,3);
 train3 = SMOTE(train3,900,3);
 train4 = SMOTE(train4,900,3);
 train5 = SMOTE(train5,900,3);
 train6 = SMOTE(train6,900,3);
 train7 = SMOTE(train7,900,3);
%  test1 = SMOTE(test1,900,3);
%  test2 = SMOTE(test2,900,3);
%  test3 = SMOTE(test3,900,3);
%  test4 = SMOTE(test4,900,3);
%  test5 = SMOTE(test5,900,3);
%  test6 = SMOTE(test6,900,3);
%  test7 = SMOTE(test7,900,3);
 
%normilize data
 train1(:,1:16) = normalize(train1(:,1:16));
 test1(:,1:16) = normalize(test1(:,1:16));
 train2(:,1:16) = normalize(train2(:,1:16));
 test2(:,1:16) = normalize(test2(:,1:16));
 train3(:,1:16) = normalize(train3(:,1:16));
 test3(:,1:16) = normalize(test3(:,1:16));
 train4(:,1:16) = normalize(train4(:,1:16));
 test4(:,1:16) = normalize(test4(:,1:16));
 train5(:,1:16) = normalize(train5(:,1:16));
 test5(:,1:16) = normalize(test5(:,1:16));
 train6(:,1:16) = normalize(train6(:,1:16));
 test6(:,1:16) = normalize(test6(:,1:16));
 train7(:,1:16) = normalize(train7(:,1:16));
 test7(:,1:16) = normalize(test7(:,1:16));

%all data 
matrix = [train1; train2; train3; train4; train5;train6; train7];
matrixtest = [test1; test2; test3; test4; test5; test6; test7];

x = transpose(matrix(:,1:16));
t = transpose(matrix(:,17));

%trainlm--Levenberg-Marquardt
%trainrp--Rprop
%trainscg--Scaled conjugate gradients

net = patternnet([20 20],'trainlm');
% net = feedforwardnet([20 20],'trainrp');
net.layers{1}.transferFcn = 'logsig';
net.layers{2}.transferFcn = 'logsig';

net.trainParam.epochs = 110; %max. epochs /iterations
net.trainParam.lr = 0.3; %learning rate
net.trainParam.mc = 0.6; % momentum constant

net.divideParam.trainRatio = 0.9;
net.divideParam.valRatio = 0.1;
net.divideParam.testRatio = 0;

[net,tr] = train(net,x,t);
%  testX = matrix(:,1:16)';
%  testT = matrix(:,17)';
 testX = matrixtest(:,1:16)';
 testT = matrixtest(:,17)';
testY = net(testX);

testIndices = vec2ind(testY);
plotconfusion(testT,testY);
[c,cm] = confusion(testT,testY);
fprintf('Percentage Correct Classification : %f%%\n' ,100*(1-c));
fprintf('Percentage Incorrect Classification : %f%%\n' ,100*c);

% smote -> resampling function

function newset = SMOTE(oldset,N,K)
    newset = oldset;
    T = sum(oldset(:,end)==1);
    if N < 100
        T = (N/100) * T;
        N=100;
    end
    numattrs = size(oldset,2)-1;
    sample = oldset(oldset(:,end) == 1,:);
    N = round(N/100);
    for i = 1:T
        dist = sqrt(sum((oldset - sample(i,:)).^2,2))';
        [~,index] = sort(dist,'ascend');
        nnarray = index(1:K);
        J = N;
        while J ~= 0    
            nn = randi(K);
            temp = [];
            for attr = 1:numattrs
                dif = oldset(nnarray(nn),attr) - sample(i,attr);
                gap = rand();
                temp = [temp (sample(i,attr)+gap*dif)];
            end
            temp = [temp 1];
            newset = [newset ; temp];
            J = J-1;
 

        end
    end
    newset = newset(randi(size(newset,1),[1,length(newset)]),:);
end