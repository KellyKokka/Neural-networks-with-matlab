train1=dlmread('1trn.SSV');
train2=dlmread('2trn.SSV');
train3=dlmread('3trn.SSV');
train4=dlmread('4trn.SSV');
train5=dlmread('5trn.SSV');
train6=dlmread('6trn.SSV');
train7=dlmread('7trn.SSV');
test1=dlmread('1tst.SSV');
test2=dlmread('2tst.SSV');
test3=dlmread('3tst.SSV');
test4=dlmread('4tst.SSV');
test5=dlmread('5tst.SSV');
test6=dlmread('6tst.SSV');
test7=dlmread('7tst.SSV');

train1(train1==255) = 1;
train2(train2(:,17)==255)=1;
train3(train3(:,17)==255)=1;
train4(train4(:,17)==255)=1;
train5(train5(:,17)==255)=1;
train6(train6(:,17)==255)=1;
train7(train7(:,17)==255)=1;
test1(test1(:,17)==255)=1;
test2(test2(:,17)==255)=1;
test3(test3(:,17)==255)=1;
test4(test4(:,17)==255)=1;
test5(test5(:,17)==255)=1;
test6(test6==255)=1;
test7(test7==255)=1;
A = rescale(train1(:,1:16),0,1)
%normalizeData=mat2gray(train1(:,1:16));
normilized = train1(:,1:16)/norm(train1(:,1:16));

x = transpose(train1(:,1:16));
t = transpose(train1(:,17));

net = feedforwardnet(10);
net.trainParam.epochs = 100;
net.trainParam.lr = 0.3;
net.trainParam.mc = 0.6;
[net,tr] = train(net,x,t);