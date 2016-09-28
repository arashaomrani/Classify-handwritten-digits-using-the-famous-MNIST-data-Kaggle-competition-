clear;
load('XTrain-28x28.mat');
 X=D(:,:,1,1:400000);
Y=y(1:40000,:);
Y=categorical(Y);
Xcv=D(:,:,1,40001:42000);
Ycv=y(40001:42000,:);
Ycv=categorical(Ycv);


layers = [imageInputLayer([28 28 1]);
          convolution2dLayer(12,120,'stride',3,'padding',2);
          reluLayer();
          crossChannelNormalizationLayer(5);
          maxPooling2dLayer(3);
         convolution2dLayer(5,300,'stride',1,'padding',1);
         reluLayer();
         crossChannelNormalizationLayer(5)
          maxPooling2dLayer(3);  
          fullyConnectedLayer(1000);
          fullyConnectedLayer(10);
          softmaxLayer();
          classificationLayer()];
opts = trainingOptions('sgdm','MaxEpoch',160,'MiniBatchSize',200,...
                       'LearnRateSchedule','piecewise', 'LearnRateDropFactor',0.9,...
                       'LearnRateDropPeriod',10);
net = trainNetwork(X,Y,layers,opts);

YTest = classify(net,Xcv);

accuracy = sum(YTest == Ycv)/numel(Ycv)
