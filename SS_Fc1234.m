function [dcnnLayers] = SS_Fc1234(imageSize1,imageSize2)

    preNet = [
        imageInputLayer([imageSize1 imageSize2 1],'Name','input')

        convolution2dLayer(3,8,'Padding',1,'Name','conv_1')        %52
        batchNormalizationLayer('Name','BN_1')
        reluLayer('Name','relu_1')
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')               %26

        convolution2dLayer(3,16,'Padding',1,'Name','conv_2')       %26
        batchNormalizationLayer('Name','BN_2')
        reluLayer('Name','relu_2')
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')               %13

        convolution2dLayer(3,32,'Padding',1,'Name','conv_3')       %13
        batchNormalizationLayer('Name','BN_3')
        reluLayer('Name','relu_3')  
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')               %6

        convolution2dLayer(3,64,'Padding',1,'Name','conv_4')       %6
        batchNormalizationLayer('Name','BN_4')
        reluLayer('Name','relu_4')
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool_4')               %3

        convolution2dLayer(3,128,'Padding',1,'Name','conv_5')     %3
        batchNormalizationLayer('Name','BN_5')
        reluLayer('Name','relu_5')
        maxPooling2dLayer(2,'Stride',2,'Name','maxpool_5')               %1

        dropoutLayer(0.5,'Name','dropout_2')
        fullyConnectedLayer(1000,'Name','fc0')
        additionLayer(6,'Name','add')
        ];

    lgraph = layerGraph(preNet);
    layer0 = [
              fullyConnectedLayer(38,'Name','fc')
              softmaxLayer('Name','softmax')
              classificationLayer('Name','classoutput')
             ];
    layer1 = fullyConnectedLayer(1000,'Name','fc1');
    layer2 = fullyConnectedLayer(1000,'Name','fc2');
    layer3 = fullyConnectedLayer(1000,'Name','fc3');
    layer4 = fullyConnectedLayer(1000,'Name','fc4');
    layer5 = fullyConnectedLayer(1000,'Name','fc5');

    lgraph = addLayers(lgraph,layer0);
    lgraph = addLayers(lgraph,layer1);
    lgraph = addLayers(lgraph,layer2);
    lgraph = addLayers(lgraph,layer3);
    lgraph = addLayers(lgraph,layer4);
    lgraph = addLayers(lgraph,layer5);

    lgraph = connectLayers(lgraph,'conv_1','fc1');
    lgraph = connectLayers(lgraph,'conv_2','fc2');
    lgraph = connectLayers(lgraph,'conv_3','fc3');
    lgraph = connectLayers(lgraph,'conv_4','fc4');
    lgraph = connectLayers(lgraph,'conv_5','fc5');
    lgraph = connectLayers(lgraph,'fc1','add/in2');
    lgraph = connectLayers(lgraph,'fc2','add/in3');
    lgraph = connectLayers(lgraph,'fc3','add/in4');
    lgraph = connectLayers(lgraph,'fc4','add/in5');
    lgraph = connectLayers(lgraph,'fc5','add/in6');
    lgraph = connectLayers(lgraph,'add','fc');
    dcnnLayers = lgraph;
end

