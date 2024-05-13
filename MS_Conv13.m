function [dcnnLayers] = MS_Conv13(imageSize1,imageSize2)

    layer1 = [
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
        fullyConnectedLayer(1000,'Name','fc1');
        additionLayer(3,'Name','add')
        ];
    
    lgraph = layerGraph(layer1);
    layer0 = [
              fullyConnectedLayer(38,'Name','fc')
              softmaxLayer('Name','softmax')
              classificationLayer('Name','classoutput')
             ];
         
	layer2 = [
            convolution2dLayer(7,8,'Padding',1,'Name','conv_6')        %52
            fullyConnectedLayer(1000,'Name','fc2');
            ];
    
	layer3 = [
            convolution2dLayer(5,8,'Padding',1,'Name','conv_7')        %52
            batchNormalizationLayer('Name','BN_6')
            reluLayer('Name','relu_6')
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_6')               %26

            convolution2dLayer(5,16,'Padding',1,'Name','conv_8')       %26
            batchNormalizationLayer('Name','BN_7')
            reluLayer('Name','relu_7')
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_7')               %13

            convolution2dLayer(5,32,'Padding',1,'Name','conv_9')       %13
            fullyConnectedLayer(1000,'Name','fc3');
            ];
    
    lgraph = addLayers(lgraph,layer0);
    lgraph = addLayers(lgraph,layer2);
    lgraph = addLayers(lgraph,layer3);
    
    lgraph = connectLayers(lgraph,'input','conv_6');
    lgraph = connectLayers(lgraph,'input','conv_7');
    
    lgraph = connectLayers(lgraph,'fc2','add/in2');
    lgraph = connectLayers(lgraph,'fc3','add/in3');
    lgraph = connectLayers(lgraph,'add','fc');
    dcnnLayers = lgraph;
end
