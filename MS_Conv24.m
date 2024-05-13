function [dcnnLayers] = MS_Conv24(imageSize1,imageSize2)

    layer0 = [
              fullyConnectedLayer(38,'Name','fc')
              softmaxLayer('Name','softmax')
              classificationLayer('Name','classoutput')
             ];

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
     
	layer2 = [
            convolution2dLayer(7,8,'Padding',1,'Name','conv_6')        %48
            batchNormalizationLayer('Name','BN_6')
            reluLayer('Name','relu_6')
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_6')               %24

            convolution2dLayer(7,16,'Padding',1,'Name','conv_7')       %26
            fullyConnectedLayer(1000,'Name','fc2');
            ];
    
	layer3 = [
            convolution2dLayer(5,8,'Padding',1,'Name','conv_8')        %51
            batchNormalizationLayer('Name','BN_7')
            reluLayer('Name','relu_7')
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_7')               %25

            convolution2dLayer(5,16,'Padding',1,'Name','conv_9')       %24
            batchNormalizationLayer('Name','BN_8')
            reluLayer('Name','relu_8')
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_8')               %12

            convolution2dLayer(5,32,'Padding',1,'Name','conv_10')       %11
            batchNormalizationLayer('Name','BN_9')
            reluLayer('Name','relu_9')  
            maxPooling2dLayer(2,'Stride',2,'Name','maxpool_9')               %5

            convolution2dLayer(5,64,'Padding',1,'Name','conv_11')       %4
            fullyConnectedLayer(1000,'Name','fc3');
            ];
    
    lgraph = layerGraph(layer0);    
    lgraph = addLayers(lgraph,layer1);
    lgraph = addLayers(lgraph,layer2);
    lgraph = addLayers(lgraph,layer3);
    
    lgraph = connectLayers(lgraph,'input','conv_6');
    lgraph = connectLayers(lgraph,'input','conv_8');
    
    lgraph = connectLayers(lgraph,'fc2','add/in2');
    lgraph = connectLayers(lgraph,'fc3','add/in3');
    lgraph = connectLayers(lgraph,'add','fc');
    dcnnLayers = lgraph;
end
