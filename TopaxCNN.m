outputFolder = fullfile('CNNTOPAZ');
rootFolder = fullfile(outputFolder,'EVEN');
categories = {'MC', 'NC','SC'};
imds = imageDatastore(fullfile(rootFolder,categories),'LabelSource','foldernames');
tb1 = countEachLabel(imds);
minSetCount = min(tb1{:,2});
imds =splitEachLabel(imds,minSetCount,'randomize');
countEachLabel(imds);
MC = find(imds.Labels == 'MC', 1);
NC = find(imds.Labels == 'NC',1);
SC = find(imds.Labels == 'SC', 1);
figure
subplot(2,2,1);
imshow(readimage(imds,MC));
figure
subplot(2,2,2);
imshow(readimage(imds,NC));
figure
subplot(2,2,2);
imshow(readimage(imds,SC)); 

net = resnet50();
figure;
plot(net);
title('Architecture of ResNet-50');
set(gca, 'YLim', [150 170]);

net.Layers(1);
net.Layers(end);

numel(net.Layers(end).ClassNames);
[trainingSet, testSet] = splitEachLabel(imds,0.3,'randomize');
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet,...
    "ColorPreprocessing","gray2rgb");
augmentedTestSet = augmentedImageDatastore(imageSize, testSet,...
    "ColorPreprocessing","gray2rgb");
w1 = net.Layers(2).Weights;
w1 = mat2gray(w1);
figure
montage(w1)
title('First Convolutional Layer Weight')
featureLayer = 'fc1000';
trainingFeatures = activations(net, ...
    augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');

trainingLables = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLables, 'Learners', ....
    'linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

testFeatures = activations(net, ...
    augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');
predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');
testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictLabels);
conMat = bsxfun(@rdivide, confMat, sum(confMat,2));
mean(diag(confMat)) %accuracy
sum(confMat,2)

newImage = imread(fullfile('SC.tiff'));
ds = augmentedImageDatastore(imageSize, newImage,...
    "ColorPreprocessing","gray2rgb");

imageFeatures = activations(net, ...
    ds, featureLayer, 'MiniBatchSize', 32, 'OutputAs','columns');

label = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');
sprintf('The loaded image belongs to %s class', label)




