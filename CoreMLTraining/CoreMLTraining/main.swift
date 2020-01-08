//
//  main.swift
//  CoreMLInference
//
//  Created by Jacopo Mangiavacchi on 12/22/19.
//  Copyright © 2019 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

let coreMLFilePath = "/Users/jacopo/S4TF_CoreML_Test/Models/s4tf_model_personalization.mlmodel"
let retrainedCoreMLFilePath = "/Users/jacopo/S4TF_CoreML_Test/Models/s4tf_model_retrained.mlmodelc"


func compileCoreML(path: String) -> (MLModel, URL) {
    let modelUrl = URL(fileURLWithPath: path)
    let compiledUrl = try! MLModel.compileModel(at: modelUrl)
    
    print("Compiled Model Path: \(compiledUrl)")
    return try! (MLModel(contentsOf: compiledUrl), compiledUrl)
}

func inferenceCoreML(model: MLModel, x: Float) -> Float {
    let inputName = "dense_input"
    
    let multiArr = try! MLMultiArray(shape: [1], dataType: .double)
    multiArr[0] = NSNumber(value: x)

    let inputValue = MLFeatureValue(multiArray: multiArr)
    let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue]
    let provider = try! MLDictionaryFeatureProvider(dictionary: dataPointFeatures)
    
    let prediction = try! model.prediction(from: provider)

    return Float(prediction.featureValue(for: "output")!.multiArrayValue![0].doubleValue)
}

func generateData(sampleSize: Int = 100) -> ([Float], [Float]) {
    let a: Float = 2.0
    let b: Float = 1.5
    
    var X = [Float]()
    var Y = [Float]()

    for i in 0..<sampleSize {
        let x: Float = Float(i) / Float(sampleSize)
        let noise: Float = (Float.random(in: 0..<1) - 0.5) * 0.1
        let y: Float = (a * x + b) + noise
        X.append(x)
        Y.append(y)
    }
    
    return (X, Y)
}

func prepareTrainingBatch() -> MLBatchProvider {
    var featureProviders = [MLFeatureProvider]()

    let inputName = "dense_input"
    let outputName = "output_true"
    
    let (X, Y) = generateData()
             
    for (x,y) in zip(X, Y) {
        let multiArr = try! MLMultiArray(shape: [1], dataType: .double)

        multiArr[0] = NSNumber(value: x)
        let inputValue = MLFeatureValue(multiArray: multiArr)

        multiArr[0] = NSNumber(value: y)
        let outputValue = MLFeatureValue(multiArray: multiArr)
         
        let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue,
                                                           outputName: outputValue]
         
        if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
            featureProviders.append(provider)
        }
    }
     
    return MLArrayBatchProvider(array: featureProviders)
}

func updateModelCompletionHandler(updateContext: MLUpdateContext) {
    print("CoreML Error: \(updateContext.task.error.debugDescription)")
    
    let updatedModel = updateContext.model
    let updatedModelURL = URL(fileURLWithPath: retrainedCoreMLFilePath)
    try! updatedModel.write(to: updatedModelURL)
    
    print("Model Trained!")
    print("Press return to continue..")
}

func train(url: URL) {
//    let configuration = MLModelConfiguration()
//    configuration.parameters = [MLParameterKey.epochs : 100]
    
    let updateTask = try! MLUpdateTask(forModelAt: url,
                                       trainingData: prepareTrainingBatch(),
                                       configuration: nil, //configuration,
                                       completionHandler: updateModelCompletionHandler)

    updateTask.resume()
}



print("Compile CoreML model")
let (coreModel, compiledModelUrl) = compileCoreML(path: coreMLFilePath)

print("CoreML model")
print(coreModel.modelDescription)

print("CoreML inference")
let prediction = inferenceCoreML(model: coreModel, x: 1.0)
print(prediction)

print("CoreML Start Training")
train(url: compiledModelUrl)

let _ = readLine()

print("Load CoreML Retrained Model")
let retrainedModel = try! MLModel(contentsOf: URL(fileURLWithPath: retrainedCoreMLFilePath))

print("CoreML inference")
let prediction2 = inferenceCoreML(model: retrainedModel, x: 1.0)
print(prediction2)


print("done!")
