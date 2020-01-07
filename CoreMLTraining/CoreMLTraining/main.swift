//
//  main.swift
//  CoreMLInference
//
//  Created by Jacopo Mangiavacchi on 12/22/19.
//  Copyright © 2019 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

func compileCoreML(path: String) -> (MLModel, URL) {
    let modelUrl = URL(fileURLWithPath: path)
    let compiledUrl = try! MLModel.compileModel(at: modelUrl)
    
    print("Compiled Model Path: \(compiledUrl)")
    return try! (MLModel(contentsOf: compiledUrl), compiledUrl)
}

func inferenceCoreML(model: MLModel, x: Float) -> Float {
    class s4tf_modelInput : MLFeatureProvider {

        /// dense_input as 1 by 1 matrix of doubles
        var dense_input: MLMultiArray

        var featureNames: Set<String> {
            get {
                return ["dense_input"]
            }
        }
        
        func featureValue(for featureName: String) -> MLFeatureValue? {
            if (featureName == "dense_input") {
                return MLFeatureValue(multiArray: dense_input)
            }
            return nil
        }
        
        init(dense_input: MLMultiArray) {
            self.dense_input = dense_input
        }
    }

    let multiArr = try! MLMultiArray(shape: [1], dataType: .double)
    multiArr[0] = NSNumber(value: x)

    let input = s4tf_modelInput(dense_input: multiArr)

    let prediction = try! model.prediction(from: input)

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
        let inputValue = MLFeatureValue(double: Double(x))
        let outputValue = MLFeatureValue(double: Double(y))
         
        let dataPointFeatures: [String: MLFeatureValue] = [inputName: inputValue,
                                                           outputName: outputValue]
         
        if let provider = try? MLDictionaryFeatureProvider(dictionary: dataPointFeatures) {
            featureProviders.append(provider)
        }
    }
     
    return MLArrayBatchProvider(array: featureProviders)
}

func updateModelCompletionHandler(updateContext: MLUpdateContext) {
    print(updateContext.task.error)
    
    let updatedModel = updateContext.model
    let updatedModelURL = URL(fileURLWithPath: "/Users/jacopo/S4TF_CoreML_Test/Models/s4tf_model_retrained.mlmodel")
    try! updatedModel.write(to: updatedModelURL)
    
    print("Model Trained!")
}

func train(url: URL) {
    let updateTask = try! MLUpdateTask(forModelAt: url,
                                       trainingData: prepareTrainingBatch(),
                                       configuration: nil,
                                       completionHandler: updateModelCompletionHandler)

    updateTask.resume()
}


let coreMLFilePath = "/Users/jacopo/S4TF_CoreML_Test/Models/s4tf_model_personalization.mlmodel"

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

print("done!")
