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


func train(url: URL) {
    let configuration = MLModelConfiguration()
    configuration.computeUnits = .all
    //configuration.parameters = [.epochs : 100]

    let progressHandler = { (context: MLUpdateContext) in
        switch context.event {
        case .trainingBegin:
            print("Training begin")

        case .miniBatchEnd:
            break
//            let batchIndex = context.metrics[.miniBatchIndex] as! Int
//            let batchLoss = context.metrics[.lossValue] as! Double
//            print("Mini batch \(batchIndex), loss: \(batchLoss)")

        case .epochEnd:
            let epochIndex = context.metrics[.epochIndex] as! Int
            let trainLoss = context.metrics[.lossValue] as! Double
            print("Epoch \(epochIndex) end with loss \(trainLoss)")

        default:
            print("Unknown event")
        }

//        print(context.model.modelDescription.parameterDescriptionsByKey)

//        do {
//            let multiArray = try context.model.parameterValue(for: MLParameterKey.weights.scoped(to: "dense_1")) as! MLMultiArray
//            print(multiArray.shape)
//        } catch {
//            print(error)
//        }
    }

    let completionHandler = { (context: MLUpdateContext) in
        print("Training completed with state \(context.task.state.rawValue)")
        print("CoreML Error: \(context.task.error.debugDescription)")

        if context.task.state != .completed {
            print("Failed")
            return
        }

        let trainLoss = context.metrics[.lossValue] as! Double
        print("Final loss: \(trainLoss)")

        
        let updatedModel = context.model
        let updatedModelURL = URL(fileURLWithPath: retrainedCoreMLFilePath)
        try! updatedModel.write(to: updatedModelURL)
        
        print("Model Trained!")
        print("Press return to continue..")
    }

    let handlers = MLUpdateProgressHandlers(
                        forEvents: [.trainingBegin, .miniBatchEnd, .epochEnd],
                        progressHandler: progressHandler,
                        completionHandler: completionHandler)

    
    
    
    
    let updateTask = try! MLUpdateTask(forModelAt: url,
                                       trainingData: prepareTrainingBatch(),
                                       configuration: configuration,
                                       progressHandlers: handlers)

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
