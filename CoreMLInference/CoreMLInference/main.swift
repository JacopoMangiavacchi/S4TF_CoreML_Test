//
//  main.swift
//  CoreMLInference
//
//  Created by Jacopo Mangiavacchi on 12/22/19.
//  Copyright © 2019 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

func compileCoreML(path: String) -> MLModel {
    let modelUrl = URL(fileURLWithPath: path)
    let compiledUrl = try! MLModel.compileModel(at: modelUrl)
    return try! MLModel(contentsOf: compiledUrl)
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

    let multiArr = try! MLMultiArray(shape: [1, 1], dataType: .double)
    multiArr[0] = NSNumber(value: x)

    let input = s4tf_modelInput(dense_input: multiArr)

    let prediction = try! model.prediction(from: input)

    return Float(prediction.featureValue(for: "Identity")!.multiArrayValue![0].doubleValue)
}


let coreMLFilePath = "/Users/jacopo/S4TF_CoreML_Test/s4tf_model.mlmodel"

print("Compile CoreML model")
let coreModel = compileCoreML(path: coreMLFilePath)

print("CoreML model")
print(coreModel.modelDescription)

print("CoreML inference")
let prediction = inferenceCoreML(model: coreModel, x: 1.0)
print(prediction)
