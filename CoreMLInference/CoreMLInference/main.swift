//
//  main.swift
//  CoreMLInference
//
//  Created by Jacopo Mangiavacchi on 12/22/19.
//  Copyright © 2019 Jacopo Mangiavacchi. All rights reserved.
//

import Foundation
import CoreML

print("Compile CoreML model")
let modelUrl = URL(fileURLWithPath: "/Users/jacopo/S4TF_CoreML_Test/s4tf_model.mlmodel")
let compiledUrl = try MLModel.compileModel(at: modelUrl)
let coreModel = try MLModel(contentsOf: compiledUrl)

print(coreModel.modelDescription)

let multiArr = try! MLMultiArray(shape: [1, 1], dataType: .double)
multiArr[0] = NSNumber(value: 1.0)

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



let input = s4tf_modelInput(dense_input: multiArr)

let prediction = try! coreModel.prediction(from: input)

print(prediction.featureValue(for: "Identity")!.multiArrayValue![0].doubleValue)
