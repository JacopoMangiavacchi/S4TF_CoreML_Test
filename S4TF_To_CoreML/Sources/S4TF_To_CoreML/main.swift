import Foundation
import TensorFlow
import SwiftProtobuf

func generateSampleData(size: Int = 100) -> (Tensor<Float>, Tensor<Float>) {
    let a: Float = 2.0
    let b: Float = 1.5
    var x = Tensor<Float>(rangeFrom: 0, to: 1, stride: 1.0 / Float(size))
    let noise = (Tensor<Float>(randomNormal: [size]) - 0.5) * 0.1
    var y = (a * x + b) + noise
    
    x = x.reshaped(toShape: [100, 1]) //size
    y = y.reshaped(toShape: [100, 1]) //size
    
    return (x, y)
}

struct LinearRegression: Layer {
    var layer1 = Dense<Float>(inputSize: 1, outputSize: 1, activation: identity)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return layer1(input)
    }
}

func trainS4TF(model: inout LinearRegression, x: Tensor<Float>, y: Tensor<Float>, epoch: Int) {
    let optimizer = SGD(for: model, learningRate: 0.03)
    Context.local.learningPhase = .training
    for _ in 0..<epoch {
        let 𝛁model = model.gradient { r -> Tensor<Float> in
            let ŷ = r(x)
            let loss = meanSquaredError(predicted: ŷ, expected: y)
            print("Loss: \(loss)")
            return loss
        }
        optimizer.update(&model, along: 𝛁model)
    }
}

func inferenceS4TF(model: LinearRegression, x: Tensor<Float>) -> Tensor<Float> {
    Context.local.learningPhase = .inference
    let score = model(x)
    return score.reshaped(toShape: [100])
}

func getWeightAndBias(model: LinearRegression) -> (Float, Float) {
    return (Float(model.layer1.weight[0][0])!, Float(model.layer1.bias[0])!)
}

func convertToCoreML(weigths: Float, bias: Float) -> CoreML_Specification_Model {
    return CoreML_Specification_Model.with {
        $0.specificationVersion = 4
        $0.description_p = CoreML_Specification_ModelDescription.with {
            $0.input = [CoreML_Specification_FeatureDescription.with {
                $0.name = "dense_input"
                $0.type = CoreML_Specification_FeatureType.with {
                    $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                        $0.shape = [1, 1]
                        $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                    }
                }
            }]
            $0.output = [CoreML_Specification_FeatureDescription.with {
                $0.name = "Identity"
                $0.type = CoreML_Specification_FeatureType.with {
                    $0.multiArrayType = CoreML_Specification_ArrayFeatureType.with {
                        $0.dataType = CoreML_Specification_ArrayFeatureType.ArrayDataType.double
                    }
                }
            }]
            $0.metadata = CoreML_Specification_Metadata.with {
                $0.userDefined = ["coremltoolsVersion" : "3.1"]
            }
        }
        $0.neuralNetwork = CoreML_Specification_NeuralNetwork.with {
            $0.layers = [CoreML_Specification_NeuralNetworkLayer.with {
                $0.name = "Identity"
                $0.input = ["dense_input"]
                $0.output = ["Identity"]
                $0.inputTensor = [CoreML_Specification_Tensor.with {
                    $0.rank = 2
                    $0.dimValue = [1, 1]
                }]
                $0.outputTensor = [CoreML_Specification_Tensor.with {
                    $0.rank = 2
                    $0.dimValue = [1, 1]
                }]
                $0.batchedMatmul = CoreML_Specification_BatchedMatMulLayerParams.with {
                    $0.weightMatrixFirstDimension = 1
                    $0.weightMatrixSecondDimension = 1
                    $0.hasBias_p = true
                    $0.weights = CoreML_Specification_WeightParams.with {
                        $0.floatValue = [weigths]
                    }
                    $0.bias = CoreML_Specification_WeightParams.with {
                        $0.floatValue = [bias]
                    }
                }
            }]
            $0.arrayInputShapeMapping = CoreML_Specification_NeuralNetworkMultiArrayShapeMapping.exactArrayMapping
            $0.imageInputShapeMapping = CoreML_Specification_NeuralNetworkImageShapeMapping.rank4ImageMapping
        }
    }
}



print("Generate Data")
let (x, y) = generateSampleData()

print("Train S4TF Model")
var s4tfModel = LinearRegression()
trainS4TF(model: &s4tfModel, x: x, y: y, epoch: 1000)

print("Weight, Bias")
let (weight, bias) = getWeightAndBias(model: s4tfModel)
print(weight, bias)

print("Test S4TF inference")
let score = inferenceS4TF(model: s4tfModel, x: x)
print(score)

print("Convert to CoreML")
let coreMLModel = convertToCoreML(weigths: weight, bias: bias)
let binaryModelData: Data = try coreMLModel.serializedData()
let modelUrl = URL(fileURLWithPath: "./s4tf_model.mlmodel")
try? binaryModelData.write(to: modelUrl)
