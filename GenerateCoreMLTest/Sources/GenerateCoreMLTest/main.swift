import Foundation
import SwiftProtobuf

print("Hello, world!")

let coreModel = CoreML_Specification_Model.with {
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
                    $0.floatValue = [1.0]
                }
                $0.bias = CoreML_Specification_WeightParams.with {
                    $0.floatValue = [1.0]
                }
            }
        }]
        $0.arrayInputShapeMapping = CoreML_Specification_NeuralNetworkMultiArrayShapeMapping.exactArrayMapping
        $0.imageInputShapeMapping = CoreML_Specification_NeuralNetworkImageShapeMapping.rank4ImageMapping
    }
}


