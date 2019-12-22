import Foundation
import SwiftProtobuf

print("Hello, world!")

let coreModel = CoreML_Specification_Model.with {
    $0.specificationVersion = 4
    $0.description_p = CoreML_Specification_ModelDescription.with {
        $0.input = [CoreML_Specification_FeatureDescription.with {
            $0.name = "dense_input"
            
        }]
        $0.output = [CoreML_Specification_FeatureDescription.with {
            $0.name = "Identity"

            
        }]
//         $0.metadata.with {
//             $0.userDefined
//         }
    }
}


